# Cheqqers RL recipe (track 33) - replaces the ad-hoc REINFORCE line

Reviewed 2026-11-14 by an independent model (Codex, read-only review of the code; answer in
~/work/_scratch/qg_cpp/codex_rl/answer.md). Sources: Danihelka et al. 2022 (Gumbel AlphaZero/MuZero), Silver et al.
2018 (AlphaZero), Antonoglou et al. 2022 (Stochastic MuZero, chance nodes), Anthony et al. 2017 (Expert Iteration).
Recipe = warm-started expert iteration with exact stochastic MCTS, FW supervision, matches vs FW early.

## 0. Decisive first measurement (before building more)
DAgger best (proxy 0.671) inside the batched MCTS at 2,200 ms vs FW, 40 colour-balanced pairs + vs random.
If clearly inferior (e.g. score < 0.35), the priority becomes targeted DAgger (labels on disagreements/tactical
failures) and policy-assisted FW move ordering; uncontrolled self-play will not rescue the deadline.

## 1. Fix the move head (bug)
src/models/cheqqers_policy_value.py forward(): move_vec = sum over slots of (h_square + move_pos[slot]). The sum
erases order: a->b and b->a (and permutations of 3-square moves) get the same vector. Count colliding legal-move
pairs on the data, replace with ordered slot projections (concat of the k slot vectors, zero-padded, or a
per-slot Linear), retrain the DAgger best with the fixed head, re-measure proxy + vs random. The C++ side
(value_net / encoder) is unaffected unless it scores moves.

## 2. Search for self-play
- Exact simulator; chance by expectation Q(s,a) = sum_o p_o V(s'_o) (mcts.cpp already does this; keep player
  perspective across multi-jumps).
- Root: Gumbel-Top-k without replacement, m = 16 candidates + Sequential Halving, 64 simulations/move (compare
  128). Interior: deterministic Gumbel selection from completed Q. Must be verified (gate-1 style tests).
- If Gumbel is not correct within ~6-8 h of work: existing PUCT with 128-256 simulations and visit-count targets.
- 512-2,048 independent trees batched; don't spend a tree's simulations on concurrent unevaluated paths.
- Throughput reality: ~40k evals/s -> at most ~51k games/h at 64 evals x 44 plies (less with chance expansion).

## 3. Targets
- Policy: Gumbel: cross-entropy to softmax(logits + sigma(completedQ)) incl. the paper's handling of unvisited
  actions; PUCT fallback: normalised visits.
- Value: WDL cross-entropy to the realised game result from the stored player's perspective, gamma = 1. No mixing
  with the root search value.
- FW teacher labels (1-2.2 s) concentrated on positions where net and FW disagree and on tactical failures.
  Never softmax alpha-beta root scores (they are bounds after the first move).

## 4. Training
- Buffer: 1M-position sliding window over several generations, uniform sampling; FW-labelled positions 20% of
  each batch at first, decaying to 5-10%. Duplicates aggregated with counts (keep outcome diversity).
- Generation: 5-10k games, then ~one pass over new-data-equivalent samples.
- AdamW lr 1e-4 (3e-5 if panel performance deteriorates), batch 1,024, wd 1e-4, grad clip 1, bf16.
- Network: d256 / 6 layers kept; distil smaller only if measured inference savings justify it.
- Opponents during generation: 70% current self-play, 20% frozen checkpoints, 10% FW.
- REINFORCE run3 is stopped (it cannot detect cycling; its +103 vs parent was -112 vs FW).

## 5. Evaluation
Every 2-4 h: fixed panel = FW 2,200 ms (40 pairs), the imitation parent, two frozen checkpoints; report each score
and panel Elo. Random play is a smoke test only. Play-time config: benchmark 800 / 3,200 / 12,800 simulations,
use the strongest that fits 2.2 s, with a deadline and warmed CUDA graphs.

## 6. Schedule and ship rule
- Hours 0-8: correctness, throughput, MCTS-vs-FW screening (step 0), move-head fix.
- Hours 8-40: improve the strongest measured configuration.
- Freeze Tue 17 Nov 14:00 CET; series vs FW by 18:00 (games in parallel; a serial 400-game series would take
  ~11 h): fresh paired openings, swapped colours, recorded chance seeds, pair-level CI, zero forfeits; ships unless
  it clearly loses (prespecified 95% upper bound below 50%; Stefano 2026-11-14: the RL agent is the primary), no
  repeated peeking beyond PLAN_rules' looks. FW is the CPU fallback.
- Self-play FW opponent (Stefano 2026-11-14): the play-time budget, 2,200 ms, in a separate generator process so GPU
  self-play never waits on it.
