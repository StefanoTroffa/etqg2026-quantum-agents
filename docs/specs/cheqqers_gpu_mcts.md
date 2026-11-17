# Cheqqers: GPU network-guided search (track 33)

Goal: an agent that uses the policy-value network on the GPU at play time and beats the shipped agent
(C++ alpha-beta with fitted weights, "FW") in a series. Assumption (Stefano, 2026-11-14): the tournament machine
has a CUDA GPU.

## Design
- Search: MCTS with chance nodes in C++ (native/cheqqers). Decision nodes use PUCT with the network's move prior;
  measurement nodes average children by their outcome probability (expectation, not sampling, when the outcome
  count is small). Terminal values exact.
- Batching: leaf collection with virtual loss in C++; one ctypes call returns a batch of encoded positions, Python
  runs the network on the GPU (torch, fp16 ok), a second call writes priors and values back. No Python loop over
  tree nodes.
- Network: src/models/cheqqers_policy_value.py (move logits over legal moves + WDL), initialised from
  results/cheqqers_pv/run1_default_474k/model.pt, retrained on FW self-play (~1.6 M lines in selfplay_fw).
- RL loop (as many iterations as time allows): self-play with MCTS + current net on GPU -> targets = visit
  distribution and game result -> retrain -> gate.
- Time: budget from settings.timeout as today (2,200 ms at 3,000). Warm-up must run one GPU move per game before
  the public port opens.

## Gates (in order, stop at the first failure)
1. Correctness: MCTS legal moves on all arena_1113 positions; chance-node expectation checked against Python
   get_move_outcomes; no errors in 1,000 self-play games.
2. Offline proxy (~/work/_scratch/qg_cpp/valnet2/tools/proxy.py, 1,500 positions, 10 s FW reference): must not be
   worse than FW at equal wall time. Note its bias toward FW; it can only reject.
3. Series vs FW under PLAN_rules (colour-balanced pairs, seed 1114, 2,200 ms, looks at 100/150/200 pairs, 95% CI
   one side), frozen library + net file.

## Shipping rule
- Decision (Stefano, 2026-11-14): the RL agent (network + MCTS on the GPU) is the primary submission. The organiser
  confirmed a GPU (>=4 GB) and repo + uv; GPU/CPU may still be decided on the day for fairness.
- Ships unless it clearly loses gate 3 (prespecified 95% upper bound of the pair-level score below 50%); only then
  FW stays the default. FW remains the automatic CPU fallback in every case.
- If shipped: CUDA missing, the net failing to load, or warm-up slower than budget -> the server falls back to FW
  and says so on stdout. FW path stays byte-identical.
- Schedule (Stefano, 2026-11-14): train until Tue 17 Nov 14:00 CET, then freeze; series vs FW done by Tue 18:00;
  ship by 21:00. Interim 40-pair match vs FW Monday ~20:00 (informational, training continues).
