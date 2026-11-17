# 🏁 Centralized Index of Concluded Experiments

> **Public version.** This is the team's working log, kept as it was written. Some entries refer to files that are
> not in this repository: internal process notes (`.agent/`, `records/`, `docs/agents/gates.md`), cluster account
> plans, and the organisers' harness (`workshop_resources/`, available at github.com/VincenzoLipardi/ETQG_2026).
> Other teams are named only as "first team", "second team".


This document registers and indexes the completed research experiments, hyperparameter search phases, and calibration runs for Quantum Tic-Tac-Toe and Checkers. 

---

## 📊 Summary of Active Tracks

| Track ID | Track Name | Active Model | Calibration Focus | Best Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Track-G** | Genetic Baseline Evolution | SNN / MLP / KAN | Baseline actor-critic generation | SNN: 76.0% vs Random, 33% vs MCTS |
| **Track-C** | Symmetric Value Calibration | SNN / MLP / KAN | Value head MSE fitting | SNN: Preserves 55% Greedy win rate |
| **Track-A** | Asymmetric Calibration (MLP) | Decoupled AC MLP | Asymmetric Player 1 & 2 tuning | MLP P1: 65% vs Random |
| **Track-K** | Decoupled Frozen Calibration (KAN) | Decoupled KAN | Policy freezing vs. Joint updates | KAN Frozen: 2.3x MCTS win rate |

---

## 🔍 Concluded Experiments Index

### 1. Track-G: Genetic Baseline Evolution (Tic-Tac-Toe)
* **Goal**: Evolve raw neural network weights on Quantum Tic-Tac-Toe (level 2 entanglement) using a genetic algorithm.
* **Outcome**: Generated baseline policies that achieved high win rates vs. Random but struggled against lookahead search (MCTS) due to uncalibrated value prediction.
* **Checkpoint References**:
  * `results/q_net_quantum_entanglement_mlp_genetic.pth`
  * `results/q_net_quantum_entanglement_kan_genetic.pth`
  * `results/q_net_quantum_entanglement_snn_genetic.pth`
* **Log Reference**: `results/training_run_genetic.log`

### 2. Track-C: Symmetric Value Calibration (Tic-Tac-Toe)
* **Goal**: Calibrate value heads of symmetric models on rolling self-play games.
* **Outcome**: Confirmed SNN architecture maintains robust representation retention (76.0% vs Random, 55.0% vs Greedy, 33.0% vs MCTS).
* **Checkpoint References**:
  * `results/q_net_quantum_entanglement_mlp_calibrated.pth`
  * `results/q_net_quantum_entanglement_kan_calibrated.pth`
  * `results/q_net_quantum_entanglement_snn_calibrated.pth`
* **Log Reference**: `results/training_run_calibration_tictactoe_kan.log`

### 3. Track-A: Asymmetric Calibration (MLP Tic-Tac-Toe)
* **Goal**: Optimize asymmetric Player 1 (X) and Player 2 (O) parameters separately using curriculum training and policy-preserving regularization.
* **Variants**:
  * **Exp A1 (Curriculum Variant 4)**: Asymmetric training with decaying policy regularization.
  * **Exp A2 (Goal-oriented Joint)**: Decoupled AC backbones with 3-dimensional WDL value heads.
* **Outcome**: Goal-oriented decoupled training yielded Player 1 peak performance (65% win rate vs Random).
* **Checkpoint References**:
  * `results/q_net_quantum_entanglement_tictactoe_mlp_p1_best_joint.pth`
  * `results/q_net_quantum_entanglement_tictactoe_mlp_p2_best_joint.pth`
* **Log Reference**: `results/goal_optimization_run.log`

### 4. Track-K: Decoupled Frozen Calibration (KAN Tic-Tac-Toe)
* **Goal**: Test if freezing the KAN policy parameters entirely during value calibration prevents policy collapse.
* **Outcome**: **Frozen-Policy Calibration** protected the policy (drawing 15/15 vs Greedy) while training the value function, improving the Q-MCTS win rate vs expert MCTS by **2.3x** (from 10% to 23.3%).
* **Checkpoint References**:
  * `results/q_net_quantum_entanglement_tictactoe_kan_best_frozen.pth`
  * `results/q_net_quantum_entanglement_tictactoe_kan_calibrated_frozen.pth`
* **Log Reference**: `results/pipeline_decoupled_wdl.log`

---

*Detailed hyperparameter settings and evaluations are tracked in [tictactoe_calibration.csv](file:///home/stefano-troffa/Personal_proj/quantum_games_rl_v2/experiments/tictactoe_calibration.csv).*

---

## 🏁 Final submission tracks (tournament re-run, deadline 15 November 2026)

Registered per `records/agentic_development_stream.md`: hypothesis and acceptance first, outcome when the run
ends. Settings are the tournament's unless stated: TiqTaqToe `quantumness=2, decoherence=0`; Cheqqers
`boardSize=6, quantumness=2`. Commands and raw results: `.agent/tasks/final-submission-2026-11-15/verification.md`
(V-numbers). Contracts: `docs/specs/final_submission_agents.md`.

| Track ID | Track Name | Status | Result |
| :--- | :--- | :--- | :--- |
| **Track-X** | Exact tables as the TiqTaqToe agent | concluded 10 Nov | 97 of 97 moves from tables, max 2.4 ms; 17 of 20 vs random |
| **Track-N** | Cheqqers: value-network blend vs pure search | to rerun (series artefact, see Track-L) | Before the fix: network side 13.5 of 40 |
| **Track-E** | TiqTaqToe: does deviating from optimal play pay? | concluded 10 Nov | vs random: OPT 0.9578, SAFE 0.9723, BR 0.9852 (BR guarantees only 0.125) |
| **Track-D** | Cheqqers: larger search budget | concluded 10 Nov, not met | Fixed code: 24 W, 1 D, 15 L for depth 8 / 2.0 s; Elo +80 (95% CI -27 to +203) |
| **Track-H** | Cheqqers: exact endgame | planned, not started | Waits for a state-count estimate |
| **Track-L** | Cheqqers: losses to the random player | concluded 10 Nov | Losses to random 10 of 100 unfixed; 2 of 100 and 1 of 20 (harness) after three fixes; the rest is evaluation |
| **Track-R** | Both games against real opponents (arena) | planned, needs the arena go-ahead | – |

### 5. Track-X: Exact tables as the TiqTaqToe agent
* **Hypothesis**: At the tournament setting the game is solved (Chris's tables, PR #6), so a lookup answers every
  position optimally and far inside the 5 s limit.
* **Acceptance**: every answer in a harness match comes from the tables (no fallback), every move legal, latency
  under 100 ms, more wins than losses against the harness's random player over 10 rounds per seat.
* **Outcome**: met. 9-0 as X and 8-1 as O; 97 of 97 answers exact; median 1.3 ms, max 2.4 ms (V3). Registered
  after the run: the adapter was built on 10 Nov before this track existed.
* **Not shown**: that exact play beats the earlier search agent. Search scored 19 of 20 in a separate 20-game run;
  the two runs are too small and not paired.

### 6. Track-N: Cheqqers value-network blend vs pure search
* **Hypothesis**: Blending `q_net_official_cheqqers_size6_q2_best.pth` into the leaf evaluation (blend 0.25,
  depth 4) makes the search agent stronger than the heuristic alone at equal budget.
* **Acceptance**: lower end of the Elo 95% interval above 0 over 40 colour-balanced games.
* **Outcome**: rejected. 13 W, 1 D, 26 L for the network side; Elo -117 (95% CI -251 to -11); no timeouts or
  illegal moves on either side (V5). The server default was changed to pure search in the working tree
  (`CHEQQERS_NN_BLEND=0`); the change awaits confirmation (decision D6).

### 7. Track-E: TiqTaqToe, does deviating from optimal play pay?
* **Hypothesis** (Stefano, meeting of 1 Nov): against opponents that are not optimal, a policy that deviates from
  the optimal one scores more. Chris's counter: any deviation is by definition suboptimal and carries risk.
* **Method**: exact evaluation over the solved tables of four policies (OPT, SAFE, BR, BR@WC) against an opponent
  that plays a uniformly random move with probability eps (`exact/tiqtaqtoe/exploit.py`).
* **Acceptance**: SAFE is worth adding if its mean score at eps = 1 exceeds OPT's by at least 0.01
  (placeholder threshold). BR is reported for its cost only.
* **Outcome at Q1** (70,384 states, 38 s; mean score over both seats, win 1, tie or draw 1/2):

  | eps | OPT | SAFE | BR | BR against its worst opponent |
  |---|---|---|---|---|
  | 1.0 | 0.9807 | 0.9838 | 0.9926 | 0.4688 |
  | 0.5 | 0.8363 | 0.8459 | 0.8777 | 0.4688 |
  | 0.2 | 0.6580 | 0.6669 | 0.6820 | 0.4688 |
  | 0.0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

  All of the difference is in the second seat; moving first the three policies score the same.
* **Outcome at Q2, the tournament setting** (1,643,542 states, 1,074 s; `exact/results/exploit-q2-d0-m1.json`):

  | eps | OPT | SAFE | BR | BR against its worst opponent |
  |---|---|---|---|---|
  | 1.0 | 0.9578 | 0.9723 | 0.9852 | 0.1250 |
  | 0.5 | 0.7964 | 0.8250 | 0.8433 | 0.3438 |
  | 0.2 | 0.6370 | 0.6549 | 0.6655 | 0.4844 |
  | 0.0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

  At eps = 0 the run reproduces the table's root (first seat 23/32 win, 1/4 loss, 1/32 draw). Against the random
  player SAFE gains 0.0145 over OPT (first seat 0.9789 to 0.9871, second 0.9368 to 0.9574): the acceptance
  threshold of 0.01 is met. Against half-random and mostly-optimal opponents the gain is 0.029 and 0.018. BR gains
  about as much again but its guaranteed score falls from 0.5 to 0.125. Both readings of the 1 November meeting
  hold: deviating pays (Stefano), and only the value-preserving kind is free of risk (Chris).
* **Limit**: each SAFE column is tuned to its own eps. A deployed SAFE must assume one; how much a wrong
  assumption costs (SAFE built for eps = 1 played against eps = 0.2) is not measured yet.
* **Status**: concluded for the question asked. Building SAFE into the server is decision D8, not taken.

### 8. Track-D: Cheqqers, larger search budget
* **Hypothesis**: The arena allows 3 s; depth cap 8 with 2.2 s beats the current depth cap 5 with 1.2 s.
* **Acceptance**: lower end of the Elo 95% interval above 0 over 40 colour-balanced games, no timeouts. The budget
  is only adopted after the arena's network latency is measured (spec step C2).
* **Outcome**: met in self-play. 30 W, 0 D, 10 L for the larger budget (13-7 as first player, 17-3 as second);
  Elo +191 (95% CI +81 to +356); no timeouts or illegal moves; its moves took 1.98 s at the median and 2.00 s at
  most, against 7 ms median and 1.21 s at most for the current setting (V8). Not adopted: 2.0 s of search leaves
  1.0 s plus the 500 ms allowance for the arena's network, which is not measured yet (spec step C2), and the series
  is against our own agent, not other teams'.

* **Rerun on the fixed code** (depth 8 / 2.0 s against depth 5 / 1.2 s, 40 games, seed 7, the committed command):
  24 W, 1 D, 15 L for the larger budget (14-5-1 as first player, 10-10 as second); Elo +80 (95% CI -27 to +203);
  no timeouts or illegal moves; its slowest move 1.84 s. The interval includes zero: acceptance not met. The
  30-10 of the first run was inflated by the series artefact of Track-L. The larger budget is probably better but
  40 games do not show it, and it is not adopted.

### 9. Track-H: Cheqqers, exact endgame
* **Hypothesis** (meeting of 1 Nov): positions with fewer than about 10 pieces can be solved exactly and looked up.
* **Acceptance**: not set. First a count of reachable states for k pieces including superpositions, and a rule for
  repeated positions and history-dependent draws (spec section 3).
* **Outcome**: not started.

### 10. Track-R: both games against real opponents
* **Why**: Tracks E and D measured our candidates against a random player, a mixed stand-in and ourselves. No
  other team's agent is available to run locally (the organisers' public repository holds only the starter kit).
  The only sample of real opponents is the arena: live pairings 12:30-14:30 CET on 10-13 November, and its public
  game records, which cover every team's games, not only ours.
* **Hypotheses**
  1. TiqTaqToe: the other teams' agents play moves below the optimal value often enough that SAFE's gain over OPT
     in Track-E applies (their measured rate of non-optimal moves is nearer eps = 0.2-1 than 0).
  2. Cheqqers: the larger budget of Track-D also scores better against other teams and never times out through
     the arena's tunnel.
* **Method**
  * TiqTaqToe, from the records: for each team, the share of its moves whose exact value is below the best and the
    value lost per move (the tables answer this for any recorded position); for each position we faced, whether
    SAFE would have played another move than OPT (`exploit.moves_at`). One connector per token means only one
    policy plays live at a time.
  * Cheqqers, live: the current budget for one part of the sessions and the larger one for the other, same
    opponents; score per opponent, forfeits, round-trip latency per move. From the records: every recorded
    position replayed through `OfficialCheqqersEnv`, legal moves compared (spec step C1).
* **Acceptance**
  * Adopt SAFE if at least one team other than the house bot plays more than 1 move in 10 below optimal value
    (placeholder) and Chris agrees.
  * Adopt the larger Cheqqers budget if it has zero forfeits, its worst round trip stays under 2.8 s
    (placeholder: 3.0 s plus the 500 ms allowance, minus a margin), and its score against other teams is not below
    the current budget's.
  * Sample sizes will be small (two games per pairing). Report counts, not rates, when a cell has fewer than 20
    games.
* **Outcome**: not started. Connecting to the arena is gated (`docs/agents/gates.md`).

### 11. Track-L: Cheqqers, losses to the random player
* **Why**: the pre-flight of 10 November (`scripts/preflight_cheqqers.sh`, 5 games per colour on the organisers'
  harness) won 9 and lost 1. The final run needs wins over the random baseline in 4 rounds; a 1-in-10 loss rate
  fails that about one time in three.
* **Finding**: in the lost game the agent, down to its last piece, split it next to an enemy piece. Cause: the
  transposition table stored forced-capture nodes under the extended depth and reused them one ply short
  (`records/lessons-learned.md`, lesson 13). Fixed in `src/search/expecti_cheqqers.py`; regression tests in
  `tests/search/test_expecti_cheqqers.py` (failed before the fix, pass after).
* **Hypothesis**: with the fix the agent loses to the random player in fewer than 2 of 100 games at the server
  setting (depth 5, 1.2 s), against a baseline measured on the unfixed code with the same seed.
* **Acceptance**: fixed losses at most 2 of 100 and not more than the baseline's; every remaining loss read and
  classified (search bug, evaluation, or chance). Then a new pre-flight on the harness: 10 of 10.
* **Open**: the static evaluation does not treat "our last piece exists with probability 1/2" as half a loss, so
  scores swing between even and odd depths (-2 against -31 for the same move). Not addressed by this fix.
* **Baseline, unfixed code** (depth 5, 1.2 s, 100 games, seed 11): 90 wins, 10 losses (7 as first player, 3 as
  second), no timeouts, and 1 move on which the search raised. Cause of the raise: the simulator refuses joint
  states above 32,768 dimensions ("QuantumLimitExceeded", five entangled queens); in `server.py` that exception
  sends the request to the June code path. Second fix, same file: a move the simulator cannot expand is scored as
  the position it starts from and counted (`unexpanded_branches`), the search goes on. Regression tests with the
  recorded payload as fixture (`tests/search/fixtures/cheqqers_entangled_queens_payload.json`).
* **Pre-flight after the first fix**: 10 of 10 on the organisers' harness, no forfeits, slowest move 1.09 s.
* **After the first fix** (same settings and seed): 94 wins, 6 losses (1 as first player, 5 as second). Above the
  target. In the lost games the agent bled material from the first moves; in game 19 its first move as Black was
  the one a clean search scores worst at every depth (-32 against -0.4), answered in 39 ms.
* **Third cause, in how series are run, not in the server**: the hub keeps one player object for a whole series,
  and the player kept its transposition table between moves. The table's values are scores for the root colour of
  the call that stored them; in the next game the player has the other colour and reads them with the wrong sign.
  Third fix, same file: the table is emptied at the start of every `get_move()` (what `server.py` already did by
  building a player per request). Regression test `TestPlayerReuse`.
* **What this invalidates**: every series run through the hub before 12:46 on 10 November mixed this artefact
  into both sides: Track-N (network blend), Track-D (budget) and the baselines above. Their numbers stand as
  measured but are not evidence for their conclusions until rerun. Track-D is being rerun; Track-N is not yet.
* **Series against the random player, depth 5 / 1.2 s unless stated, 100 games each, seed 11**:

  | Code | Wins | Losses | Search errors |
  |---|---|---|---|
  | Unfixed | 90 | 10 | 1 |
  | Unfixed, depth 8 / 2.0 s | 92 | 8 | 0 |
  | Fix 1 (table depth) | 94 | 6 | 1 |
  | Fixes 1-2 (plus simulator limit) | 96 | 4 | 0 |
  | Fixes 1-3 (plus table per move) | 98 | 2 | 0 |

* **Outcome**: acceptance met at its edge: 2 losses of 100 (1 as each colour), no timeouts, no search errors,
  slowest move 1.10 s. Both remaining losses are even trades down to one or two pieces followed by a lost
  endgame in which our last pieces were split or half-captured: the open evaluation point above, not a search
  bug. A harness pre-flight of 10 games per colour on the fixed server is the last check of this track.
* **Harness pre-flight, fixed server, 10 games per colour**: 19 wins, 1 loss, no forfeits, no fallback, slowest
  move 1.09 s. In the lost game the agent was down to about 1.5 pieces against two queens and three pawns.
* **Classification of the two series losses** (each move replayed with a 20 s budget, depth cap 8): in game 3
  the long search plays the same move in 14 of 16 positions, including every move of the lost endgame (splitting
  its last pieces while behind); in game 80 it differs in 5 of 10. So more time alone does not remove these
  losses: the evaluation prefers trades into small endgames and splits of the last pieces. Next track: an
  evaluation term for the risk of holding the last pieces in superposition, measured by the same 100-game series.
* **Status**: concluded. Loss rate against the random player about 10% before, 2 of 100 and 1 of 20 after.

---

## 🖥️ Cluster tracks (plans of 11 November 2026: `docs/specs/alice_training_plan.md`)

| Track ID | Track Name | Status | Where it runs |
| :--- | :--- | :--- | :--- |
| **Track-V** | Cheqqers: evaluation weights | planned | stefano-pc now (plan A); ALICE job array once an account works (plan B) |
| **Track-G** | Cheqqers: value network retrained against the fixed search | planned | one GPU: ALICE `gpu-l4-24g` or the RTX 5090 |
| **Track-P** | Paper-grade series for the write-up | planned, after 15 Nov | ALICE job arrays (plan C) |

### 12. Track-V: Cheqqers, evaluation weights
* **Hypothesis**: the losses left after Track-L come from the hand-set evaluation, first of all the bonus for a
  piece in superposition (`0.12 * (1 - p)` per square in `OfficialCheqqersEnv.evaluate_state`, the function the
  search uses; an earlier version of this entry named the fallback function), which pays for pieces that cannot
  block and can be captured,
  and from the absence of any cost for holding the last pieces split. Removing the bonus or making it a penalty,
  and adding a last-pieces term, lowers the loss rate against the random player without losing to the current
  evaluation.
* **Method**: weights as data (`EvalWeights`), series at fixed depth 5 with no time cut, same seeds for every
  candidate. Stage 1 one change at a time (9 candidates, 400 games against random, 200 against the default);
  stage 2 the passing changes combined (1,000 games against random). On ALICE the same as a grid of 729 vectors.
* **Acceptance**: stage 1, fewer losses to random than the default on the same seeds and the Elo 95% interval
  against the default above 0; stage 2, at most 1% losses to random and no search errors; then 40 of 40 on the
  organisers' harness. Thresholds are placeholders. "No candidate passes" is a valid outcome: keep today's.
* **Outcome**: not started. Needs prerequisites 1-3 of the plan.

### 13. Track-G: Cheqqers, value network retrained
* **Hypothesis**: a value network trained by the league against the fixed search and the Track-V evaluation adds
  strength at equal budget. Track-N's "the network loses" was measured with the series artefact and is void.
* **Acceptance**: against pure search at equal depth, Elo 95% interval above 0 over at least 400 games, and no
  more losses to random than pure search. Inference stays on CPU inside the move budget.
* **Outcome**: not started. Needs prerequisite 4 (the trainer without the RTX 5090 settings) and, first, a rerun
  of Track-N with the existing checkpoint on the fixed code.

### 14. Track-P: paper-grade series
* **Purpose**: the numbers for the write-up due 8 December: the submitted agents against the random player, the
  June agent, depth variants and the network variant, 1,000 games a pairing, both colours, fixed seeds.
* **Acceptance**: every number in the write-up comes from a committed command and a stored per-game file.
* **Outcome**: not started; after the submission.

### 15. Track-U: Cheqqers rules of 11 November 2026 (compliance, not an experiment)
* **Trigger**: Vincenzo's mail of 11 November (received 13 November): deadline moved to Tuesday 17 November
  23:59, a new Drive folder, and an updated `local-cheqqers-server.js`. The file differs from the 30 October one
  in four places: a slide never passes an enemy piece; squares passed over on a slide or a capture are measured
  and must be empty; a landing or passed-over square that turns out occupied, or a piece to be jumped that turns
  out absent, fails the move; a capture with nothing to capture does nothing. It is not our own
  `local-cheqqers-server_v3.js` experiment (104 lines differ).
* **Finding**: `src/env/cheqqers_official.py` implemented none of it. A new differential test that lets the
  harness play random games and hands every position to the simulator showed, against the new file, wrong legal
  moves in 80 of 1,979 positions and a wrong result in 11; against the 30 October file, 2 and 4 of 1,987, so the
  earlier "0.0% divergence" in the README was not exact either.
* **Fix**: the four changes ported (`find_moves`, `get_involved_squares`, the measurement loops of `apply_move`
  and `get_move_outcomes`, `_apply_take_move`). After it, over 900 harness games on three seeds (28,000
  positions): the harness's board is always among the simulator's outcomes; legal moves differ in 2 positions
  and the side to move after a capture in 1, all three from piece identities the payload does not carry.
  Recorded-position tests fail on the old simulator (6 of 9) and pass on the new one.
* **Consequence**: every Cheqqers series before 13 November (Tracks N, D, L) was played on the 30 October rules,
  in the simulator on both sides. Their relative results may carry over; their numbers are not measurements of
  the game the tournament plays. Rerun what a decision depends on.
* **Agent on the new harness** (organisers' file of 11 November, engine 1.0.2, against its random player):
  before the port 9 of 10; after the port: see V10.
* **Fifth difference, found the same day by the fit's data generation** (it stopped after 820 games): the engine
  refuses to entangle more than 15 squares ("QuantumLimitExceeded"). The harness applies a move inside a
  try/catch, so a refused move leaves the position as the measurements made it and the same side to move. The
  simulator raised instead: in a series that ends the run, in the search it used to end the move (Track-L).
  Checked against the harness with the engine's limit lowered to four squares on both sides: 1,096 of 3,298
  moves refused, the turn stays in every one; the simulator now does the same (differential tests
  `TestEngineLimitAgainstHarness`). The search keeps its own guard for a simulator that raises.

---

## 🎯 Best-agent tracks (13-17 November 2026; comparison and contracts: `docs/specs/cheqqers_best_agent.md`)

All on the rules of 11 November (Track-U). Thresholds marked placeholder are first guesses, to be argued, not tuned.

| Track ID | Track Name | Status |
| :--- | :--- | :--- |
| **Track-S** | Series that can be trusted; arena latency; time budget | step 1 done 13 Nov (paired openings, per-game records, pair-by-pair summary; identical agents score exactly 50%); latency and budget open |
| **Track-W** | Faster search without changing its moves | step 1 built 13 Nov: per-position cache, 1.5 times faster at fixed depth, same moves and position counts; the 200-position check of the acceptance still to run |
| **Track-V2** | Evaluation fitted to our own games (replaces Track-V's hand sweep) | in progress 13 Nov: weights are data (defaults reproduce the old values on 82 recorded positions), fit tool tested, 20,000 depth-2 games generating |
| **Track-Q** | Search efficiency: PVS, aspiration windows, forced captures never evaluated statically | steps 1 (quiescence) and 2 (three-square moves near the root) built 13 Nov, both off by default, neither separated from the plain search in 20 to 45 pairs |
| **Track-B** | Value network on search targets | planned, after the submission unless the rest is done |
| **Track-K** | Against the June winner's public agent | done 16 Nov: 38 W / 1 D / 1 L in 40 games (organisers' harness, real start, 3 s a move) |
| **Track-T** | What thinking time is worth: the same agent at 2.2 s against itself at 0.55 s | done 13 Nov: +38 Elo for four times the time (-21 to +99, 60 pairs) |
| **Track-O** | Opening book: a long search's choice for the first positions of a game | built 13 Nov, no book made; the review says the games are not lost in the opening |
| **Track-M** | Move quality: an agent's choices on recorded decisions against a long-search judge | first table 13 Nov: orders the known pairs; quiescence gives away least |

### 16. Track-S: trusted series, latency, budget
* **Hypothesis 1**: with random two-ply openings each played once per colour and the same chance seed for both
  games of a pair, a series of 400 games resolves differences of about 35 Elo; today's single start position
  repeats the same few games.
* **Acceptance 1**: a series is reproduced exactly from its seed; two identical agents score 50% within the
  interval over 200 games.
* **Outcome 1 (13 November)**: built (`--opening-plies`, `--out`, `summarize`). Two identical depth-2 agents
  over 100 pairs with two-ply openings: 100 pairs split 1-1, score 0.500 with no spread at all, as it must be
  when both games of a pair share the opening and the chance stream and the agents are the same; the series is
  reproduced game for game from its seed (test). Read pair by pair, a series therefore measures the agents and
  not the luck; the per-game interval the older summary prints (+-49 Elo here) overstates the noise. Two-ply
  openings give only 44 distinct starts in 100 pairs: use four plies. White won 126 of the 200 games: moving
  first is worth about 63% at depth 2. Acceptance met.
* **Hypothesis 2**: the arena's round trip leaves room for more than 1.2 s of search.
* **Acceptance 2**: budget = 3000 - 500 - worst measured latency - 300 ms; the larger budget beats 1.2 s in a
  paired series (lower end of the Elo 95% interval above 0) and has zero timeouts in an arena session.

### 17. Track-W: faster search, same moves
* **Hypothesis**: a third of the search time recomputes square probabilities from amplitudes (profile of
  13 November: 3,800 positions a second). Caching them per position object is worth at least 1.5 times the speed.
* **Acceptance**: on 200 recorded positions at fixed depth 4 the chosen move and its score are identical before
  and after; the differential and rules tests pass; positions per second at least 1.5 times (placeholder).
* **Outcome, step 1 (13 November)**: `OfficialCheqqersEnv` can keep, per position, every square's probabilities
  and the legal moves (`cache_enabled`, off unless asked for); every method that changes a position empties it.
  The search switches it on for its own copy of the position (`use_cache`, on by default). Measured on 24
  positions from arena games at fixed depth 3: 57,113 positions visited with and without, the same move each
  time, 1,440 a second without and 2,158 with: 1.50 times. At a time limit the deepest iteration started moves
  from 5.25 to 5.42 plies on average at 1.2 s and from 5.75 to 6.00 at 2.2 s: a quarter of a ply. With the cache
  forced on everywhere (`QG_CHEQQERS_CACHE=1`) the differential test against the harness passes over 600 games
  and every arena game on record replays to the arena's result. Two tests fail in that forced mode and are
  expected to: one edits squares by hand and then asks for moves, which is why the cache is off by default; the
  other asserts that default. Guard: `tests/search/test_cheqqers_search_cache.py`.
* **Still owed**: the acceptance as written (200 positions, depth 4, move and score); the test checks 29 positions
  at depths 2 and 3 and compares the move and the count of positions, not the score.
* **Step 2 (13 November)**: the squares' probabilities stay cached across changes that touch only the pieces'
  properties, the turn or the open capture sequence (they were recomputed 88 times per applied move). Same
  moves and position counts on the benchmark; with the cache forced on, the differential test passes over 300
  games and all 951 arena games on record replay. The gain is about a tenth on a loaded machine and has to be
  measured again on a quiet one.
* **Ruled out (13 November)**: PyPy 3.11. Six middlegame positions at fixed depth 4, the same 8,097 positions
  visited: 3,600 to 4,100 a second on the project's CPython 3.13, 2,100 a second on PyPy after warm-up.
* **Where the time goes now** (profile, cache on): applying a move and enumerating its outcomes 54% (of which
  the end-of-game check 20% and move generation for the capture-sequence rule), evaluation 15%, move ordering
  10%, copying positions 8%.

### 18. Track-V2: evaluation fitted to our own games
* **Hypothesis**: the hand-set constants are far from what results support (the superposition bonus, the queen at
  2.6, no cost for a piece the enemy can capture). A logistic regression of the game result on presence-weighted
  features, from 20,000+ fast self-play games on the new rules, beats the hand-set evaluation at equal depth.
* **Method**: `EvalWeights` as data with today's values as defaults; one `features()` function shared by the fit
  and the evaluation; positions with a capture pending and the first plies excluded (as Scan does); output as a
  win probability, which also tightens the bounds the chance-node pruning uses.
* **Acceptance**: against the current default over 400 paired games, lower end of the Elo 95% interval above 0;
  no more losses to the random player over 400 games than the default; 20 of 20 on the organisers' harness.
  "No fit passes" is a valid outcome.
* **First look, 584 games only (pipeline check, not a result)**: held-out log-loss material alone 0.452, the
  hand-set evaluation 0.455, fitted 0.445: on this sample the hand-set evaluation predicts results no better
  than counting pieces. The fit halves the superposition bonus, lowers the queen to about 1.9, and puts its
  weight on the share of material and on pieces the opponent can capture. Positions with a capture pending are
  left out, so the three features that are zero in quiet positions (`threat_mover`, `multi_jump`,
  `exposed_waiter`) cannot be fitted that way: either they keep their hand-set values or the search stops
  evaluating such positions (Track-Q). To be decided on the full data.
* **Fits of 13 November** (generation stopped at 9,944 depth-2 games to free the machine for the arena):

  | Data | Held-out log-loss on self-play: material, hand-set, fitted | Log-loss on 93 arena games between agents (2,626 quiet positions; hand-set 0.5735, material 0.5833) |
  |---|---|---|
  | 5,732 games, all positions | 0.5275, 0.5173, 0.4995 | 0.5662 |
  | 9,944 games, all positions (`results/eval_fit/r2_all_squash.json`) | 0.5356, 0.5246, 0.5132 | 0.5705 |
  | 9,944 games, quiet positions only (`r2_quiet_squash.json`) | 0.5210, 0.5186, 0.5096 | 0.5691 |

  The fits agree with each other from 5,700 games on: queen 2.2 to 2.4 men (hand-set 2.6), centre 0.9 (0.14),
  squared advancement 1.0 to 1.3 (0.25), back rank 0.5 to 0.6 (0.18), share of material 1.3 to 1.4 (not in the
  hand-set evaluation), a piece the opponent can capture -0.3 to -0.9 (not in it), queens on the long diagonals
  -0.45 (hand-set +0.22 and +0.10).
* **Scored on real games** (`fit_evaluation.py import-arena`, `score`): every fit predicts the results of the
  arena's games between agents a little better than the hand-set evaluation, by about as much as the hand-set
  one beats counting pieces. 93 games, positions of one game are not independent: a direction, not a result.
* **In play**: the 5,732-game fit against the hand-set evaluation at depth 2, four-ply openings, 200 pairs
  (400 games): score 0.534, Elo +23 (95% interval -6 to +54); 43 pairs won 2-0, 31 lost 0-2, 115 split. The
  acceptance (lower end above 0) is missed by a little; the series at 2,200 ms is still to be played.
* **Status**: open; the server plays the hand-set weights.

### 19. Track-Q: search efficiency
* **Hypothesis**: principal-variation search, aspiration windows and not evaluating statically while a capture is
  forced give the same or a better move with fewer positions.
* **Acceptance**: fewer positions for the same depth on the recorded positions; the series of Track-V2's kind
  against the version without them.

### 20. Track-B: value network on search targets
* **Hypothesis**: trained on the scores of deeper searches (TreeStrap, Descent) instead of game outcomes, a small
  network evaluated in NumPy adds strength at equal time.
* **Acceptance**: equal-time paired series, lower end of the Elo 95% interval above 0. Not started.

### 21. Track-K: against the June winner
* **Purpose**: the only real opponent available outside the arena: the public agent that won Cheqqers in June
  (`michaelkoelle/cheqqers-ai`), through the organisers' harness of 11 November (`scripts/harness_match.sh`).
* **Acceptance**: none; a measurement. At least 20 games per colour, forfeits counted.
* **Status**: blocked. Starting third-party code was refused in the session of 13 November; Stefano can start it.

### 22. Track-R, first data: the arena session of 12 November (other teams' games)
* **Source**: the arena's public record file, 502 Cheqqers games of Thursday 12 November (rules version
  d691637b68), fetched 13 November 12:22. Each game has the moves, the measurement outcomes the arena drew, the
  time the arena measured per move, and the result.
* **Who played**: two teams and the house bot (we were not connected).

  | Pairing | Result |
  |---|---|
  | The new seventh team vs house bot | 432 won, 6 lost (216 of 219 with each colour) |
  | One of the June teams vs house bot | 57 won, 1 lost |
  | Those two against each other | 2 - 3 - 1 draw over 6 games (one ended by an unreachable agent) |

* **Move times measured by the arena** (network included): 1.69 s at the median and 1.91 s at the 95th percentile
  for the first team, 2.50 s and 2.65 s for the second, which also had a 3.5 s move and one timeout. Both use
  most of the 3 s; ours answers in at most 1.1 s. Three-square moves (splits and merges) are 9-10% of their
  moves, 14% of the random bot's.
* **Reading**: both lose to the random bot in 1-2% of games, as our agent did in the simulator. Beating the random
  bot does not separate the agents; games between agents do, and there are only six so far.
* **Simulator check on real games**: all 502 replayed from the recorded moves and outcomes. Every move is legal in
  the simulator, every move measures as often as in the arena, all 499 games that ended normally end with the
  arena's result, and in the 44 sampled the final position has the arena's own hash
  (`tests/compliance/test_cheqqers_arena_replay.py`, names removed). The arena plays the rules of 11 November.

### 23. Track-T: what thinking time is worth
* **Why**: on 13 November the two other teams in the arena beat our agent 14 games to 4 and were level with each
  other; both use 1.7 to 2.5 s a move and move almost only classically (4% and 12% three-square moves). Whether
  our deficit is depth or judgement decides what the remaining days go to.
* **Hypothesis**: four times the thinking time is worth at least 70 Elo to our search in self-play (the usual
  figure for a doubling in full-width game programs is 50 to 100).
* **Measurement**: the same agent at 2,200 ms a move against itself at 550 ms, depth cap 12, four-ply random
  openings played once per colour, 60 pairs (4 seeds of 15), both players in one process so that machine load
  falls on both alike. `leiden_sparring_hub.py cheqqers --agent-a "T2200:expecti_heuristic@depth=12,ms=2200"
  --agent-b "T550:expecti_heuristic@depth=12,ms=550" --num-pairs 15 --seed 51..54 --opening-plies 4`.
* **Decision rule, fixed before the run**: lower end of the 95% interval above +50 Elo: speed comes first
  (a faster interpreter, pruning three-square moves deep in the tree, parallel search at the root). Upper end
  below +50: speed is not where the games are lost; the evaluation and the opening come first. In between: both,
  cheapest first.
* **Outcome (13 November)**: 60 pairs, score 0.554, Elo +38 (95% interval -21 to +99): 17 pairs won 2-0 by the
  agent with four times the time, 10 by the other, 32 split, 1 other. By the rule fixed beforehand this is the
  in-between case: neither lever is excluded, cheapest first. Records:
  `results/series/track_t_2200_vs_550_s5{1..4}.jsonl`.
* **A reading withdrawn the same day**: in all 32 split pairs the same colour won both games, and this entry
  first said the random opening had decided them and that later series should use two-ply openings. Checked: a
  depth-4 search puts 11 of 12 random four-ply openings within a third of a man of level. Two agents that are
  the same program at different speeds mostly play the same moves, so the same colour wins both games of a pair
  whatever the opening; identical agents split every pair. And two-ply openings are worse, not better: there
  are 49 of them and white won 129 of the 201 games played from them on 13 November (the last random move is
  black's). **Series use four-ply openings.**

### 24. Track-O: opening book
* **Why**: in the 11 games lost to other teams on 13 November that ended normally by 13:05, our agent was a man
  down by ply 7 to 9 in most; a search of 8 s (median depth 7) chose another move than the one played at 67 of
  190 decisions (35%), the first or second decision of the game among them in every game. That a longer search
  disagrees is not yet evidence that its move is better.
* **Hypothesis**: playing the first decisions from a table of moves chosen by a search ten times longer than the
  time limit allows gains strength at no cost in time, on any hardware.
* **Acceptance**: (1) lookup is exact: for a position built from a harness payload the book returns the stored
  move as the harness lists it, and nothing when the position is not in it, a capture sequence is open, or the
  stored move is not legal; (2) the server answers a book position in under 50 ms and plays as before without a
  book; (3) strength: a series from the start position in which the opponent's first two moves are random and
  ours come from the book, against the same agent without it, each opening once per colour, lower end of the
  95% interval above 0 over 200 pairs (placeholder); (4) no more games lost to the random player than without.
* **Known limit**: the key is the board hash (side to move, and every square's colour, crown and presence), which
  does not tell apart two positions with the same presences and different entanglement.
* **Acceptance (3) replaced on 13 November, before any result**: the series described needs a runner that makes
  only one side's first moves random; instead, (3) in the book positions where the agent at 2.2 s plays another
  move, the book's move is worth more than the agent's under a search of at least 60 s, on average and in two
  thirds of the cases (`opening_book.py check`, placeholder thresholds), and the arena of 16 November gives the
  games against real opponents.
* **Built (13 November)**: `src/search/opening_book.py` (`build`, `show`, `check`), the player's `opening_book`
  argument, the server's `CHEQQERS_OPENING_BOOK` (unset: every move is searched, as before), the series option
  `book=<file>`; `tests/search/test_opening_book.py`. The module was drafted before this entry and before its
  test (the order the stream protocol forbids) and rewritten against the tests. No book has been built yet.
* **Evidence against its premise (13 November, `arena_records.py review`)**: in the 15 games lost to other teams
  a 15 s judge puts the cost of our first ten plies at 4 evaluation units a decision (25 is one man; 9 of 60
  decisions cost half a man or more, the worst 32), against 65 a decision in plies 10 to 19 and 31 after. The
  games are not lost in the opening. The book stays cheap insurance; it is not where the strength is.

### 25. Track-R, third data: the review of 13 November's lost games
* **Tool**: `uv run python src/arena/arena_records.py review <games.jsonl> --team "<name>" --only-losses`: at
  every move where the team had a choice, a search longer than a game allows values every legal move with a full
  window (`ExpectiCheqqersPlayer.root_values`); the loss is the best value minus the played move's.
* **Result** (15 games lost on the board to the two other teams by 13:20, 233 decisions, judge 15 s on a loaded
  machine, median depth 6): mean loss 30; 34 decisions cost half a man or more. In 13 of the 15 games one or two
  moves cost between 99 and 933, that is, the judge sees a lost game where another move kept it level; they fall
  between ply 14 and ply 57. The judge is no deeper than the agent was nominally (5 to 6 plies): the agent stopped
  one ply short of what decides the game, again and again.
* **Reading**: the games are lost to tactics just beyond the horizon. That is what Track-Q's first step is for.

### 26. Track-Q, step 1: quiescence on pending captures
* **Observation** (28 positions from arena games, fixed depth): 43% of the positions the search evaluates
  statically have a capture pending for the side to move (954 of 2,211 at depth 2; 10,482 of 24,996 at depth 3).
  The evaluation only guesses at such a position with its threat terms.
* **Hypothesis**: searching on through pending captures (compulsory, so few moves) until none is pending or the
  extension budget of the line is spent is worth more in play than the nominal depth it costs (measured: 1.9 to
  2.8 times the time at equal depth with a budget of 4, about 3 times with 8).
* **Acceptance**: (1) off by default, and then the search is the one of before: same moves, same counts (test);
  (2) on, no position with a capture pending is evaluated before the budget is spent (test); (3) equal time,
  2,200 ms a move, two-ply openings once per colour, against the search without it: lower end of the 95%
  interval above 0 over 200 pairs (placeholder; a first read at 45 pairs each for budgets 4 and 8 decides which
  budget gets the 200); (4) the moves of Track-R's review that cost 99 or more are not played at 2,200 ms;
  (5) no more games lost to the random player than without, over 40 games in the organisers' harness.
* **Results (13 November)**:
  - (1), (2) pass (`tests/search/test_expecti_cheqqers.py::TestQuiescence`). Chains of captures on barely
    present pieces can exceed any budget (one position spent 40 extensions), so the budget stays finite.
  - (3) first read, budget 4 against the plain search, 2,200 ms, two-ply openings: 45 pairs, score 0.533, Elo +23
    (95% interval -50 to +99; 13 pairs 2-0, 10 pairs 0-2, 22 split). Budget 8 was stopped at 15 pairs with
    score 0.40. Nothing shown either way; the 200 pairs are owed, with four-ply openings.
  - (4) the 16 moves of the review that cost 99 or more, replayed at 2,200 ms on a loaded machine: the plain
    search plays 5 of them again, budget 4 plays 1, budget 8 plays 3. Eight of the sixteen were played in the
    arena by the 1.2 s agent with its depth cap of 5, which answered in 0.2 to 1.1 s; at 2,200 ms and no cap the
    plain search already avoids them.
  - (5) first data, the arena on 13 November against the house bot: 64 won, 1 lost with budget 4; 149 won,
    7 lost for the plain search earlier that day. Pre-flight in the harness: 4 of 4.
  - In the arena's last 45 minutes it met both teams: 7 won, 7 lost against the first (the plain search at
    1.2 s had won 1 of 7 against it) and 1 won, 4 lost against the second (the plain search at 2.2 s: 7-7). The
    two configurations never met the same team under the same settings, so this does not compare them.
* **Status**: open; not the default.

### 27. Track-Q, step 2: three-square moves only near the root
* **Observation** (40 arena games between agents, 13 November): the legal moves are 5 on average and 20%
  three-square in the first 20 plies, 7.6 and 48% in plies 20 to 39, 27.9 and 68% from ply 40 (a queen can split
  to any two of its targets). In those positions the search reaches depth 3 in 2.2 s; 6 of the 16 game-losing
  moves of the review were played there. The two other teams play three-square moves in 4% and 12% of theirs.
* **Hypothesis**: considering splits and merges only in the first one or two plies of the search (our move and
  the reply) and two-square moves below lets it go one to two plies deeper in the late game, worth more than
  what it stops seeing.
* **Acceptance**: (1) off by default (`quantum_plies=None`), and then the search is unchanged (test); (2) with
  the limit, no three-square move is tried below it and the root still tries them (test); (3) equal time,
  2,200 ms, two-ply openings, against the best version without it: lower end of the 95% interval above 0 over
  200 pairs (placeholder); (4) Track-R's game-losing moves not played at 2,200 ms; (5) no more games lost to the
  random player, which plays three-square moves in 15% of its moves, over 40 games in the harness.
* **First results (13 November)**: (1), (2) pass (`TestQuantumMovesNearTheRoot`). On the review's 16 game-losing
  positions at 2,200 ms on a loaded machine the search finishes 5.0 plies on average without the limit, 7.4 with
  a limit of 1 and 6.9 with 2, and plays the costly move again in 5, 1 and 1 of them. Series against the plain
  search, 2,200 ms, two-ply openings, 40 pairs each: limit 1 scored 0.544 (Elo +30, -34 to +97; 9 pairs 2-0,
  5 pairs 0-2), limit 2 scored 0.488 (-9, -90 to +71). Two more plies of depth did not show as wins against
  the plain search; with Track-T (four times the time: +38) this says depth is worth little to this agent
  against itself.
* **Status**: open; not the default.

### 27b. Track-Q, a defect found on the way: a queen's long slide counted as a capture
* **Found** 13 November, reading the move ordering. `_is_capture_move` called any two-square move that crosses
  two rows or more a capture. For men that is right; a queen slides that far without capturing. In positions
  with queens (the late game, where the search finishes 3 plies and 6 of the 16 game-losing moves were played)
  every long slide was searched first as if it were a capture, never became a killer move, and was extended by a
  ply when it was the only legal move.
* **Fix**: the simulator says whether a capture is pending (`capture_pending`, cached per position); captures
  being compulsory, a legal move is a capture exactly then. The squares' presences are also read once per
  position for ordering instead of once per move. Tests: `TestQueenSlidesAreNotCaptures`.
* **Effect on the default agent**: its moves can differ in positions with queens (one of six benchmark positions
  changed its move at depth 4). Guards rerun: suite 168 passed; with the cache forced on, the differential test
  over 300 games.

### 28. Track-M: move quality on recorded decisions
* **Why**: on 13 November four candidates (quiescence with budgets 4 and 8, three-square moves limited to 1 and
  2 plies) played 15 to 45 pairs each against the plain search at 2,200 ms and none separated from it; the
  intervals are 150 to 200 Elo wide. Two hundred pairs per candidate cost a night each, and the arena gave 32
  games against other teams in two hours.
* **Instrument**: `src/arena/move_quality.py`. `judge`: at every recorded decision (a position of an arena game
  in which the side to move had a choice) the default search values every legal move with a full window for
  60 s. `compare`: each candidate chooses a move there at its own time limit; its loss is the judge's best value
  minus the value of its move. Reported: mean loss, decisions costing half a man or more and two men or more,
  depth finished.
* **Hypothesis**: on a few hundred decisions this orders candidate searches that a series of the same cost
  cannot tell apart.
* **Acceptance of the instrument, before it is used to choose anything**: (1) an agent that searches as the
  judge does loses nothing and a shallower one loses (tests); (2) it orders two pairs whose order is known from
  elsewhere: 2,200 ms above 550 ms, and 2,200 ms above the old default of 1,200 ms with depth cap 5 (the arena:
  1 won, 6 lost against 10 won, 14 lost); (3) a candidate's advantage must show in the count of costly
  decisions, not only in the mean.
* **Known limit**: the judge is our own search and evaluation with more time. It sees tactics; it cannot see
  what the evaluation misjudges, and it favours candidates that think like it (the plain search). A candidate
  that prunes moves the judge considers is at a disadvantage here that it may not have in play.
* **First table (13 November)**: our 801 decisions in the day's 88 games between agents; judge 30 s a position
  on 13 processes, median depth 6 (2 to 14); each candidate on 7 processes at its own time limit.

  | Agent | Mean loss | Half a man or more | Two men or more | Mean depth finished |
  |---|---|---|---|---|
  | The moves played in the arena (four configurations) | 24.1 | 93 | 58 | |
  | Plain search, 2,200 ms, cap 12 | 20.8 | 93 | 59 | 4.6 |
  | The server's default: 1,200 ms, cap 5 | 30.1 | 115 | 73 | 3.9 |
  | Plain search, 550 ms | 30.1 | 143 | 76 | 3.8 |
  | Quiescence, budget 4, 2,200 ms | **13.4** | **73** | **39** | 4.4 |
  | Three-square moves in the first ply only, 2,200 ms | 26.0 | 117 | 65 | 6.6 |
  | Three-square moves in the first two plies, 2,200 ms | 21.8 | 112 | 58 | 6.1 |

* **Acceptance of the instrument**: (1) tests pass. (2) Both known orders come out, in the mean and in both
  counts: 2,200 ms above 550 ms, and above the old default. (3) Quiescence's advantage is in the counts: a
  third fewer decisions costing two men or more than the plain search.
* **What it does not settle**: the judge reached depth 6 at the median, no deeper than the candidates, and it
  considers three-square moves everywhere, so the two candidates that limit them (and search two plies deeper
  than the judge) are judged by a shallower search that thinks differently. Their row is not evidence against
  them. `compare --rejudge-seconds` judges only the decisions where the candidates differ (240 of 801 between
  the plain search and quiescence, 469 among the four), a move at a time and longer; to run overnight.
* **Reading**: quiescence with budget 4 is the candidate to confirm: fewest costly decisions here, 1 of the 16
  game-losing moves repeated against 5, 92-2 against the house bot against 149-7, level with the plain search
  in the arena against teams (8-11 against 10-14) and in 45 pairs of self-play (+23 Elo, -50 to +99).
* **Status**: instrument accepted for tactics; candidates that change which moves are searched need the deeper
  judge.

### 29. Track-R, fourth data: the arena session of 13 November by configuration
* **Source**: the arena's record file fetched at 14:31, 599 games of the day; ours tabulated with
  `arena_records.py`. Normal endings only; won-lost.

  | Our configuration (live from, CET) | House bot | First team | Second team |
  |---|---|---|---|
  | 1.2 s, depth cap 5: the server's default (12:30) | 13-1 | 1-6 | |
  | 2.2 s, cap 8 (12:37) | 46-2 | 1-1 | 2-6 |
  | 2.2 s, cap 10, cache (13:06) | 90-4 | | 7-7 |
  | 2.2 s, cap 12, cache, quiescence with budget 4 (13:45) | 92-2 | 7-7 | 1-4 |

* **Forfeits**: three of ours, each the game the arena started while the connector was being restarted to change
  servers; one of an opponent's. `src/arena/agent_switch.py` removes the cause.
* **Time measured by the arena** for our moves at the 2.2 s budget: 2.15 s at the 95th percentile, 2.34 s at
  most, against 3 s plus 0.5 s grace. No late reply, no fallback, no error in the servers' logs.
* **Reading**: the default configuration is the weak one (1 of 7 against a team; 18 of 43 at 2.2 s). At 2.2 s
  the agent is below the two teams, not far below. Our games against a team repeat their first 5 to 22 plies
  and then differ; the samples per cell are a handful of distinct games.
* **Simulator check**: all 1,101 games on record replay; the 1,090 normal endings agree in result and hash.


### 30. Track-S, C++ series: choosing the search setting with the native engine (13 November, night)
* **Why**: the search now runs in C++ (about 70 times the positions a second of the Python); series of 200 pairs
  become affordable, and the server's setting (three-square moves everywhere, quiescence off, hand-set weights,
  extension 4) has to be chosen against quiescence, the three-square limit and the fitted weights.
* **Built**: `cpp=1` in a Cheqqers agent spec (`InProcessCheqqersAgent(engine="cpp")`: `native_cheqqers.best_move`
  with the same options); `--pair-start` (a shard plays pairs `k..k+n-1` of the series its seed defines);
  `read_series_files` keys pairs by (seed, pair) so shards merge, and raises on a game read twice; the summary
  gives games won/drawn/lost, native errors and mean depth finished. Tests:
  `tests/arena/test_series_native_and_shards.py` (the native agent plays the Python's move at fixed depth; shards
  of deterministic agents replay the whole series game for game). Native and Python agents chose the same move at
  depth 3 in 202 of 202 positions of 40 random games, and in 15 of 15 with each option (weights, quiescence,
  three-square limit).
* **Method, fixed before the run** (`~/work/_scratch/qg_cpp/series/PLAN_rules.txt`): frozen library
  `libcheqqers_2026-11-13_v1.so` (sha256 30b4c999...), code snapshot; 2,200 ms a move both sides, depth cap 12,
  seed 1114, four-ply random openings once per colour; first a sanity series, the C++ search against the Python
  search, 50 pairs, passing if the lower end of the 95% interval is above 0, otherwise everything stops; then each
  candidate against the C++ default over 200 pairs, looks at 100 and 150 pairs only.
* **Outcome: the sanity series failed.** C++ default against Python, 50 pairs: 48 won, 3 drawn, 49 lost, score
  0.495, Elo -3 (95% interval -72 to +64); 11 pairs 2-0, 24 split, 12 pairs 0-2. The C++ search finished 7.7
  plies on average against 5.4; slowest moves 2,004 ms and 2,058 ms; no fallback, no native error. In all 24
  split pairs the same colour won both games; median game 30 plies. Seventy times the positions and two more
  plies did not show as wins against the same search, as with Track-T (four times the time: +38, -21 to +99).
  By the rule, the candidate series were stopped.
* **Interim data, stopped before the first look** (not a result; shards completed when stopped, pairs not
  contiguous): fitted weights (`results/eval_fit/r2_all_squash.json`) against the C++ default, 63 pairs: 83 won,
  1 drawn, 42 lost, score 0.663, Elo +117 (+63 to +178), 24 pairs 2-0 against 4 pairs 0-2. Quiescence (budget 4),
  56 pairs: +22 (-30 to +75). Three-square moves in the first two plies, 56 pairs: +22 (-38 to +83), depth 10.2
  against 7.9. No fallback, no native error, slowest move 2,003 ms.
* **Reading**: depth is not what this agent lacks against itself; the evaluation is. The fitted weights are the
  only candidate whose interval is clear of 0 so far, on a third of the planned pairs. Records:
  `~/work/_scratch/qg_cpp/series/out/` (sanity) and `out_pc/` (candidates, both machines).
* **Status**: open; the server's setting unchanged. The candidate series resume from where they stopped
  (`run_queue.sh` skips finished shards) if the failed sanity check is accepted as a finding rather than a fault.
* **Continued (lead's decision, 13 Nov 22:28, logged in PLAN_rules.txt):** the failed sanity check is accepted as
  a finding (fixed-depth identity 202 of 202 holds; in self-play the opening and the chance draws decide most
  games), not a fault, and steps 2-6 ran under the same rules. Records merged from both machines:
  `~/work/_scratch/qg_cpp/series/out_merged/`. Each candidate against the C++ default (R), Elo with 95% interval:

  | Candidate | 100 pairs | 150 pairs | 200 pairs | Verdict by the rules |
  |---|---|---|---|---|
  | Fitted weights (FW) | +83 (+39 to +130) | +86 (+51 to +124) | +96 (+65 to +129) | helps (stop at 100) |
  | Three-square moves in the first 2 plies (QP2) | +14 (-31 to +59) | +21 (-16 to +58) | +45 (+14 to +77) | helps (at 200) |
  | Quiescence, budget 4 (Q1) | +31 (-5 to +68) | +30 (+0 to +61) | +9 (-19 to +36) | not shown |
  | FW and QP2 together (step 5) | +53 (+12 to +94) | +70 (+38 to +104) | +59 (+31 to +88) | helps |

  The rules did not say how to choose between FW alone and the combination, which scored below FW alone against R.
  Added by the lead before it ran: FW+QP2 against FW, 200 pairs, the combination chosen only if its lower bound is
  above 0. Result: -10 (-41 to +20), 184 won, 20 drawn, 196 lost. **Pick: FW alone.**
  Step 6, FW against the random player, 50 pairs: 98 won, 0 drawn, 2 lost. **The pre-stated pass (at most 1 of 100
  lost or drawn) failed by one game.** Both losses were in games whose four random opening plies include two of
  ours (the tournament has no random openings: pair 15, black, 23 plies; pair 21, white, 31 plies). The
  organisers' requirement is a won match of 4 rounds; the server default before the change won about 97% against
  the house bot in the arena. Adopted with this noted.
  No native error, no fallback, no move above 2,016 ms in any series.
* **Adopted (13 Nov, 23:50)**: `server.py` plays `config/cheqqers_eval_weights_2026-11-13.json` (the same file,
  sha256 f694a32a..., copied out of the ignored `results/eval_fit/`) unless CHEQQERS_EVAL_WEIGHTS is set; the
  start-up test checks it (`tests/arena/test_warmup.py`). Everything else at the C++ default.
* **Reading**: the evaluation is the lever, depth is not, against this opponent. That is the case for the
  transformer of `docs/specs/cheqqers_transformer_training.md`: a learned evaluation, not more search.

### 31. Track-V, a small value network at the C++ search's leaves (14 November, night)
* **Why**: track 30 showed the evaluation is the lever (fitted weights +96 Elo against the default; depth ~0). A
  ReLU MLP 144 -> 64 -> 32 -> 1 trained on native self-play (`src/training/train_cheqqers_value_mlp.py`, run1 on
  stefano-pc) predicts held-out decisive results better than the fitted linear evaluation: log-loss 0.510 against
  0.530. It is small enough to run at every leaf in C++.
* **Built**: `native/cheqqers/value_net.{h,cpp}` (forward pass in double, the first layer summed over the non-zero
  inputs only); `SearchOpts::net`, used in `Searcher::leaf` for non-terminal leaves as +-900 tanh(logit / 2) (the
  fitted evaluation's squash scale; the logit is the side to move's, negated for the other side; terminal leaves keep
  +-1000 / 0). New C entry points `cq_best_move_net`, `cq_selfplay_net`, `cq_net_*`; CqOpts and ABI 2 unchanged.
  `native_cheqqers.ValueNet` reads the JSON; `best_move(..., value_net=)` and `selfplay(..., value_net=)` (None =
  CHEQQERS_VALUE_NET if set, the server's switch; "" = none); hub spec key `net=<file>` (only with cpp=1; hub agents
  never read the environment variable). Native ledger entry in `native/cheqqers/LEDGER.md`.
* **Checks**: `tests/native/test_value_net.py` (7 tests): the C++ input equals `encode_position` flattened, bit for
  bit, on all 1,451 positions of the arena fixture (record built from the Python engine) and on native self-play
  records; on 3,684 arena positions the C++ logit differs from a float64 numpy forward by at most 2.7e-15 (from a
  float32 PyTorch forward by 1.5e-6); at depth 1 every root score is the outcome-weighted net value; the search with
  a net returns legal moves; without one the results are bit-identical. Default path: tests/native 201 passed,
  tests/arena 76 passed, identity gate 1 (387,712 checks) and gate 2 (532 searches) 0 mismatches on the frozen
  library.
* **Speed**: bench (6 positions, depth 4, quantum_plies=2) 416k nodes/s with the net against 626k with the fitted
  weights (-34%); at the series setting (2,200 ms, no qp) 324-344k against 500-538k nodes/s; the depth finished on
  the 6 positions was the same (7/6/8/7/10/5). In the games the net finished 7.5 plies on average against 8.1.
* **Method, fixed before the run** (section "NET vs FW" of `~/work/_scratch/qg_cpp/series/PLAN_rules.txt`): frozen
  library `libcheqqers_net_2026-11-14_v1.so` (sha256 89a406e5...), snapshot `tree_net_2026-11-14`, network sha256
  364d9841...; NET = fitted weights + net against FW = fitted weights (the server default); 2,200 ms, depth cap 12,
  seed 1114, four-ply openings once per colour, 200 pairs, looks at 100/150/200, early stop only if the 95% interval
  is clear of 0.
* **Outcome: the network hurts; stopped at the first look.** Pairs 0..99: 78 won, 13 drawn, 109 lost, score 0.422,
  **Elo -54 (95% interval -92 to -17)**; 8 pairs 2-0, 61 split, 20 pairs 0-2. All 155 complete pairs (not the rule's
  sample): -58 (-91 to -25). No native error, no fallback, slowest move 1,994 ms (none above 3,000 ms).
* **Reading**: a better log-loss on recorded positions did not make a better leaf evaluation. Candidate causes, not
  tested: (1) a third fewer nodes a second, about half a ply less in games; (2) the net learnt on the positions
  where a move was chosen, while the leaves the search evaluates include many positions no game reaches (pending
  captures, quantum branches), where the linear evaluation's structure extrapolates and an MLP may not; (3) a
  0.02 log-loss gain is small next to the evaluation's job of ranking siblings. Before another series: measure the
  net's ranking of siblings on leaf positions sampled from searches, and train on such positions.
* **Status**: closed for run1; the server's setting unchanged (CHEQQERS_VALUE_NET unset). Records:
  `~/work/_scratch/qg_cpp/series/out/NET_vs_FW` (laptop), `out_net_pc/` (stefano-pc), merged in `out_net_merged/`.

### 32. Track-V, second and last attempt: residual value nets, leaf-trained nets, smaller nets (14 November, night)
* **Why**: track 31's net lost as a full replacement of the leaf evaluation. Three cheaper hypotheses, tested
  cheapest first, with an offline proxy as the gate before any series (lead's rule: only a candidate that beats FW on
  the proxy goes to a series; "beats" in a series = lower end of the 95% interval > 0).
* **Built** (additive; default path bit-identical):
  * Residual leaf mode (`value_net.h`): logit = logit_per_man * (w . features) + lambda * net(x), with the search's
    weights; JSON key `{"residual": {"lambda": x}}` (`native_cheqqers.ValueNet` calls the new `cq_net_set_residual`).
    lambda 0 gives FW's search exactly (the features are antisymmetric in the point of view).
  * Leaf sampler: `cq_selfplay_leaves` (`selfplay(..., leaf_rate=)`, `selfplay_native.py --leaf-rate`) writes each
    searched leaf with probability leaf_rate as `{"leaf","turn","d","cp","x"}` (+ `"state"` with `--state`); its
    own generator, so the games are unchanged.
  * `train_cheqqers_value_mlp.py`: `--residual` (FW logit as a fixed offset, last layer started at 0), `--hidden`
    with no value (linear 144 -> 1), `--leaves` (leaf lines, labelled by the game result), `--leaf-npz` (leaves with
    a search label), `--search-weight` (positions' own root search value as target), `--cache` (npz).
  * Checks (`~/work/_scratch/qg_cpp/valnet2/tools/check_residual.py`): zero residual (lambda 0 and 1) = FW search
    in 80 of 80 fixed-depth searches (move, nodes, depth, score); a random residual net's depth-1 root scores =
    numpy within 5.7e-13; self-play with leaf sampling writes the same position lines as without; 806 of 806 leaf
    states re-import to the written input. Default path vs the frozen track-31 library: see the native ledger.
* **Data**: training on the FW self-play shards 00-14 and the default-weight shards 00-08 of stefano-pc (812,845
  positions of decisive games); validation shard_15 of the FW self-play (16,685). Leaves: 2 runs of FW self-play on
  the laptop (300 ms, three-square moves everywhere, FW weights): 515k leaves labelled by the game result; 285k
  leaves with states, labelled by a depth-4 FW search of the leaf (positions with one legal move: the
  outcome-weighted value of their children). About half the sampled leaves have a capture pending. On leaves the FW
  logit predicts the game result worse than a coin (log-loss 0.74 against 0.69) but tracks the depth-4 search value
  closely (correlation 0.87; log-loss 0.475 against the label's own entropy 0.443: 0.03 nats of room).
* **Held-out log-loss** (positions, decisive games; FW linear 0.529): residual 144-32-1 0.498; 144-16-1 0.507;
  linear 144-1 0.521; 144-64-32-1 0.498; trained on the root search values instead of results 0.497; with the
  depth-4-labelled leaves 0.499 (leaf search-label log-loss 0.472 against FW 0.475); on leaves only 0.518 (leaf
  0.462).
* **Proxy** (`valnet2/tools/proxy.py`): 1,500 middlegame positions (ply 8-60, >= 2 legal moves, at most 2 per
  game) from FW self-play games no net was trained on (500 from shard_15, 1,000 from games played after the
  training snapshot); reference = FW search of 10 s, depth cap 20 (mean depth 8.2); each candidate a 300 ms search,
  depth cap 12, FW weights + the net, 8 parallel workers; agreement with the reference move, paired sign test
  against FW in the same batch (load kept constant within a batch). FW against itself: 0.711-0.720, repeat
  +7/-1 to +18/-12. Track 31's net, the known loser (-54 in the series), scores 0.651 (+119/-212): the proxy ranks it
  correctly.

  | Candidate (lambda) | agreement | vs FW same batch | nodes/s vs FW |
  |---|---|---|---|
  | FW | 0.711-0.720 | - | 300k (8 workers, loaded) / 393k (idle) |
  | run1 net (track 31) | 0.651 | +119/-212 | -43% |
  | residual 144-32-1 (0.25 / 0.5 / 0.75 / 1) | 0.699 / 0.710 / 0.683 / 0.667 | 0.5: +121/-125, +125/-125, +120/-127 in three batches | -25% |
  | residual linear 144-1 (0.5 / 1) | 0.704 / 0.681 | +91/-104, +107/-155 | -11% |
  | residual 144-16-1 (0.5 / 1) | 0.696 / 0.674 | +103/-128, +125/-183 | -19% |
  | residual 144-64-32-1 (0.5 / 1) | 0.684 / 0.647 | +115/-158, +130/-228 | -41% |
  | search-value target, 32 (0.5 / 1) | 0.691-0.695 / 0.664-0.669 | e.g. +109/-132 | -25% |
  | + leaves, result labels, 32 (0.5 / 1); 16 (0.5) | 0.672 / 0.635; 0.695 | +100/-158; +93/-117 | -25%; -19% |
  | + leaves, depth-4 search labels, 32 (0.5 / 1); 16 (1) | 0.703 / 0.679; 0.690 | +107/-133; +122/-167 | -25%; -19% |
  | leaves only, search labels, 32 (0.5 / 1) | 0.705 / 0.701 | +107/-129, +129/-157 | -25% |
  | FW at 225 ms / 400 ms (what the time is worth) | 0.713 / 0.739 | +20/-30 / +42/-14 | - |

  At 1 s instead of 300 ms (FW 0.782 and 0.779): residual 32 at 0.5 0.720, leaves-only 0.740 / 0.730, 16 with
  leaves 0.724; all below FW with p < 0.001.
* **A second proxy that failed its control**: agreement restricted to the positions where the 10 s FW search and the
  10 s candidate search choose the same move (symmetric in the two evaluations). Residual 32 at 0.5: 0.842
  against FW 0.808 (+84/-45, p = 0.001); but track 31's net also scores above FW there (0.843 against 0.824,
  +93/-73), so this variant does not separate a known loser from FW and was not used.
* **Outcome: no candidate beats FW on the proxy; by the rule, no series was played.** The best, the residual
  144-32-1 at lambda 0.5, ties FW at 300 ms (0.710-0.715 against 0.711-0.720, the same number of positions won
  and lost) and loses at 1 s. Its node rate is 25% lower; FW given 25% less time scores 0.713, so the evaluation's
  own contribution is about zero. Larger lambda, larger nets and leaf training all move further from FW and score
  lower; the leaf search labels show why there is little to gain: the linear evaluation is already within 0.03 nats
  of a depth-4 search on the leaves.
* **Reading and limits**: the proxy's reference is FW itself, so it can reject but hardly certify a different
  evaluation; the one symmetric variant tried failed its control. A candidate that only ties (residual 32 at 0.5)
  could still differ in games; the only test of that is a series, which the rule did not allow.
* **Status**: closed. The server keeps FW (`config/cheqqers_eval_weights_2026-11-13.json`), CHEQQERS_VALUE_NET
  unset. Nets, logs and proxy records: `~/work/_scratch/qg_cpp/valnet2/` (nets/, proxy/, logs/, tools/).

### 33. Track-G, the transformer as the player, the C++ search as the teacher (14-16 November) — DONE: ships
* **Why**: tracks 31-32 showed that a network inside the alpha-beta search costs more nodes than it gains. Plan of
  14 Nov (Stefano, via the lead), replacing the first plan of this track (GPU MCTS first): the fast C++ search
  generates games and labels, a policy-value transformer trained on them picks the move; a search on top of the
  policy only if it beats the plain policy. GPU MCTS kept as groundwork (gate 1 passed).
* **Built** (additive; the FW search path is unchanged):
  * `native/cheqqers/mcts.{h,cpp}` + C API (`cq_mcts_*`, `cq_mcts_collect_many/apply_many` on a persistent thread
    pool): MCTS with chance nodes by expectation (all outcomes of a move created and evaluated together), PUCT,
    virtual loss, moves pending at the network skipped by selection; `src/search/gpu_mcts_cheqqers.py` (driver,
    CUDA-graph and chunked evaluators). `cq_encode_position` (the network input of any position).
  * `cq_sp_*`: batched self-play stepping in C++ (encode/apply for thousands of games on a few threads) for GPU RL.
  * `src/search/cheqqers_policy_player.py` (argmax of the policy, one forward pass a move) and hub agent keys
    `pv=<model.pt>` (the policy plays) and `topk=<k>` (with cpp=1: the C++ search among the policy's k best moves).
  * `src/training/train_cheqqers_transformer.py`: `--cache-dir` (shards parsed once, in parallel, into npz arrays),
    `--exclude-states` (proxy positions out), `--dedup` (one row per unique position, soft targets; search labels
    override played moves), `--extra-npz/--extra-weight` (labelled positions), `--extra-val-npz`, model size flags.
  * `src/training/dagger_cheqqers.py` (`play`: the policy plays batched games on the GPU, against itself or with FW
    on one side; `label`: the FW search labels the positions it reached, 1 s). `src/training/selfplay_pg_cheqqers.py`
    (pure self-play policy gradient, side experiment).
* **Checks**: FW path: identity gate 2 (532 searches, 17,820,003 nodes) and native golden (5,893 searches) 0
  mismatches against the pre-track library; tests/native + tests/arena 287 passed. New tests: test_policy_player.py
  (native encoding = training input bit for bit; hub `pv=`), test_gpu_mcts.py (chance-node Q = expectation over the
  Python get_move_outcomes; deterministic; many-tree batches = single trees), tests/training/test_train_dedup.py.
  MCTS + C API clean under ASan/UBSan in a standalone driver (26 positions, 4 batch sizes, 26 threaded trees).
* **Data finding**: the FW self-play data (2.14 M positions) holds only 401,026 unique positions: FW is deterministic
  until a chance event, so games repeat. 88% of validation positions also occur in training; the first model's
  top-1 was 0.93 on those and 0.30 on unseen ones (val top-1 0.856 = memorisation). Headline metrics from then on:
  proxy agreement (1,500 positions, 10 s FW reference; none occurs in training by encoding) and top-1 on unseen
  positions.
* **Results so far** (proxy top-1; FW 300 ms = 0.720, FW 1 s = 0.782 on the same positions):

  | model | data | proxy | unseen val top-1 | vs random | vs FW (2,200 ms) |
  |---|---|---|---|---|---|
  | pv1 d256/6 layers, 12 epochs | all self-play rows | 0.539 | 0.296 | 90/0/10 (50 pairs) | 26/0/54, Elo -127 [-217,-51] (40 pairs) |
  | pv2 dedup, 16 epochs | unique positions | 0.487 | 0.336 | - | - |
  | pv4 dedup, 64 epochs | unique positions | 0.569 | 0.352 | - | - |
  | pv5 = pv4 + 50k DAgger labels (1 s) | + labels | 0.649 | 0.339 | 90/0/10 | - |
  | RL snapshot A (pv5 + ~100 policy-gradient iterations) | self-play | 0.563 | - | 94/0/6 | 23/9/48, Elo -112 [-193,-42]; vs pv5: 46/11/23, +103 [+43,+169] |
  | DAgger round 1 (loop, 86k labels) | + labels | 0.671 | 0.335 | 98/0/2 | - |
  | DAgger round 2 (128k labels; not promoted under the later rule) | + labels | 0.691 | 0.343 | 87/0/13 | - |
  | round-1 data, ordered move head (step 1) | + labels | 0.674 | 0.336 | 91/0/9 | - |
  | **round 1 inside the MCTS** (PUCT, chance by expectation, 2,200 ms, ~15k evals/move, 3 games sharing the GPU) | - | - | - | 97/0/3 | **37/6/37, score 0.500, Elo 0 [-54,+54]** (40 pairs) |

* **Step 0 (recipe, docs/specs/cheqqers_rl_recipe.md)**: search on top of the policy is worth ~+110-127 Elo: the
  round-1 network that loses ~-110 as a pure policy plays FW even inside the MCTS. One MCTS process alone on the
  RTX 5090: 58,276 evals a move (28.4k/s) at 2,200 ms; the match measured 15,380 (GPU shared by 3 games + training).
* **Step 1**: the summed move head scores a->b and b->a (and the permutations of three-square moves) the same: 13.7%
  of positions have such twins among their legal moves, the target is one of them in 2.2%. Fixed as an option
  (move_head="ordered"); not adopted (vs random 91/0/9 vs round 1's 98/0/2 on the same data).
* **Bugs found on the way**: an MCTS chance node could end up without children when every outcome probability was
  below the 1e-9 filter (out-of-bounds read; now all outcomes are kept then); REINFORCE side line stopped (ad hoc).
* **Generations (recipe sections 2-4)**: Gumbel root (m 16, n 64), completed-Q targets, 2,048 games in flight with
  refill, ~155k games/hour self-play only (~56k with 10% FW opponents at 30 ms), ~3-7 min a generation.

* **Panel A (14 Nov, 08:00-09:30 UTC; recipe section 5)**: MCTS(net) at 2,200 ms, 3 games in parallel on the RTX
  5090 alone (generation and training paused), seed 1114 pairs 0..39:

  | agent inside the MCTS | vs FW, 40 pairs | vs round 1 (MCTS), 10 pairs | evals/move median |
  |---|---|---|---|
  | Gumbel generation 10 (from round 1) | 43/6/31, score 0.575, Elo +53 [-1,+109] | 10/2/8, +35 [-51,+125] | 83,555 |
  | DAgger round 5 (pure policy -48 vs FW) | 40/6/34, score 0.537, Elo +26 [-36,+90] | 11/2/7, +70 [-68,+237] | 35,914 |

  The DAgger network's sharper priors fill the MCTS batches less (hypothesis for the lower evals/move, not tested).
* **Panel B (14 Nov, 11:30-13:50 UTC)**, same settings, fb=0 and native errors 0 in every comparison: generation 48
  (after lr 3e-5 + value anchor from generation 22) vs FW 42/8/30, score 0.575, Elo +53 [-8,+116] (= generation 10);
  vs round 1 -35, vs round 4 0, vs generation 10 +17 (10 pairs each); DAgger round 7 inside the MCTS vs FW 41/1/38,
  +13 [-55,+82]. Value head on the fixed held-out set (selfplay_fw/15 + selfplay/09): CE 0.790 (gen 0) -> 0.642
  (gen 5) -> 0.747 (gen 19, drift) -> 0.678 (gen 48, after the fix). A replay of generation 48 vs FW with every move
  logged: 42/7/31, +48 [-12,+111], 96k evals/move, move time max 1,983 ms (tree freed on a background thread).
* **Loss diagnosis** (31 lost games, the move where the MCTS value most exceeds a 3 s FW search): positional 10,
  capture 9 (7 of 9 with FW's own move), endgame 5, low evals 4 (decided endings), measurement 3; median gap 0.10;
  MCTS value -0.14 vs FW -0.36 at those moves. Scales: the MCTS value is P(win) - P(loss) from the network's WDL
  head; FW's root score was mapped to score / 900 = tanh(logit / 2), the fitted evaluation's own squash (its weights
  were fitted to game outcomes, track 30), which equals P(win) - P(loss) only if FW's logit is a calibrated log-odds of
  winning. A deep alpha-beta root score is not calibrated in that sense, so "the network is over-optimistic" is a
  hypothesis consistent with these numbers, not a measured fact; what does hold is that no single class of mistakes
  dominates the losses.
* **Bugs found and fixed on the way**: server built its MCTS with batch 128 while every panel ran 256 (one construction
  path now, tested); the server's settings guard rejected the tournament payload (start_rows = 0 after
  from_tournament_json), so it would have played FW silently (caught by an end-to-end server test); a failed DAgger
  training (CUDA out of memory: the generator held 28.8 GB) was evaluated with fallback moves (gates now check training
  and fallbacks; GPU shares capped: generator 0.55, trainer 0.28, producer 0.08); freeing a ~1e5-node tree cost
  114 ms at the end of every move (now on a background thread).
* **DAgger line paused after round 10** (pure policy vs FW: round 5 -48, round 7 -44, rounds 9-10 -44, plateau);
  its 446k labelled positions stay in the generation buffer's FW share.
* **Status on 15 November**: in progress (generation loop, DAgger loop, panels every ~2 h; final series at the Tue 14:00 freeze on
  fresh seed 1117); this entry is completed at the end of the track.

#### Result (15-16 November): the network agent ships
* **Shipped**: MCTS (C++, chance nodes by expectation) + the transformer `config/cheqqers_net.pt` (checkpoint
  day_ckpt_00225 of the concurrent Gumbel self-play loop, sha256 `4fd1cbdb...`), batch 256, c_puct 1.5, fpu 0
  (sweep S1, seed 1115, 152 pairs a setting: fpu 0 vs 0.25 paired difference +0.036 [+0.005, +0.067]).
* **Final series** (plan fixed and appended to PLAN_rules before the first game; seed 1117, fresh openings of 4
  random plies, each once per colour; 2,200 ms a move for both; looks at 100/150/200 pairs): stopped at 100 pairs,
  MCTS + network vs FW **108/16/76, score 0.580, +56 Elo [+18, +96]**; 0 fallbacks, 0 native errors, slowest move
  1,981 ms, sha256 of library, network and code re-checked (17 of 17). Ship rule (ships unless the upper bound < 0):
  ships, and the whole interval is above 0.
* **Benchmarks** (issue #9): vs random 196/0/4 (100 pairs, 200 ms a move); vs the early checkpoint gen010, 2,000 games
  batched on the GPU at 128 simulations a move: 1187/115/698, +87 Elo [+72, +102].
* **Resources** (RTX 5090, one game): ~2.7 GB GPU memory per server, ~44% utilisation, ~290 W, 90-130k network
  evaluations a move at 2.2 s. CPU only (32 threads): ~7,700 evaluations a move (above the 5,000 floor, so a strong
  CPU plays the network; not measured against FW).
* **Arena, 16 November** (first live test; names replaced): against the strongest other team 59/7/27 over 93 games
  (score 0.67), 0.69 as white and 0.65 as black; against the house bot 199/0/6. Two forfeits by timeout came from the
  organisers' tunnel (our replies left in 1.95 s and 1.35 s; the arena measured > 3.5 s): from 12:52 the arena ran
  with `CHEQQERS_TIME_LIMIT_MS=1500`. The tournament runs the agents on the organisers' hardware, without a tunnel,
  so the default stays.
* **Colour**: from the real start position black does slightly better in every pairing of agents in the arena
  records (white's score 0.43 over 146 agent-vs-agent games, 0.44 between two other teams); in our series, which
  start from 4 random plies, white does better (0.60 over 232 games). Our series therefore never measured the start
  position every tournament game uses (open: an opening book built with the network's long search, Track-O).
* **Not done**: tree reuse between moves; time management across a game (the arena limit is per move); the
  network on CPU against FW.

### 34. Track-O, second attempt: the network's own opening book (16-17 November) — NOT SHIPPED
* **Why**: in the 16 Nov arena 21 of 27 losses against the strongest team fell in 4 repeated lines (both agents
  are nearly deterministic before the first measurement), and 13 of our 14 black losses followed white's 13-10 or
  13-11, where a 30 s search of our network prefers another reply than the 2.2 s one (13-10: 4/6/7 instead of 4-6;
  13-11: 5-7 instead of 4-7). Training and every benchmark had started from random openings, never from the real
  start position.
* **Built**: `opening_book.py build-net` (our first 4 decisions of both colours from the real start, each searched
  30 s by the shipped network; every opponent reply and outcome; plus our positions in the recorded arena games up
  to ply 24): 875 positions. The MCTS path of `server.py` answers booked positions (`CHEQQERS_OPENING_BOOK`).
  `book_check.py`: at 2.2 s the shipped network already plays the book's move in 95% of the 875 positions (100%
  of white's first 3 moves); the book changes black's first reply after 13-10 and 13-11, and 45 other positions.
* **Test** (plan fixed before the games): our network as black, white's first move forced to 13-10 / 13-11
  alternately by `forced_opening_proxy.py`, then normal play; organisers' harness, 3 s a move; book on vs off,
  20 games each, against FW and against the June winner's public agent; 0 forfeits, 0 unfinished.

  | Opponent | no book | book |
  |---|---|---|
  | FW | 10/1/9 = 0.53 | 8/2/10 = 0.45 |
  | June winner (LMU, public) | 19/0/1 = 0.95 | 16/1/3 = 0.83 |

* **Decision**: the book does not help (rule: ships only if it clearly helps): not shipped. The arena signal was a
  correlation with how one opponent continued those lines, not a weakness of our replies against others.
* **Also found**: the organisers' engine refuses moves that entangle more than 15 squares and the harness then
  asks the same side again; for our search such a move is a free pass, and FW repeated one 8,453 times (a local
  game stuck for 4.5 h). Fixed in `server.py` (`playable_cheqqers_moves`, shipped).
* **Kept for later**: `OPEN_LO/OPEN_HI` in the training loop (a third of games from the real start, running
  16 Nov evening to 17 Nov morning, checkpoints 246-568, untested); the book tools.
