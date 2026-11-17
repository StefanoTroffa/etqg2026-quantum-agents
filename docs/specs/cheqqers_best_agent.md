# Spec: the strongest Cheqqers agent we can submit by 17 November

Written 13 November 2026. Upstream: Tracks U, S, W, V2, Q, B in `experiments/CONCLUDED_EXPERIMENTS.md`. This file
compares what we have with the state of the art for learning and playing a game of this kind, and turns the gap
into an ordered plan. Sources are given by name and link; "ours" means measured in this repository on the date
shown.

---

## 1. Where the agent stands (13 November)

| Item | State | Evidence |
|---|---|---|
| Rules | Simulator equal to the organisers' harness of 11 November on legal moves and move results, except 2 positions in 28,000 where the payload does not say which half-pieces belong together | Track-U, V11 |
| Qualification | 20 of 20 against the harness's random player on the new rules, slowest move 1.09 s of 3 s | V10 |
| Search | Iterative-deepening expectimax with alpha-beta, Star1 cut-offs at chance nodes, a transposition table per move, killer and history ordering, one-ply extensions for forced captures | `src/search/expecti_cheqqers.py` |
| Speed | About 3,800 positions a second in pure Python; depth 4 to 7 in 1.2 s depending on the position; a third of the time recomputes square probabilities | Profile of 13 November (S32) |
| Evaluation | Hand-written, a dozen constants in `OfficialCheqqersEnv.evaluate_state`: material (queen 2.6), advancement, back rank, centre, queen diagonals, a bonus for superposition, capture threats; none of them fitted to data | Read 13 November |
| Learned part | A 72-128 value network from a self-play league; its one comparison with pure search was run with a faulty series runner and on the old rules, so nothing is known | Track-N, Track-L |
| Known weakness | Loses a few percent of games to a random player by trading down and splitting its last pieces; a 20-second search plays the same moves, so it is the evaluation, not the depth | Track-L |
| Testing | Colour-balanced series with Elo interval and a sequential test; sound since 10 November (three bugs fixed); all series so far on the old rules | Track-L, Track-U |

## 1b. After the arena session of 13 November (written the same afternoon)

The table above was written before our agent had met another team. What the session and the afternoon added,
each with its track in `experiments/CONCLUDED_EXPERIMENTS.md`:

| Finding | Evidence | Consequence |
|---|---|---|
| The server's default (1.2 s a move, depth cap 5) won 1 game of 7 against a team; at 2.2 s the agent won 18 of 43 against the two teams, which are level with each other | Track-R, fourth data | The default is the weak configuration. Changing it is gated: attention A9 |
| The lost games are lost by one or two moves each, between ply 14 and ply 57, that a longer search values as losing the game; the first ten plies cost almost nothing | Track-R, third data (`arena_records.py review`) | Tactics, not the opening. The row "Known weakness" above (evaluation, not depth) was about games against the random player and does not describe these |
| Half of those moves were played by the default stopping at its cap with most of its time unused | Track-Q step 1, result (4) | Let the clock end the search |
| Four times the time is worth +38 Elo in self-play (-21 to +99); two more plies from limiting three-square moves did not show against the plain search; quiescence neither | Tracks T and Q | Depth is worth little to this agent against itself. Self-play series of 40 pairs cannot rank these candidates |
| A queen's long slide was treated as a capture by the search | Track-Q 27b | Fixed; the arena games of the 13th were played before the fix |
| 43% of the positions the search evaluates have a capture pending | Track-Q step 1 | Quiescence exists and is off; its worth in play is open |
| PyPy is slower than CPython here | Track-W | Dropped |
| Fitted weights predict the arena's games between agents slightly better than the hand-set ones | Track-V2 | A candidate, not a result |
| The simulator replays all 1,101 arena games on record | V12 | The search sees the game the arena plays |

### Plan from here to Tuesday 17 November, 23:59 (replaces steps 4 to 7 of section 3)

| When | What | Acceptance |
|---|---|---|
| Fri 13, evening | Track-M: a 30 s judge on our 801 arena decisions; every candidate (plain, old default, 550 ms, quiescence 4 and 8, three-square limit 1 and 2, the combination) at its own time limit | The instrument orders the two pairs whose order is known (2.2 s over 550 ms; 2.2 s over the old default) before it is used to choose |
| Fri night to Sat | Series at 2,200 ms, four-ply openings, 200 pairs each against the plain search, for the candidates Track-M ranks above it and for the fitted weights | Lower end of the 95% interval above 0, or the candidate stays off |
| Sat-Sun | The configuration to show on Monday; optionally an opening book made with it (`opening_book.py build`, then `check`); the review rerun with a 60 s judge | Pre-flight in the harness, differential test, suite |
| Mon 16, 12:30-14:30 | Arena through the agent switch: first hour the pick, second hour the runner-up; records kept; review the same afternoon | No forfeit; results by team and by configuration |
| Mon evening | The submission's configuration; defaults in `server.py` changed on Stefano's go (A9) | D9 decided |
| Tue 17 by noon | Freeze: pre-flight 20 per colour, differential test on the newest harness, suite, README, commit and push on Stefano's go | All green |

## 2. Current approach against the state of the art

"State of the art" here means two things: what wins in games of this class, and what the reinforcement-learning
literature offers. They do not point the same way, and the table says which applies.

| Component | State of the art | Ours | Gap, and what closing it takes here |
|---|---|---|---|
| **Overall architecture** | For perfect-information games with few chance outcomes, full-width search with a fast learned evaluation: draughts (Scan), Othello (Logistello), 2048 (n-tuple networks with expectimax), backgammon (neural evaluation with 0-3 plies). Stockfish (alpha-beta with a small network) still beats Lc0 (AlphaZero-style) in chess | The same architecture, with an unlearned evaluation | None in kind. Keep it |
| **AlphaZero family** (AlphaZero, MuZero, Stochastic MuZero, Gumbel variants, EfficientZero) | Self-play with tree search guiding a policy-value network; Stochastic MuZero handles chance nodes and beat GNU Backgammon; Gumbel variants improve the policy with very few simulations. All need a fast simulator and days of GPU self-play; MuZero-style methods solve learning a model, which we have exactly | Not used. A league trainer exists but trains a value network on game outcomes | Not closable in four days in Python on CPU, and selective tree search misses shallow tactics that full-width search sees (Baier and Winands). Rejected for the submission; the only variant worth a later look is Gumbel AlphaZero with our exact simulator |
| **Chance nodes** | Enumerate when outcomes are few; Star1/Star2 pruning (Ballard 1983; Hauk, Buro, Schaeffer 2004: Star2 with good ordering "considerably outperforms Expectimax"); sample only when chance is dense. Average in win-probability space, with tight value bounds | Exact enumeration and Star1. Values are raw scores in +-900 with wins at +-1000 | Bounds are loose relative to typical scores, so chance pruning rarely fires; a bounded, calibrated evaluation fixes both the pruning and the meaning of an average (Track-V2) |
| **Evaluation** | Fitted to data. Texel tuning: logistic regression of the game result on the features, about 100 Elo in chess from ~64,000 games. Scan: pattern tables fitted on self-play; on the draughts forum "already with 32K training games the evaluation surpassed the handcrafted" one, and 1,000 games beat material alone; Scan itself was fitted on 600,000 fast self-play games by plain gradient descent in "a few hours on a single core", skipping positions with a capture pending; Kingsrow gained 72 Elo over its best hand-written evaluation the same way. Networks have not beaten fitted tables in draughts (Scan with NNUE lost 10-18 Elo to Scan). NNUE: a small network trained on moderate-depth search scores, +80 Elo in Stockfish | Hand-set constants, never fitted | The largest gap that four days can close. Constants as data, features weighted by presence probability, logistic regression on our own games (Track-V2) |
| **Learning signal for a network** | Search-backed targets beat outcome targets: TreeStrap (Veness et al. 2009) took a linear chess evaluation from 1068 (TD-Leaf) to 2157 Elo; Descent / Athénan learns from the minimax values of the whole search tree with a value-only network, won Olympiad golds including Canadian draughts, and its stochastic version beat expectiminimax on EinStein würfelt nicht! | Outcome targets from league games, blended into the evaluation at 0.35 | Retrain on search targets, infer in NumPy, accept only by an equal-time series (Track-B, after the submission unless the rest is done) |
| **Search efficiency** | Transposition move, killers, history, principal-variation search (~10% fewer nodes), aspiration windows, never evaluating a position where a capture is forced, late-move reductions at depth; no null move in draughts (zugzwang, forced captures) | Transposition move, killers, history, forced-capture extension | PVS, aspiration windows, quiescence on forced captures; small each, about half a ply to a ply together (Track-Q) |
| **Speed** | Draughts engines search millions of positions a second on bitboards in C; pure-Python chess (Sunfish) reaches 30,000-80,000 a second | 3,800 a second | The second largest gap. First the cheap part: cache the square probabilities the profile shows being recomputed (Track-W). A compiled core is out: the organisers' machine is unknown |
| **Time use** | Use most of the allowance with a hard deadline and a fallback move | 1.2 s of 3 s | Free strength once the arena's latency is known (Track-S step 2) |
| **Endgames** | Exact databases (Chinook 10 pieces, Scan 6 pieces, Kingsrow 12). For our board, an upper bound by hand: all classical positions with at most 6 pieces are about 86 million (about 43 MB at 2 bits), with at most 8 about 3.5 billion (1.7 GB) | None | A classical database is exact only while every piece is certain; either side can split at any time. It could serve as a leaf value in fully classical positions. Not for the 17th (Track-H stays open) |
| **Openings** | Books from long searches or self-play | None | Optional: an offline book of the first plies from long searches |
| **Measuring strength** | Sequential tests (Stockfish's fishtest), colour-swapped pairs from varied openings, the same chance seed for both games of a pair, a fixed gauntlet of older versions against self-play illusions | Colour-balanced series, sequential test, one start position | Paired random openings and a gauntlet (random player, previous default) before any tuning (Track-S step 1) |
| **Published agents for this game** | Raat et al., "Quantum Checkers" (IEEE CoG 2025): a random player and Monte Carlo tree search with 200-800 random rollouts, its edge over random shrinking as the quantum level rises; heuristic evaluation named as future work. The winner of the June 2026 tournament (public repository `michaelkoelle/cheqqers-ai`, by its author's account): tree search with heuristic priors, up to 1,000 playouts, 22-ply guided rollouts, an evaluation of presence-weighted material (queen 1.75) with a penalty of 0.35 for every piece the enemy can capture and small terms for support, edge and centre; splits slightly discouraged; no chance nodes and no entanglement, only per-square probabilities; "written in roughly two hours", 100% against random; a self-play transformer was tried first and dropped at 75% | A full-width searching agent with exact chance nodes and exact entanglement | In method we are ahead of both: exact outcomes where they approximate, full width where they sample. In evaluation the June winner has one thing ours lacks, a per-piece penalty for being capturable (ours has only one global threat term), and it values a queen at 1.75 where ours says 2.6. Whether our agent beats it is not known: running its code locally was blocked by this session's safety settings (section 4) |

### What the comparison says not to do before the 17th

* Train anything AlphaZero-like from scratch, or adopt MuZero-family toolkits: wrong problem (we have the model) and wrong budget.
* Null-move pruning: unsound with forced captures.
* A league or population: it answers strategy cycles in hidden-information games; here it costs compute and adds noise.
* A compiled search core: the largest possible speed gain and the one deployment risk we cannot test.
* Sampling at chance nodes: built for dense chance; ours have two to four outcomes.

## 3. Plan to Tuesday 17 November, 23:59

Each step is a track with its acceptance written here before it runs; a step that fails its acceptance is dropped,
not tuned until it passes. All series on the rules of 11 November.

| # | Track | What | Acceptance (**placeholders** where marked) | Cost | When |
|---|---|---|---|---|---|
| 0 | U | Rules of 11 November in the simulator | Done: V10, V11 | | Fri 13 |
| 1 | S | Series that can be trusted: random two-ply openings played once with each colour, both games of a pair on the same chance seed, a gauntlet of the random player and the current default; result per game in a file | The runner reproduces a series bit for bit from its seed; two identical agents score 50% +- the interval over 200 games | half a day | Fri 13 |
| 2 | W | Cache each position's square probabilities; no change of behaviour | Same move as before on 200 recorded positions at fixed depth 4; at least 1.5 times the positions per second (**placeholder**) | hours | Fri 13 |
| 3 | S | Arena, 12:30-14:30 Friday and Monday: round-trip latency per move, results against other teams, records of all teams' games kept | Latency known; the budget set to 3000 - 500 - worst latency - 300 ms | sessions | Fri 13, Mon 16 |
| 4 | V2 | Evaluation as data, then fitted: features = the present terms plus presence-weighted piece-square values, mobility, pieces exposed to capture, last-pieces-in-superposition; logistic regression of the result on positions from 20,000+ fast self-play games; output bounded as a win probability | Against the current default over 400 paired games: lower end of the Elo 95% interval above 0, and no more losses to the random player over 400 games than the default | a day, an overnight run | Sat 14 - Sun 15 |
| 5 | Q | Principal-variation search, aspiration windows, no static evaluation while a capture is forced, value bounds tight to the bounded evaluation | Same or better move on recorded tactical positions; fewer positions for the same depth; the series as in step 4 | half a day | Sun 15 |
| 6 | S | Budget: raise the time per move to what step 3 allows | Series against the 1.2 s default as in step 4; zero timeouts through the arena on Monday | hours | Mon 16 |
| 7 | | Freeze: harness pre-flight 20 per colour on the new file, differential test, full suite, README, commit, push | All green; Stefano's go | hours | Tue 17, by noon |

Optional, only if steps 1-6 are done: an opening book from long searches (small gain, no risk at move time);
Track-B, the value network retrained on search targets (uncertain gain, a day and a half).

### Midstream contracts for the steps that add code

* **Step 1** (`src/arena/leiden_sparring_hub.py`): `--openings N --opening-plies K --out FILE`; one JSON line per
  game with seed, opening, colours, result, plies, slowest move, search errors. Owned by the arena layer; must not
  reach into the player.
* **Step 2** (`src/env/cheqqers_official.py`): a cache of marginals per environment object, invalidated wherever
  amplitudes change (`measure_square`, the move applications, `clone` copies it). Must not change any returned
  value: the differential and rules tests are the guard.
* **Step 4** (`src/env/cheqqers_official.py`, `src/search/expecti_cheqqers.py`, new `src/training/fit_evaluation.py`):
  `EvalWeights` frozen dataclass with today's constants as defaults, loadable from JSON; `features(env, colour)`
  returning the feature vector the evaluation is linear in, so that the fit and the evaluation cannot drift apart;
  the fitted file under `config/`, chosen by `CHEQQERS_EVAL_WEIGHTS` in the gateway. Default behaviour unchanged
  until the acceptance is met and Stefano confirms (gate: default player).
* **Step 5** (`src/search/expecti_cheqqers.py`): no interface change.

State on 13 November: steps 0 and 1 are done (Track-U, Track-S). Of step 4 the contracts are built and tested
with the default agent unchanged: `EvalWeights`, `eval_features` (17 features, the last four new and weighing
nothing by default), `logit_per_man` for the win-probability scale, `src/training/fit_evaluation.py`,
`weights=` in the series command, `CHEQQERS_EVAL_WEIGHTS` in the gateway. One choice is open and the series will
settle it: positions with a capture pending are left out by draughts engines because their search never
evaluates them; ours does, and three features are non-zero only there, so the fit is tried with those positions
kept.

## 4. What we don't know

* **How we do against the June winner.** Its code is public (MIT) and is a single HTTP server like ours; the two
  can play through the organisers' harness with `scripts/harness_match.sh`. Starting that server from this session
  was refused (third-party code). Stefano can start it himself:
  `cd <clone of michaelkoelle/cheqqers-ai> && PORT=8090 node ai-server.js`; then
  `scripts/harness_match.sh <harness dir> http://127.0.0.1:8080/move http://127.0.0.1:8090/move 10`. Its team is
  one of the six and may submit something stronger than its June agent.
* Whether quantum moves help at all against strong play. One 2026 preprint (Scholten, Westerbaan, Samardjiska;
  abstract only) proves for a class of quantised games that they give no advantage against a perfect classical
  player but can amplify an imperfect opponent's mistakes; whether it covers these rules is open. The June winner
  discourages its own splits. Our evaluation rewards them. The fit of step 4 will say what our own games show.

* How strong the other teams' Cheqqers agents are, and whether they exploit superposition better than our
  evaluation understands it. The arena sessions are the only source.
* The speed of Vincenzo's machine: a time budget buys an unknown depth there. Fixed-depth results are the ones we
  can state.
* Whether a fitted evaluation generalises from self-play to other opponents. The gauntlet and the arena are the
  checks; a fit that wins self-play and loses to the random player more often is rejected.
* Whether the payload's missing piece identities cost anything in play (2 positions in 28,000 in the simulator).

## Sources

Ballard, "The *-minimax search procedure for trees containing chance nodes" (1983). Hauk, Buro, Schaeffer,
"*-Minimax performance in backgammon" and "Rediscovering *-minimax search" (2004). Veness, Silver, Uther, Blair,
"Bootstrapping from game tree search" (NIPS 2009), https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search.
Cohen-Solal, "Learning to play two-player perfect-information games without knowledge", https://arxiv.org/abs/2012.10700,
and the stochastic extension, https://arxiv.org/abs/2302.04318. Antonoglou et al., "Planning in stochastic
environments with a learned model" (ICLR 2022). Danihelka et al., "Policy improvement by planning with Gumbel"
(ICLR 2022). Silver et al., AlphaZero, https://arxiv.org/abs/1712.01815. Texel's tuning method,
https://www.chessprogramming.org/Texel%27s_Tuning_Method. Stockfish NNUE,
https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/. Scan, https://github.com/rhalbersma/scan, and
the World Draughts Forum thread on fitted evaluations, https://damforum.nl/viewtopic.php?p=128524 (forum
evidence). Jaśkowski and others on n-tuple networks for 2048, https://arxiv.org/abs/2111.11090. Sequential tests:
https://www.chessprogramming.org/Sequential_Probability_Ratio_Test and
https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html. Baier and Winands on Monte
Carlo tree search and minimax hybrids, https://dke.maastrichtuniversity.nl/m.winands/documents/mcts-minimax_hybrids_final.pdf.
Raat et al., "Quantum Checkers" (IEEE CoG 2025), https://arxiv.org/abs/2506.05962. The June 2026 winner's
agent, https://github.com/michaelkoelle/cheqqers-ai. Letouzey on Scan's training, https://damforum.nl/viewtopic.php?p=112675.
Gilbert on Kingsrow's fitted evaluation, https://www.bobnewell.net/nucleus/checkers.php?itemid=1177. Fierz's
engine tutorial (zugzwang and null move, endgame databases), https://www.fierz.ch/strategy2.htm. Draughts NNUE
thread, https://damforum.nl/viewtopic.php?t=8298. Scholten et al., https://arxiv.org/abs/2607.06550. Sunfish (pure-Python chess
speed), https://github.com/thomasahle/sunfish.
