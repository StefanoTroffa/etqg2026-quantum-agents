# Results: exact solution of TiqTaqToe, quantumness 0-2

28 October 2026, updated 31 October 2026. Squares are zero-based as on the site (0 1 2 / 3 4 5 / 6 7 8). Single
round, auto-measurement when the board is full. "Value" scores X win = +1, O win = -1, tie or draw = 0. The first
two sections (the quantumness 0 and 1 numbers and their method) are the earlier solve: X moves first, decoherence
off. "Exact teacher tables" below covers more configurations.

## Numbers (exact rationals, `results/q1-solution.json`)

| Setting | Value | P(X wins), X maximises it | P(O wins), O maximises it | Best opening | Canonical states |
|---|---|---|---|---|---|
| Quantumness 0 | 0 | — | — | (ordinary tic-tac-toe) | — |
| Q1, manual measurement off | **1** | **1** | 0 | split [0,4] | 28,308 |
| Q1, manual measurement on (site's "Casual" preset) | **7/8** | **7/8** | 0 | split [0,4] | 128,609 |

Opening values (value / P(X wins)) for the 11 symmetry classes of X's first move:

| Opening | Manual off | Manual on |
|---|---|---|
| split [0,4] | 1 / 1 | 7/8 / 7/8 |
| split [0,2] | 7/8 / 7/8 | 1/2 / 3/4 |
| split [0,8] | 7/8 / 7/8 | 3/4 / 3/4 |
| split [0,5] | 3/4 / 3/4 | 3/4 / 3/4 |
| split [1,4] | 3/4 / 3/4 | 3/4 / 3/4 |
| split [0,1] | 5/8 / 3/4 | 1/2 / 3/4 |
| split [1,3] | 5/8 / 3/4 | 5/8 / 3/4 |
| split [1,7] | 1/2 / 3/4 | 1/2 / 3/4 |
| X on 4 | 1/4 / 1/2 | 1/4 / 1/2 |
| X on 0 | 0 / 1/2 | 0 / 1/2 |
| X on 1 | 0 / 1/4 | 0 / 1/4 |

Findings:

- **The site's claim holds, but only with manual measurement off.** Opening with the split [0,4] (corner + centre)
  wins for X with probability 1 against any defence. With manual measurement on, which is the default and the
  "Casual" preset, O can measure pairs as a move and holds X to 7/8. [0,4] is still the unique best opening.
- At Q1, O cannot force any chance of winning under either setting: against X's best play O's win probability is 0.
  With manual measurement on, the missing 1/8 is a tie or draw.
- Every split opening beats every classical opening. The best classical first move (centre) is worth only 1/4.

## Method

At quantumness <= 1 each superposition is an independent 50/50 pair on two previously empty squares. A state is
therefore (measured tokens, list of pairs, side to move), taken modulo the 8 board symmetries. Each move strictly
increases 2 x (measured squares) + (covered squares), so positions never repeat. The game graph is a DAG, and
memoised backward induction in exact `Fraction` arithmetic is the full solution. No value iteration or depth cutoff
is needed. Code: `tiqtaqtoe/q1.py` (model and solver), `tiqtaqtoe/solve_q1.py` (driver, ~110 s).

## Verification

- `tiqtaqtoe/engine.py` is a literal port of the referee's rule functions: the organisers' harness
  `../workshop_resources/local-tiqtaqtoe-server.js` (lines 41-643, unchanged since 4e1a18a, sha256
  `3c488fd02209f97e272e64d925de61a93fa7f59e2f327c93e2ee321217990d95`) on `@quantumplayed/quantum-game-engine` 1.0.3
  (npm integrity `sha512-r3mX5yeP...CW9puQ==`). It runs on the full amplitude state in floats, as the JS does, and
  returns every outcome the engine's sampler can draw, with its probability, instead of sampling.
- `tests/test_oracle.py`, run with `python3 -m unittest tests.test_oracle -v` (7 s), replays JS traces record by
  record. At every recorded state it compares the legal moves in harness order; for every legal move, the outcome
  states grouped by (branches, measured squares, decoherence counters, turn, winner), with probabilities and every
  amplitude within 1e-9; and on each harness outcome, the port's winner and empty squares. Without traces (they live
  in the gitignored `results/data/oracle/`) it is skipped. The traces come from `oracle/trace.mjs` (plan Task 1, Node
  25.9): 30 random games per configuration, X and O first alternating, seed 2026, run from `exact/` as

  ```
  node oracle/trace.mjs --engine ~/tqt-oracle/node_modules/@quantumplayed/quantum-game-engine/dist/index.js \
    --harness ../workshop_resources/local-tiqtaqtoe-server.js --quantumness Q --decoherence D --games 30 \
    --seed 2026 > results/data/oracle/q{Q}-d{D}.jsonl
  ```

  Result (29 October 2026; q1-d1 and q1-d3 added 31 October 2026): every (state, move) pair matches. The largest
  probability difference is 1.4e-17; the amplitudes agree exactly. The test requires all 11 traces to be present.

  | Trace | States | (state, move) pairs | Distinct pairs | Outcomes |
  |---|---:|---:|---:|---:|
  | q0-d0 (quantumness 0, decoherence off) | 229 | 1,283 | 904 | 1,283 |
  | q1-d0 | 310 | 4,974 | 3,522 | 6,182 |
  | q1-d1 | 252 | 4,671 | 3,285 | 6,897 |
  | q1-d2 | 261 | 4,607 | 3,264 | 6,231 |
  | q1-d3 | 279 | 4,977 | 3,591 | 6,259 |
  | q2-d0 | 344 | 5,777 | 4,355 | 7,368 |
  | q2-d1 | 288 | 5,339 | 3,797 | 7,595 |
  | q2-d2 | 291 | 5,192 | 3,797 | 7,200 |
  | q2-d3 | 324 | 5,655 | 4,230 | 7,346 |
  | q3-d0 | 343 | 6,991 | 5,474 | 11,208 |
  | q4-d0 | 345 | 7,673 | 6,193 | 12,088 |
  | Total | 3,266 | 57,139 | 42,412 | 79,657 |

- What the traces cover: every move kind (measure moves from Q1, entangling from Q2, 522 phase moves at Q4),
  decoherence limits 1-3 with their counters, full-board collapses, groups of three squares or more (Q3, Q4), and
  winner codes 0-4. What they cannot test: no branch is ever pruned, so neither the sampler's rule that the last
  outcome absorbs pruned mass nor the engine's degenerate measurement is reached. And no recorded state tells apart
  three port items: the winner read from the first branch by basis index (in no state do branches disagree on the
  winner), a measurement taking its whole group rather than direct partners (no group is a chain), and
  `resolveDeterministicSquares` (every square it marks is certain, or collapses when its partner is measured next).
  A deliberate mutant of each of those three passes the oracle test; mutants of the other port items fail it. At
  quantumness <= 2 the three cannot change an outcome: groups are two squares, branches agree on every measured
  square and on whether any square is empty, and every pre-marked square is certain or collapses with its partner.
- Harness behaviour beyond the plan's description, ported as the code does: a phase move (Q4) on two occupied
  squares that are in no group is legal, because `squareToPair` is undefined for both (`isLegalMove`, line 406).
  The Q4 trace has 45 of them; the first is O's `[3, 8]` at `q4-d0.jsonl` line 19.
- Not ported, on purpose (no trace line reaches any of them): the engine's degenerate measurement (a
  measurement whose outcomes include one of mass <= 1e-12; harness play never reaches such a state), which raises
  `NotImplementedError`; the engine's `cleanAmplitudes`
  (1e-10), which only ever drops what `serializeEngineState` drops next; pairs `[a, b]` with a >= b, which
  `isLegalMove` can accept but `getAvailableMoves` never lists (the match loop forfeits a remote player who sends
  one), and which `legal` rejects; and the display fields (`board`, `statistics`, pair slot numbers).
- `tests/test_q1.py`, run with `python3 -m unittest tests.test_q1 -v`; 6 tests pass in 7 s:
  - 150 random games (Q0, Q1 with manual measurement on and off), checking more than 1,000 (state, move) pairs. At
    each pair the compact model and the engine port give the same legal moves and the same outcome distribution
    with winner codes, to 1e-9.
  - Hand cases: a classical line wins with pairs still pending; a full board collapses all pairs (16 outcomes for 4
    pairs); equal lines after measurement give a tie.
  - Q0 solves to a draw.
- A deliberate mutant that never collapses a full board fails the cross-check.

## Exact teacher tables

31 October 2026. `python3 -m tiqtaqtoe.solve --quantumness 1 --decoherence 0 [--manual-off]` (rules hash
`8039f497ddf07985`) solves each configuration backward at five draw weights w in {0, 1/4, 1/2, 3/4, 1}, in exact
Fractions, with tables in `results/data/q1-d0-m{0,1}/` (gitignored) and the summary in `results/q1-d0-m{0,1}.json`.
An entry is (P(X wins), P(O wins), P(tie), P(draw), expected plies) under optimal play for the draw weight w; the
mover maximises its own wins plus w times ties and draws. Values are exact Fractions. `engine.py` matches
the JS on all 57,139 traced (state, move) pairs (Verification above). The compact model agrees with `engine.py` in
`tests/test_compact.py` (about 33,000 pairs) and `tests/test_q1.py`. Task 7's Bellman check finds 0 violations on
every table. Machine: the Mac, Python 3.14, one process.

The rows below the first three were added on 31 October 2026 (plan Task 10): every other configuration up to
quantumness 2, with decoherence 1, 2 and 3 and with manual measurement on and off, run in one series on the same
machine. Each one was counted completely (one rules hash, no `RepeatError`, none out of reach of the 5,000,000-state
fit rule), solved, verified and checked against a passing JS trace (Verification above), and its checkpoints were
cleaned up. Solve seconds and peak RSS are those of the solve step alone (`/usr/bin/time`); the summary's seconds
add the count. The largest, Q2 with decoherence 3 and manual measurement on, has 1,261,461 states and took 3,200 s
and 651 MiB to solve (verify: 6,677 s and 1.22 GiB).

| Configuration | Canonical states | Solve seconds | States/s | Peak RSS | Root entry, w = 1/2 | Root entry, w = 0 |
|---|---:|---:|---:|---:|---|---|
| Q1, decoherence off, manual measurement off | 28,308 | 31.4 | 901 | 35.3 MiB | (1, 0, 0, 0, 33/4) | (1, 0, 0, 0, 33/4) |
| Q1, decoherence off, manual measurement on | 70,384 | 143.3 | 491 | 62.4 MiB | (7/8, 0, 0, 1/8, 31/4) | (7/8, 1/8, 0, 0, 61/8) |
| Q2, decoherence off, manual measurement on (tournament default) | 1,643,542 | 3,549 | 463 | 682 MiB | (23/32, 1/4, 0, 1/32, 79/8) | (9/16, 7/64, 5/64, 1/4, 71/8) |
| Q1, decoherence 1, manual measurement off | 3,743 | 5.8 | 645 | 21.5 MiB | (1, 0, 0, 0, 5) | (1, 0, 0, 0, 5) |
| Q1, decoherence 1, manual measurement on | 9,388 | 18.0 | 520 | 22.2 MiB | (1, 0, 0, 0, 5) | (1, 0, 0, 0, 5) |
| Q1, decoherence 2, manual measurement off | 14,405 | 17.9 | 805 | 28.7 MiB | (3/4, 1/8, 0, 1/8, 61/8) | (3/4, 1/8, 0, 1/8, 63/8) |
| Q1, decoherence 2, manual measurement on | 36,145 | 77.7 | 465 | 36.8 MiB | (3/4, 1/8, 0, 1/8, 61/8) | (3/4, 1/8, 0, 1/8, 63/8) |
| Q1, decoherence 3, manual measurement off | 41,231 | 47.6 | 866 | 40.2 MiB | (1, 0, 0, 0, 7) | (1, 0, 0, 0, 7) |
| Q1, decoherence 3, manual measurement on | 97,278 | 237.2 | 410 | 63.5 MiB | (1, 0, 0, 0, 7) | (1, 0, 0, 0, 7) |
| Q2, decoherence off, manual measurement off | 92,441 | 148.5 | 622 | 136.8 MiB | (3/4, 3/16, 1/16, 0, 245/32) | (3/4, 1/4, 0, 0, 119/16) |
| Q2, decoherence 1, manual measurement off | 8,424 | 13.8 | 611 | 32.5 MiB | (147/256, 13/128, 3/16, 35/256, 927/128) | (167/256, 47/256, 31/256, 11/256, 1753/256) |
| Q2, decoherence 1, manual measurement on | 22,336 | 45.7 | 489 | 48.8 MiB | (165/256, 25/128, 1/8, 9/256, 855/128) | (167/256, 47/256, 35/256, 7/256, 1757/256) |
| Q2, decoherence 2, manual measurement off | 50,344 | 72.6 | 694 | 97.5 MiB | (85/128, 11/64, 3/64, 15/128, 539/64) | (21/32, 13/64, 1/32, 7/64, 507/64) |
| Q2, decoherence 2, manual measurement on | 145,039 | 320.6 | 452 | 124.0 MiB | (163/256, 13/64, 3/64, 29/256, 2205/256) | (79/128, 17/128, 1/8, 1/8, 545/64) |
| Q2, decoherence 3, manual measurement off | 392,255 | 521.3 | 752 | 258.9 MiB | (61/128, 5/32, 3/64, 41/128, 543/64) | (75/128, 81/256, 7/128, 11/256, 1947/256) |
| Q2, decoherence 3, manual measurement on | 1,261,461 | 3,200.0 | 394 | 651.4 MiB | (15/32, 13/64, 9/64, 3/16, 251/32) | (9/16, 17/64, 5/64, 3/32, 547/64) |

| Configuration | Bellman violations | Floor (16 rows) | Policy disagreement, w pairs 0-1/4, 1/4-1/2, 1/2-3/4, 3/4-1 |
|---|---:|---|---|
| Q1, decoherence off, manual measurement off | 0 | all 16 rows at or above the value | 0.0453, 0.0082, 0.0173, 0.0020 |
| Q1, decoherence off, manual measurement on | 0 | all 16 rows at or above the value | 0.0442, 0.0118, 0.0194, 0.0048 |
| Q2, decoherence off, manual measurement on | 0 | all 16 rows at or above the value | 0.0913, 0.0252, 0.0437, 0.0126 |
| Q1, decoherence 1, manual measurement off | 0 | all 16 rows at or above the value | 0.0598, 0.0037, 0.0171, 0.0005 |
| Q1, decoherence 1, manual measurement on | 0 | all 16 rows at or above the value | 0.0321, 0.0015, 0.0068, 0.0002 |
| Q1, decoherence 2, manual measurement off | 0 | all 16 rows at or above the value | 0.0905, 0.0139, 0.0274, 0.0037 |
| Q1, decoherence 2, manual measurement on | 0 | all 16 rows at or above the value | 0.0604, 0.0077, 0.0186, 0.0035 |
| Q1, decoherence 3, manual measurement off | 0 | all 16 rows at or above the value | 0.0528, 0.0138, 0.0218, 0.0066 |
| Q1, decoherence 3, manual measurement on | 0 | all 16 rows at or above the value | 0.0514, 0.0085, 0.0142, 0.0061 |
| Q2, decoherence off, manual measurement off | 0 | all 16 rows at or above the value | 0.1163, 0.0396, 0.0604, 0.0186 |
| Q2, decoherence 1, manual measurement off | 0 | all 16 rows at or above the value | 0.1853, 0.0398, 0.0585, 0.0226 |
| Q2, decoherence 1, manual measurement on | 0 | all 16 rows at or above the value | 0.1205, 0.0207, 0.0336, 0.0160 |
| Q2, decoherence 2, manual measurement off | 0 | all 16 rows at or above the value | 0.1477, 0.0373, 0.0624, 0.0237 |
| Q2, decoherence 2, manual measurement on | 0 | all 16 rows at or above the value | 0.1076, 0.0308, 0.0479, 0.0154 |
| Q2, decoherence 3, manual measurement off | 0 | all 16 rows at or above the value | 0.1340, 0.0441, 0.0666, 0.0208 |
| Q2, decoherence 3, manual measurement on | 0 | all 16 rows at or above the value | 0.1018, 0.0318, 0.0507, 0.0159 |

- **State counts.** Measurement off gives 28,308, equal to the count in `results/q1-solution.json`: when the turn
  fixes the token counts, the colour swap (X and O exchanged, used by the tables) merges no two classes. Measurement
  on gives 70,384, fewer than the 128,609 of the table above: a measure move spends a turn without placing a token,
  so the token counts no longer fix the turn, and positions that are colour swaps of each other merge. (This is the
  expected cause; no test in `tests/test_solve.py` counts the merges.) The two counts are of different classes (the
  old one keeps X and O apart), so they are not comparable one to one.
- **Roots agree with the earlier solve.** Value 1 with manual measurement off, and 7/8 with it on at w = 1/2, as in
  the first table. The w = 0 and w = 1 distributions are a tie-break artefact (ruling R12), not a finding: at those two
  weights the stored policy breaks value ties by fewer plies, so among moves of equal value it may pick a quick loss.
  The Q1 root with manual measurement on stores O winning with probability 1/8 at w = 0 for that reason. The Q0 root
  stores (0, 1, 0, 0, 6) at w = 0, a quick loss, although Q0 is a draw, and (1, 0, 0, 0, 5) at w = 1. The values are
  exact at every w; at w in {0, 1} the split between the outcomes worth w, and the plies, reflect the tie-break.
  The same holds for the "Root entry, w = 0" column above (Q2 included) and for any w = 0 or w = 1 distribution in
  this file. Only the w = 1/4, 1/2 and 3/4 distributions describe play.
- **Floor.** The teacher plays the compact model against a random and a greedy opponent, 200 games per row, at w = 0
  and w = 1/2, on each side and with each player first.
  The criterion is mean >= value - 3 x stderr; every row passes, and the smallest margin (mean - value) is 0.0.
- **Load time.** `Teacher(Config(1, 0, True))` takes 0.1 s after the import, well under the 5 s the spec allows.
- **Q2, the tournament default** (quantumness 2, decoherence off, manual measurement on; `results/q2-d0-m1.json`).
  The count (337 s, 100 MiB, no `RepeatError`) found 1,643,542 canonical states, complete, so the solve ran on the Mac:
  the Q1 rates predicted about 1.5 GB and 56 min, and it took 3,549 s (the summary's 3,886.2 s is the count's 337 s
  plus the solve) and 682 MiB peak (`/usr/bin/time`: 715,096,064 bytes). The verify run took 7,292 s and 1.19 GiB.
  States/s is states over the solve alone. The floor margins (mean - value) range from 0.178 upward. Load time:
  `Teacher(Config(2, 0, True))` takes 2.7 s including the import, under the 5 s the spec allows.
- **Q2 root values.** X first, w = 1/2: value 23/32 + 1/2 x 1/32 = 47/64, from P(X wins) = 23/32, P(tie) = 0 and
  P(draw) = 1/32. With O first the floor rows give X (second) 17/64 and O (first) 47/64 at w = 1/2, and at w = 0 X
  first gets 9/16 and X second 13/64. In plain terms: moving first is worth a lot at Q2, about 0.73 against 0.27 at
  w = 1/2, and the first mover can no longer force a win, as X could at Q1 with manual measurement off.
- **Roots at w = 1/2, X first** (value = P(X wins) + 1/2 x (P(tie) + P(draw)), exact; the w = 0 and w = 1 columns
  are tie-break artefacts and are left out of these readings):

  | | Q1, manual off | Q1, manual on | Q2, manual off | Q2, manual on |
  |---|---|---|---|---|
  | Decoherence off | 1 | 7/8 | 25/32 | 47/64 |
  | Decoherence 1 | 1 | 1 | 377/512 | 371/512 |
  | Decoherence 2 | 13/16 | 13/16 | 191/256 | 367/512 |
  | Decoherence 3 | 1 | 1 | 169/256 | 81/128 |

- **Q1 with decoherence.** With decoherence 1 or 3, X wins with probability 1 whether manual measurement is on or off
  (root entries (1, 0, 0, 0, 5) and (1, 0, 0, 0, 7)), so the manual-measurement defence that holds X to 7/8 without
  decoherence is gone. With decoherence 2, X does not: P(X wins) = 3/4, P(O wins) = 1/8, P(draw) = 1/8, value 13/16,
  and again the same with measurement on or off. This is a reading of the numbers; we have not traced why 2 differs
  from 1 and 3.
- **Q2 with decoherence.** Decoherence changes the value, and the first mover's edge shrinks as the limit grows.
  With manual measurement on: 47/64 (decoherence off), 371/512, 367/512 and 81/128 for decoherence 1, 2 and 3
  (0.734, 0.725, 0.717, 0.633). With measurement off the sequence is not monotone: 25/32, 377/512, 191/256, 169/256
  (0.781, 0.736, 0.746, 0.660), with decoherence 2 above decoherence 1. Decoherence 3 is the lowest in both series,
  yet X, moving first, still has the better of it (value above 1/2) in every Q2 configuration.
- **Manual measurement, on against off.** At Q2 the value is higher with measurement off at every decoherence
  setting, by 3/64, 3/256, 15/512 and 7/256 (decoherence off, 1, 2, 3 respectively; that is, 0.047, 0.012,
  0.029 and 0.027), so letting O measure pairs as a move helps O somewhat. At Q1 the two agree at decoherence 1, 2
  and 3 and differ only with decoherence off (1 against 7/8).
- **Tests.** `python3 -m unittest discover -s tests -v`: 131 tests pass in 60 s, among them
  `test_q1_numbers_reproduce_results` (the tables reproduce `results/q1-solution.json`).

## The agent, and the live arena (12 November 2026)

- **Move server.** `python3 -m tiqtaqtoe.server --port 8081` answers the tournament's `POST /move` from the tables
  (`tiqtaqtoe/server.py`): the stored policy's move at draw weight 1/2, about 50 ms a move. When the tables cannot
  answer (unsolved settings, decoherence on without counters in the payload, a legal-move list unlike ours, a
  malformed payload) it plays a random legal move and logs why, so it never forfeits by its own reply.
- **Against the harness's random player** (Q2, decoherence off, manual measurement on; 5,000 rounds a seat on the
  compact model): it wins 96.7% of rounds moving first (loss 1.2%) and 91.1% moving second (loss 3.7%), and over 99%
  of 3-round matches. Through the real harness (`local-tiqtaqtoe-server.js`, Node), two 3-round matches against its
  random AI were won 2-0 and 3-0, all 27 moves from the tables.
- **The arena** (organisers' sparring arena, Q2, decoherence off, 5000 ms a move; engine 1.0.2, whose `dist/` is
  byte-identical to the 1.0.3 we verified against). One other team and the random house bot took part. All 4,904
  game records of the session were replayed with `engine.py` from the recorded measurement outcomes; every replay
  ends as the arena recorded it, a further check of the rules port.
  - Every move of both agents is value-optimal at w = 1/2 by our tables: 12,332 of ours, 17,107 of the other team's.
  - Head to head (1,364 games, we were X throughout, each side moving first in half): our score 0.733 ± 0.017 moving
    first and 0.265 ± 0.016 moving second, against the optimal values 47/64 = 0.734 and 17/64 = 0.266.
  - Against the house bot: 0.988 moving first for both teams; moving second 0.932 for us (682 games) against 0.959
    for the other team (1,088 games). Both play optimally, so the difference is in the choice among moves of equal
    value; a tie-break that prefers the move a weak opponent is most likely to get wrong would close it at no cost
    against a strong one.
  - Two forfeits by timeout, both with our reply 22 s and 61 s late at the arena. The connector gives up on the agent
    after 7 s, so the delay was between the connector and the arena (tunnel or network), not the server.

## Limits

- Single round only. The multi-round match (best of N with tie resolution) is not modelled.
- Every configuration with quantumness 0-2 is solved (31 October 2026): Q1 and Q2, decoherence off, 1, 2 and 3,
  manual measurement on and off. Quantumness 3 and 4, the turn timer and the multi-round match remain.
  `engine.py` supports entangling moves, phase moves and decoherence on full amplitudes.
- The tie-break between moves of equal value is the spec's: fewer plies, then the mover's own outright win, then
  move order. So the stored policy is one canonical strategy per w, but at w = 0 and w = 1 it can prefer a quick loss
  to a longer draw (ruling R12; see above). A solver tie-break that prefers not losing when values are equal needs a
  change to `solve.py` and a full re-solve.
