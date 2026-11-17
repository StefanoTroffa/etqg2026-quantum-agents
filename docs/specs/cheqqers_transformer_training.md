# Cheqqers: a transformer that picks moves, trained on fast self-play

Decided by Stefano on 13 November 2026: the C++ engine first (factor 50 in positions per second), then a
transformer trained on many games so the agent picks moves with a learned judgement, not by search alone. GPU
training on stefano-pc (RTX 5090) starts only after the C++ engine passes its identity gate.

## Why a network, and why with search

Our losses against the other teams are one or two tactical moves per game just past the search horizon
(`cheqqers_best_agent.md`, section 1b). A network helps in two ways a faster search alone does not:

* **Move ordering (policy).** Alpha-beta searches the best move first when it can guess it; a good guess cuts the
  tree and buys one or two plies at the same time.
* **Leaf judgement (value).** A value learned from deep searches knows about threats the static evaluation misses.

The network does not replace the search: the agent plays the search's move, with the network as its prior and
leaf value (the AlphaZero / expert-iteration pattern). Pure policy play is measured as a baseline only.

What we already know: on 10 November an MLP value network blended into the search lost 13.5 / 40
(Elo -117, CI -251..-11). That network was trained on old rules and on weak targets; this plan changes both, and
keeps the same rule: nothing ships unless it beats the pure search at the same time per move.

## Pipeline

1. **Data from the C++ engine** (`native/cheqqers`, self-play driver). Games from random two-ply openings, both
   colours, search at a fixed node budget per move (first value 200k nodes, about 0.6 s at the target speed).
   Per position: the tournament payload, the score of every root move, the move played, the final result.
   Run on the 16 laptop cores or the 32 stefano-pc cores, one process per core. Target: 1 million positions.
2. **Encoding.** `QuantumBoardTransformerNet` in Cheqqers mode (72 inputs, 288 actions: 18 squares x 4 directions,
   6 splits, 6 merges), with a mapping between payload moves and the 288 actions, tested both ways on every
   legal move in the data. Entanglement groups added as tokens if the 72 inputs prove too little (measured, not
   assumed: compare validation loss with and without).
3. **Training on the 5090.** Policy: cross-entropy to softmax(root scores / T). Value: win/draw/loss from the
   final result blended with the search score. Validation on held-out whole games, never positions of training
   games.
4. **Use in play** (opt-in in the server, CHEQQERS_NET=path): policy orders the root and the first two plies;
   value blended at the leaves with weight w. CPU inference only: the organisers run our server on their machine.
   Each network call is timed; the search budget already includes it.
5. **Second round if time allows.** Self-play with the network-guided search, retrain (expert iteration).

## Acceptance (fixed before any run)

* Network-guided search against the pure C++ search, same wall-clock time per move on CPU, 200 game pairs
  (each opening with both colours): lower end of the 95% interval on Elo above 0. Otherwise it does not ship
  and the pure C++ search is the submission.
* Against the random player: no worse than the pure search (about 97% wins).
* Zero timeouts in the series, slowest move below 2.6 s with the 3,000 ms limit.
* The server is ready before it listens (`src/arena/warmup.py`): network loaded and has answered once.

## Order and dates

| When | What |
|---|---|
| Fri 13 night | C++ engine: rules + search, identity gate, speed (agent running) |
| Sat 14 | Best setting for the C++ search (time, depth cap, quiescence, evaluation weights) by series; self-play data generation starts |
| Sat 14 night - Sun 15 | Training on stefano-pc; series network vs pure search |
| Mon 16 12:30-14:30 | Arena: the pick and the runner-up through `agent_switch.py` |
| Tue 17 by noon | Freeze, commit, push, submit |

## Risks

* CPU speed of the organisers' machine is unknown: the time budget is fixed, so a slower machine loses depth,
  and the network's share of the budget grows.
* The C++ library must build or load on their machine. The server falls back to the Python search, loudly, if
  it does not; that fallback is the 1.2-2.2 s Python agent.
* One million positions may still be too little for a transformer; the validation loss will tell.
