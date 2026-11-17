# Quantum TiqTaqToe and Quantum Cheqqers agents (ETQG 2026)

Our agents for the follow-up of the **1st European Tournament in Quantum Games** (ETQG 2026), submitted on
17 November 2026, by Chris Chalkias and Stefano Troffa. See [CREDITS.md](CREDITS.md).

| Game | Agent | Evidence |
|---|---|---|
| **TiqTaqToe** (quantumness 2, decoherence off) | the exact solution of the game: optimal play by table lookup | every arena move value-optimal; scores equal the game values 47/64 and 17/64 |
| **Cheqqers** (6x6, quantumness 2) | Monte Carlo tree search with chance nodes and a transformer policy-value network, trained by Gumbel self-play | +56 Elo [+18, +96] (our series) and +99 Elo [+70, +129] (independent replication) against our expectimax player; 38-1-1 against the published June winner |

## Run

Linux x86-64, [uv](https://docs.astral.sh/uv/), ~1.5 GB of RAM. For the Cheqqers network: an NVIDIA GPU with at
least 4 GB free and a driver for CUDA 12.8 (570 or newer). Without one, Cheqqers is played by the expectimax
player (FW) instead, and a warning line says so.

```bash
uv sync
PORT=8080 uv run python server.py
```

One server answers both games on `POST /move`, in the format of the organisers' local testers. It prints `READY`
after every move path has answered a test move; the start-up lines say which agent plays. Time per move follows
each request's `settings.timeout` (2.2 s of search at the 3 s Cheqqers limit).

To play against it or pre-flight it you need the organisers' tester (`local-cheqqers-server.js`, from
github.com/VincenzoLipardi/ETQG_2026) and `npm install @quantumplayed/quantum-game-engine@1.0.2`:

```bash
node local-cheqqers-server.js --p1 human --p2 remote --p2-url http://127.0.0.1:8080/move
scripts/preflight_cheqqers.sh <tester dir> http://127.0.0.1:8080/move 2 <server log>
```

## How it works

**TiqTaqToe** (`exact/`, standard library only). A potential of the position rises with every move, so the game
graph is acyclic and backward induction over all reachable positions, modulo the 8 board symmetries, gives exact
values in rational arithmetic. The tournament setting has 1,643,542 canonical positions (about an hour to solve).
Tables for quantumness 1 and 2 (decoherence off) are included; the summaries of all 19 solved configurations are in
`exact/results/*.json`, and `cd exact && uv run python -m tiqtaqtoe.solve --quantumness Q --decoherence D`
regenerates any table. Method, verification and numbers: [exact/docs/results.md](exact/docs/results.md).

**Cheqqers** (`native/cheqqers/`, `src/`).
- A C++ simulator of the rules, checked against the organisers' engine (random games, replays of all arena games).
- **FW**: expectimax alpha-beta (quantum moves are chance nodes) with 18 evaluation weights fitted on game data.
- **The network agent**: a 6-layer transformer (4.95M parameters) reads 18 square tokens and one global token;
  the value head predicts win/draw/loss, the policy head scores each legal move from the squares it involves.
  MCTS with chance nodes valued by expectation, PUCT (c = 1.5, first-play urgency 0), GPU batches of 256:
  about 90,000-130,000 network evaluations in 2.2 s on an RTX 5090.
- **Training**: imitation of FW (games and 1 s search labels, DAgger), then Gumbel AlphaZero self-play (16 root
  candidates, 128 simulations a move) with a share of games against FW and older networks.
- Moves the organisers' engine would refuse (more than 15 entangled squares) are never played.

## Reproduce

- Running, benchmarking and training: [docs/agents/cheqqers-handoff.md](docs/agents/cheqqers-handoff.md);
  scripts in `scripts/cheqqers_rl/`.
- Every experiment, including what did not work (a network inside alpha-beta, an opening book from long searches,
  training from the real start position): [experiments/CONCLUDED_EXPERIMENTS.md](experiments/CONCLUDED_EXPERIMENTS.md).
- Tests: `uv run pytest tests` (457 pass) and `cd exact && uv run python -m unittest discover -s tests`.

## License

MIT, see [LICENSE](LICENSE). The organisers' engine, testers and arena are theirs and are not included.
