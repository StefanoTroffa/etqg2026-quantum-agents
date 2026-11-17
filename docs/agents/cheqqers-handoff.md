# Cheqqers network agent: handoff (16 November 2026)

For anyone (person or agent) who has to run, play, benchmark or train the Cheqqers agent without Stefano's
machines. Deadline: **Tuesday 17 November 2026, 23:59 CET**.

## 1. What ships

| Piece | Where | Notes |
|---|---|---|
| Move server (both games) | `server.py` | `uv sync && PORT=8080 uv run python server.py`; prints `READY` once every move path has answered |
| Cheqqers agent | MCTS in C++ (`native/cheqqers/mcts.cpp`) + transformer (`config/cheqqers_net.pt`, sha256 `4fd1cbdb...`) | `src/search/gpu_mcts_cheqqers.py`; chance nodes valued by expectation over outcomes |
| Network | 6 layers, d_model 256, 8 heads, 4.95M parameters, policy + value (win/draw/loss) heads | `src/models/cheqqers_policy_value.py` |
| Play settings | batch 256, c_puct 1.5, fpu 0 | `PLAY_*` in `gpu_mcts_cheqqers.py`; env `CHEQQERS_MCTS_BATCH/_CPUCT/_FPU` override |
| CPU fallback | FW: expectimax alpha-beta, fitted weights `config/cheqqers_eval_weights_2026-11-13.json` | used automatically when CUDA is missing or too slow (< 5,000 evaluations a move at start-up), with a ⚠️ line |
| TiqTaqToe | Chris's exact tables (`exact/`, Q1/Q2 committed) via `src/search/exact_tiqtaqtoe.py` | 1 ms a move |

Evidence: final series against FW, 2,200 ms a move each, 100 pairs (seed 1117): 108/16/76, **+56 Elo [+18, +96]**,
0 fallbacks, 0 native errors. vs random 196/0/4. vs early checkpoint gen010: +87 [+72, +102] (2,000 games).

**Read the start-up lines.** The line `✅ Cheqqers engine: MCTS with .../cheqqers_net.pt ... on cuda` means the network
plays. A ⚠️ line naming the MCTS means FW plays instead. Torch comes from the cu128 index (`pyproject.toml`): the
NVIDIA driver must support CUDA 12.8 (driver >= 570). With an older driver torch reports "driver too old", CUDA is
unavailable and the server plays FW.

Resources (RTX 5090, one game): ~2.7 GB GPU memory per server process, ~44% GPU utilisation, ~290 W while
searching, ~90-130k network evaluations a move. CPU only (32 threads): ~7,700 evaluations a move.

## 2. Settings that matter

| Variable | Default | Effect |
|---|---|---|
| `CHEQQERS_MCTS_MODEL` | `config/cheqqers_net.pt` | another network; empty = FW only |
| `CHEQQERS_TIME_LIMIT_MS` | unset: 2,200 ms of search at the arena's 3,000 ms limit | the search time itself (the MCTS stops 150-250 ms early) |
| `CHEQQERS_MCTS_MIN_EVALS` | 5000 | below this at start-up, FW plays |
| `PORT` | 8081 | HTTP port |

**Arena lesson (16 Nov):** behind the organisers' Cloudflare tunnel one reply took > 3.5 s at the arena although
the server answered in 1,951 ms: a timeout loss. With `CHEQQERS_TIME_LIMIT_MS=1500` (moves <= ~1,350 ms) there
were no timeouts afterwards. Use 1500 when the network path to the arena is long.

## 3. Play against it yourself

```bash
PORT=8080 uv run python server.py &        # wait for READY
cd <dir with the organisers' local-cheqqers-server.js (github.com/VincenzoLipardi/ETQG_2026, Cheqqers/) and `npm install @quantumplayed/quantum-game-engine@1.0.2`>
node local-cheqqers-server.js --p1 human --p2 remote --p2-url http://127.0.0.1:8080/move
```

Pre-flight both colours against the harness's random player: `scripts/preflight_cheqqers.sh <harness dir>
http://127.0.0.1:8080/move 3 <server log>` (exit 0 only if every game is won, no forfeit).

## 4. Benchmarks

Set up the working folder once (any path; every script reads `QG_T33`):

```bash
bash scripts/cheqqers_rl/setup_layout.sh /scratch/$USER/qg_t33
export QG_T33=/scratch/$USER/qg_t33
tar -xf cheqqers_training_data.tar -C $QG_T33      # only needed for training and the gen010/gen048 references
```

On a machine other than Linux x86-64 with a recent glibc, rebuild the library first: `scripts/build_native.sh`,
then copy `native/cheqqers/libcheqqers.so` to `$QG_T33/lib/libcheqqers_t33_dev10.so`.

- **Fast: network vs network** (batched on the GPU, 2,000 games in ~2 min on a 5090):
  ```bash
  uv run python src/training/gumbel_selfplay.py --model <candidate.pt> --frozen-model <reference.pt> \
    --frozen-frac 1.0 --games 2000 --parallel 1000 --n 128 --threads 6 --seed 777 --out <dir>/b --gpu-mem-frac 0.4
  ```
  The last lines give the candidate's W/D/L against the frozen side. Use it to screen checkpoints.
- **Slow: the ship decision.** `bash $QG_T33/tools/series_final.sh <model.pt> 256 1.5 0.0 <seed> 200 3`: MCTS + network
  vs FW at 2,200 ms each, colour-balanced pairs, looks at 100/150/200 pairs only, `final/RESULT.txt` with the
  SHIP DECISION, validity (fallbacks, native errors, moves > 2,100 ms) and a sha256 re-check. ~1-2.5 h with 3 games
  at a time on one GPU. It pauses the training loop (touches `gen/STOP`). Use a seed not used before
  (1114-1117 are taken) so the openings are fresh.

## 5. Training on a cluster

The loop is expert iteration with Gumbel MCTS (recipe and sources: `docs/specs/cheqqers_rl_recipe.md`):

- generator: `gumbel_selfplay.py --continuous`, m 16 root candidates, 128 simulations a move, 20% of games
  against a frozen older model; hot-swaps to the trainer's newest checkpoint every 300 s;
- trainer: `train_concurrent.py` (AdamW 1e-4, batch 1,024, bf16), buffer = newest 1M self-play rows + FW games 5%
  + FW labels 7.5% + value anchor 10%, replay cap 6, checkpoint every 150 s to `gen/ckpt/`;
- FW pool: `fw_loop.sh`, games against FW at 2,200 ms (CPU, 1 thread a worker).

Start, from the shipped network: `bash $QG_T33/tools/gen_loop10.sh $QG_T33/candidates/day_ckpt_00225.pt`.
Stop: `touch $QG_T33/gen/STOP` (every part stops cleanly). Progress: `gen/gens.jsonl` (one tracker line every
25 min: policy head-to-head vs gen010/gen048, value cross-entropy, games/h); `uv run python $QG_T33/tools/gen_trends.py`.

The defaults are sized for one RTX 5090 + 24 cores (`PAR=1024 TH=6`, generator GPU memory 45%, trainer 20%).
On a cluster node, set `PAR`, `TH`, `GMEM` and the FW pool size (`FWN`) for the node, and run the loop inside
one job (it is several cooperating processes with pidfiles in `gen/`; it does not span nodes).

**Rules learned the hard way**
- Track processes by their pidfiles in `gen/`, never by `pgrep`/`pkill` (a pattern matched a waiting parent and
  deadlocked a panel for 40 min).
- A checkpoint replaces `config/cheqqers_net.pt` only after it (1) beats the shipped network in the fast benchmark
  and (2) wins a `series_final.sh` against FW on fresh openings with a valid RESULT. Training losses and the
  policy head-to-head are not evidence of playing strength.
- Nothing else on the GPU during a series: the agents think against the clock.

## 6. Open items

- README section 0 still partly describes the 13 November agent; this file is current.
- `server.py` still loads the old SNN/MLP/KAN TiqTaqToe models at start-up (unused; their load errors are noise).
- PR `feat/final-submission` -> main, merge, tag.
