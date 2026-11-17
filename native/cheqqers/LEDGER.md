# Native Cheqqers engine: ledger

Every change to native/cheqqers, its speed and its identity result. Speed: the benchmark of
`~/work/_scratch/qg_cpp/bench_search.py` (6 middlegame arena positions of 13 Nov, depth 4, quantum_plies=2, 8,067
nodes), run natively by `~/work/_scratch/qg_cpp/tools/bench_native.py <arena_1113.jsonl> <passes>`; nodes/s over the
whole call from Python (native-only rate in brackets), median of the passes. Identity: the gate named, its counts.
Machine: this laptop, Ryzen 7 4800H, WSL, g++ 9.4, `-O3 -std=c++17 -ffp-contract=off`, no -march.

Target: >= 347,000 nodes/s with the same moves and the same 8,067 nodes. Python baseline: 6,950 nodes/s (lead,
13 Nov); re-measured 2026-11-13 21:00: 7,015 / 7,038 / 6,660 / 7,009 nodes/s, same moves.

| date-time (CET) | change | nodes/s median (passes) | speed-up vs 6,950 | identity |
|---|---|---|---|---|
| 2026-11-13 20:55 | A+B: first full port (engine.cpp, search.cpp, api.cpp), std::vector state, unique_ptr children | 261,758 (5) [native-only 264,458] | 37.7x | bench: 8,067 nodes, same 6 moves; gate 1 pytest 129 passed |
| 2026-11-13 21:10 | C1: check_win_states tests "any move" with early exit (it only reads emptiness); resolve() reads the cached move list instead of copying it; stable insertion sorts instead of std::stable_sort (no temp buffers); capture-depth simulations cached (classical squares only: same numbers) | 296,110 / 293,838 / 296,240 (3x7) -> 296,110 | 42.6x | bench 8,067 nodes, same moves; pytest tests/native 200 passed |
| 2026-11-13 21:14 | C2: shared-library codegen: -fvisibility=hidden, -fno-semantic-interposition, scratch reached through an initial-exec TLS pointer (was a 256 KB thread_local: __tls_get_addr per access in a dlopen'ed .so) | 323,983 / 327,920 / 329,880 (3x7) -> 327,920 | 47.2x | bench 8,067 nodes, same moves (pytest after C3) |
| 2026-11-13 21:13 | C3: allocation churn: per-thread pool of Env objects (clones and capture-depth simulations reuse an old Env and its vectors' capacity), scratch move lists kept in the Env, outcome branches and joint-system buffers in the per-thread scratch (results swapped in, old storage reused), per-ply outcome/legal-move buffers in the search | 477,172 / 456,774 / 467,150 (3x7) -> 467,150 [C++ driver 492k] | 67.2x | bench 8,067 nodes, same moves; pytest tests/native 200 passed (C2+C3) |
| 2026-11-13 21:16 | gate 1 full sweep (scripts/native_identity_sweep.py rules): 599 arena games of 13 Nov + 44 fixture games (18,925 positions) + 210 seeded random games (200 6x6 at quantumness 0-3, 10 8x8; 6,604 positions), cache off and on, native replay after every move | - | - | 387,712 position + move-outcome checks, 0 mismatches |
| 2026-11-13 21:25 | server.py: CHEQQERS_ENGINE=cpp (default python) calls native best_move with move_budget's time and depth and the same options; failure -> Python with the time left; library loaded at start | - | - | server on a test port: warm-up + 6 benchmark payloads answered by engine=cpp, all legal, 1,984-1,994 ms at the 3,000 ms limit (2,200 ms budget x 0.9); machine loaded by the sweep |
| 2026-11-13 21:35 | cq_selfplay: whole games in C++ (JSONL per position: features, presences, pieces, moves, root scores, move, depth, nodes, optional flat state) | - | - | test: recorded states re-import with the same legal moves; Python search on them plays the recorded move with the same nodes |
| 2026-11-13 21:50 | Scope change (Stefano via lead): native is THE engine. CHEQQERS_ENGINE default cpp (python only as explicit opt-in); server refuses to start (SystemExit before warm-up and port binding) if the library cannot be built/loaded, or if CHEQQERS_NN_BLEND>0; no mid-move Python fallback: a search error stops the search and returns the last finished iteration's move (ABI 2: CqResult.failed), payload saved to logs/cheqqers_native_failures/, else legalMoves[0]; CHEQQERS_NATIVE_BUILD=0 forbids compiling | - | - | tests/arena/test_native_engine_serving.py (refuses to start; failure path answers legal and never calls the Python search; default engine) + test_warmup.py: 7 passed. The new test caught a missing module-level `import time` in server.py (fixed) |
| 2026-11-13 21:40 | measured only: -march=native build (CQ_MARCH=native) vs portable | portable 458,241 / 466,320 / (316,056 outlier under load); native 464,683 / 463,321 / 459,767 | ~66x either | same 8,067 nodes and moves; no gain worth a CPU-specific binary: portable kept |
| 2026-11-13 21:44 | FINAL (portable build, ABI 2, all changes above) | 467,482 / 467,205 / 471,910 / 466,530 (4x9) -> 467,344 | 67.2x (66.6x vs re-measured 7,012) | bench 8,067 nodes, same 6 moves; full suite 399 passed, 5 skipped. Deeper runs of the same positions (3 passes): depth 5 414,943 nodes/s (59.7x), depth 6 433,447 (62.4x) |
| 2026-11-13 22:05 | gate 2 full sweep (scripts/native_identity_sweep.py search, C3 search code), started 21:20: 6 benchmark + 260 arena positions (every 45th ply of the 599 + 44 games), default options and quantum_plies=2, depth 4, no time limit | - | - | 532 searches, 17,820,003 nodes: same move, nodes, last depth and score bits in all, 0 mismatches (Python 6,013 s vs native 64 s of worker time) |
| 2026-11-14 00:05 | Speed pass 2. How measured: official bench (`bench_native.py`, 9 passes) with each library snapshot (`~/work/_scratch/qg_cpp/speed2/libs/{base,step1..4}.so` via CHEQQERS_NATIVE_LIB) run in turn, 3 rounds, during a quiet window (load 2.7-4.0); "median" below = median of the 3 round medians. Rounds slowed 30-40% by the other agent's load were seen for base, step1 and step2 (one each), so read the rows together with the interleaved C++-driver best-of (5 rounds x 60 reps, load ~12) given in each row. The same-session base measured 440,650 (rounds 440,650 / 282,367 / 442,062), C++ best-of ~430k | - | - | - |
| 2026-11-14 00:05 | S1 move generation: the capture scan is made once per position and colour (`takes_of`, cache on) and every caller filters it: the turn's captures for the move list, `capture_pending`, the any-move test; `eval_features` merges both colours by square (= the scan of both colours). Piece masks per colour plus a mask of squares with an empty chance (`scan_prep`), so scans visit only those colours' pieces. `has_any_move_skip_depth` tests plain moves first, as one mask test per piece (the order of the two scans does not change the answer); the capture scan runs only if there is no plain move. `find_moves` per-piece body shared (`piece_moves`), each flag read only in its branch. Search: quantum_plies filter in place (was one new vector per node); `resolve` compares fields without `to_array` | 446,830 (446,830 / 463,227 / 339,770) [C++ best-of ~455k, +6% vs base] | 64.3x; 0.96x vs 467,344 (+1.4% vs same-session base) | bench 8,067 nodes, same 6 moves; gate 1 rules 387,712 checks 0 mismatches; gate 2 official search sweep 532 searches, 17,820,003 nodes, 0 mismatches (3,514 s wall); tests/native+arena 277 passed |
| 2026-11-14 00:05 | S2 probabilities: each joint system keeps the get_target_probabilities sums of all its entity indices (`System::compute_sums`, one pass, the same additions in the same order), reset where its amplitudes or entities change (`touch()`), copied with the system on clone; `square_probs`, `target_probs`, `disentangle` and measurements read them; independent squares inline in `square_probs`. `disentangle` invalidates only when it changes a square, `cleanup_props` only when it removes a piece (otherwise the same position, same numbers); no per-system vector copy in `disentangle` | 529,990 (326,461 / 529,990 / 536,854) [C++ ~495k, +8% vs S1] | 76.3x; 1.13x vs 467,344 | bench 8,067 nodes, same moves; gate 1 rules 0 mismatches; gate 2 search 532 searches, 17,820,003 nodes, 0 mismatches (Python side cached from this first run: `speed2/tools/sweep_search_cached.py`, same positions and comparison as the sweep; Python unchanged); tests/native+arena 277 passed |
| 2026-11-14 00:05 | S3 layout/hashing/memory: board key computed once per position (`bkey`, made by check_win_states, read by the search's state_key; dropped with the move caches, checked against turn); transposition table = open addressing (linear probing, exact key compare on the stored codes, generation-counter clear, insert only while size < max_tt_size, so the same map as the unordered_map); joint-system storage recycled (tensor products in scratch buffers, dropped systems' vectors reused by new systems). Not done: compact history keys (2.6 entries per clone in the benchmark: not worth it) | 591,040 (387,031 / 591,040 / 594,894) [C++ ~533k, +8% vs S2] | 85.0x; 1.26x vs 467,344 | bench 8,067 nodes, same moves; gate 1 rules 0 mismatches; gate 2 search 532 searches 0 mismatches (cached Python); tests/native+arena 277 passed |
| 2026-11-14 00:05 | S4 classical fast path: a plain move of a certain piece ({1: 1+0j}) to a certain-empty square ({0: 1+0j}), or a single capture with three such squares, when no reached joint system lists any of them (not even stale), in SUPERPOSITION/ENTANGLEMENT: set the squares' final amplitudes directly instead of building the joint system that disentangle dissolves at the end of the same move (`Env::classical_set`, apply_classical, apply_take). 18.6k of 41k apply_classical calls at depth 6 take it | 613,618 (613,618 / 606,017 / 615,686) [C++ ~556k, +4% vs S3] | 88.3x; 1.31x vs 467,344 (1.39x vs same-session base; C++ 1.29x) | bench 8,067 nodes, same moves; gate 1 rules 0 mismatches; gate 2 search 532 searches 0 mismatches (cached Python); golden native-vs-base 5,893 searches / 155 M nodes (9 option sets incl. depth 6, no cache, small TT) 0 mismatches; tests/native+arena 277 passed; full suite 425 passed, 5 skipped (00:20, repo library = S4 engine + the value-net additions below) |
| 2026-11-14 00:05 | REVERTED S5: carry the square probabilities through clone_into and update them in the classical fast path (finalize/apply_classical stopped dropping them; phase_pass only for INTERFERENCE) | - | - | gate 1 rules: mismatches in random game 57 (6x6, quantumness 1 = SUPERPOSITION), cache on, positions 4/5/8: "child positions differ". Search golden was 0 mismatches. Reverted, not investigated further |

## Can it go faster? (measured 2026-11-13 21:50-22:05, final portable build)

### Where the time goes
gprof, static -pg driver (`~/work/_scratch/qg_cpp/prof/driver.cpp`, -O2, inlining partly off, so ~300k nodes/s
under the profiler), 6 benchmark positions, quantum_plies=2; self time grouped by stage
(`~/work/_scratch/qg_cpp/tools/stages.py`):

| stage | depth 4 (300 reps, 4.45 s sampled) | depth 6 (25 reps, 2.37 s sampled) |
|---|---|---|
| move generation (find_moves, split/merge lists, capture-depth simulations, resolve) | 28.3% | 35.4% |
| clone / apply bookkeeping / memory (clone_into, finalize, cleanup, check_win_states own work) | 22.5% | 19.0% |
| joint-system probability sums (square_probs, probs_direct, target_probs) | 18.7% | 19.4% |
| quantum ops, measurement, disentangle | 14.4% | 11.4% |
| evaluation (eval_features) | 7.6% | 4.6% |
| hash, repetition history, TT (board_key x2 per node, unordered_map with 104-byte keys) | 6.5% | 8.9% |
| search control (decision/chance frames) | 1.8% | 0.8% |

Per node at depth 6 (1.38 M nodes): find_moves 3.3 calls, square_probs 8.2 calls (recomputed after every
invalidation), probs_direct 43 calls, clone 1.0, disentangle 1.3, board_key 1.4.

After speed pass 2 (S4 build): PC sampling of the -O3 code (`~/work/_scratch/qg_cpp/speed2/sprof/sprof.sh`, SIGPROF
sampler linked into the static driver, outermost non-inlined function), depth 4 / depth 6: clone/apply bookkeeping
(clone_into, move_outcomes, finalize, check_win_states, memmove) 25% / 25%; move generation (masked scans, any-move,
split/merge, resolve) 17% / 15%; probability sums (square_probs, compute_sums) 13% / 12%; quantum ops + disentangle
11% / 11%; evaluation 6% / 6%; board key 5% / 5%; search control and ordering 6% / 9%. The largest single items:
square_probs 10%, clone_into 9.5%, the capture scans 9-10%.

### What one more ply costs (native, iterative-deepening totals over the 6 positions)
quantum_plies=2: nodes x4.5, x3.5, x2.9, x2.7, x2.5, x2.6, x2.3, x2.2 from depth 1->2 ... 8->9; time per ply x2.2-2.6
from depth 5 on (x3.6 once, 6->7). Default options (three-square moves at every ply): x5.5-7.1 per ply up to depth 6,
x12 from 6 to 7.
Within the real budget (2,200 ms, deadline 1,980 ms), per position: quantum_plies=2 finishes depth
13/10/14/14/15/9 with 0.90-1.07 M nodes; default options 6/6/8/7/9/5 with 0.80-0.85 M nodes (410-540 k nodes/s in
these longer searches). So at quantum_plies=2 each factor 2 of speed buys about 0.8 ply (log 2 / log 2.4), a factor
4 about 1.6 plies; with default options a factor 2 buys about 0.4 ply. Python at 7 k nodes/s searched ~14 k nodes
in the same time: depth 6-7 at quantum_plies=2 (estimate from these node counts).

### Remaining levers, ranked (estimates, not measured)
Same results (identity kept):
1. Move generation as masks (bitboards of certain-empty / possibly-occupied / colour per position, computed once
   per position) and one shared take-move scan for check_win_states, capture_pending, eval_features and the next
   decision node: 28-35% of the time -> est. 1.3-1.5x overall.
2. Incremental probabilities: recompute P0/P1 only for the squares of systems an operation touched (independent
   squares are two products); today 8 recomputations per node: est. 1.15-1.2x.
3. Compact position layout for N <= 18 (Env is ~7 KB sized for 10x10; history keys 104 bytes), board key computed
   once per node and reused by check_win_states and the TT probe, open-addressing TT: est. 1.15-1.3x together.
4. Make/unmake instead of clones: hard here (joint systems are rebuilt, not updated) and partly covered by the pool;
   est. 1.1x for much work. Not recommended.
Together, est. 1.5-2.5x: +0.5-1 ply at quantum_plies=2.
Changes results (the moves differ from the Python's at the same depth, so identity tests no longer apply; each needs
its own colour-balanced series, CW3):
5. Parallel search over the cores: root splitting or lazy SMP with a shared table. On this 16-core laptop est. 4-8x
   (root of 8-30 moves, uneven subtrees) = +1.5-2.5 plies at quantum_plies=2. The organisers' machine's core count is
   unknown; with 2 cores ~1.6x.
6. Aspiration windows / principal-variation search at decision nodes: est. 1.2-1.5x fewer nodes (Star1 chance
   nodes limit the gain).
7. Star2 probing at chance nodes, late-move reductions: est. up to 2x nodes, with a strength risk to measure.
The largest lever already in hand is quantum_plies=2 itself (branching ~2.4 instead of ~6 per ply).

## Value network at the leaves (2026-11-14, additive; default path unchanged)
`value_net.{h,cpp}`: a small ReLU MLP (`src/training/train_cheqqers_value_mlp.py`'s value_mlp.json, read in Python by
`native_cheqqers.ValueNet` and passed as flat arrays through `cq_net_new`). `SearchOpts::net` (null = linear
evaluation); `Searcher::leaf` calls `net->value(env, root_color)` = +-900 tanh(logit/2) (side to move's logit, negated
for the other side) on non-terminal leaves, terminal leaves keep Env::evaluate's +-1000/0. New entry points
`cq_best_move_net`, `cq_selfplay_net` (`cq_best_move` / `cq_selfplay` call them with NULL), `cq_net_new/free/inputs/
forward/encode`; CqOpts and ABI 2 unchanged; the wrapper binds the new symbols only if present, so older libraries
still load. Python: `best_move(..., value_net=)` / `selfplay(..., value_net=)`, None = CHEQQERS_VALUE_NET if set,
"" = none; hub spec `net=<file>` (with cpp=1 only; hub agents never read CHEQQERS_VALUE_NET).

| when | change | nodes/s | vs before | checks |
|---|---|---|---|---|
| 2026-11-14 00:07 | value net 144-64-32-1 at the leaves (first layer as column sums over the non-zero inputs, double) | d4 q2 bench: 416,451 with net vs 626,430 fitted weights (-34%); series setting (2,200 ms, no qp): 324-344k vs 500-538k | same depth finished on all 6 bench positions (7/6/8/7/10/5) | tests/native/test_value_net.py 7 passed (encoding bit-equal to encode_position on 1,451 fixture positions + native self-play records; forward vs numpy f64 max 2.7e-15 on 3,684 positions); existing tests/native 201 passed; gate 1 rules sweep 387,712 checks 0 mismatches; gate 2 search sweep 532 searches 0 mismatches (lib sha 89a406e5) |
| 2026-11-14 03:25 | track 32 (additive): residual leaf mode (`ValueNet::set_residual`, `cq_net_set_residual`: logit = lpm * (w . features) + lambda * net, JSON `{"residual": {"lambda": x}}`); leaf sampler (`SearchOpts::leaf_sink`, `cq_selfplay_leaves`: each non-terminal leaf with probability leaf_rate, its own generator; with write_state also the leaf's flat state); `encode_position` shared by the net and the sampler; `cq_selfplay_net` = `cq_selfplay_leaves` with rate 0 | residual 144-32-1: -25% nodes/s in the 300 ms proxy (8 workers), 144-16-1 -19%, linear 144-1 -11% | - | fixed-depth searches against the frozen track-31 library (`libcheqqers_net_2026-11-14_v1.so`) on 400 proxy positions: FW depth 4 and track-31 net depth 3 byte-identical output (moves, nodes, depth, root scores; 35.6 M nodes); zero residual (lambda 0 and 1) = FW in 80/80 searches; random residual net depth 1 vs numpy 5.7e-13; leaf sampling leaves the games unchanged; leaf states re-import to the same input (806/806). Repo library rebuilt from these sources (sha256 6c767ae1...); tests/native + tests/arena 284 passed |

## Track 33 (2026-11-14, additive; the alpha-beta search and its default path unchanged)
New sources `mcts.{h,cpp}` (built into the same library by scripts/build_native.sh; native_cheqqers.SOURCES lists
them) and new C entry points appended to `api.cpp`; no existing function changed.

| when | change | checks |
|---|---|---|
| 2026-11-14 05:05 | `cq_encode_position`; MCTS with chance nodes by expectation for a network run by the caller (`cq_mcts_new/collect/apply/root/free`, many trees on a persistent thread pool: `cq_mcts_collect_many/apply_many`); batched self-play stepping (`cq_sp_*`) | gate 2 search sweep (cached Python) 532 searches, 17,820,003 nodes, 0 mismatches; native golden vs the pre-track library (`6c767ae1...`) 5,893 searches 0 mismatches; tests/native + tests/arena 287 passed; tests/native/test_gpu_mcts.py (chance-node Q = expectation over Python get_move_outcomes; many trees = single trees); ASan/UBSan standalone driver clean |
| 2026-11-14 06:40 | MCTS selection skips moves whose outcomes are pending at the network (was: collisions filled batches with 1-3 rows in losing positions); Gumbel root (Gumbel-Top-m + Sequential Halving), completed-Q interior selection and policy targets (`MctsOpts::gumbel_*`, `cq_mcts_gumbel`, `cq_mcts_done`); Gumbel self-play batches with refill, frozen-network and FW opponents (`cq_gz_*`) | tests/native/test_gumbel_mcts.py (target = softmax(logits + sigma(completed Q)) recomputed in Python; budget; single winner; top-m; 15 visits with 16 candidates; seeds; generator samples; frozen/FW sides) |
| 2026-11-14 07:50 | FIX (dev10): a chance node could get no children when every outcome probability was below the 1e-9 filter (their sum > 0); the next descent indexed an empty vector (out of bounds). Now all outcomes with p > 0 are kept in that case; an empty chance node raises | full suite tests/native + tests/arena + tests/training 335 passed (library `libcheqqers_t33_dev10.so`); FW identity on dev10: gate 2 search sweep 532 searches, 17,820,003 nodes, 0 mismatches; native golden vs the pre-track library 5,893 searches 0 mismatches |
| 2026-11-14 16:00 | identity on stefano-pc (the machine of the panels and the series): its own build of dev10 (sha256 de65843d...) vs its own build of the pre-track sources (commit 222dc7a, sha256 282cd55a...) | gate 2 search sweep 532 searches 0 mismatches; native golden 5,893 searches 0 mismatches. Across machines (laptop build vs PC build of the same sources): 18 of 5,893 golden searches differ, all in the squashed-evaluation option set ("logit"): root scores differ in the last bit (1 ulp; tanh from different libm), 4 of them with a different node count (ties in move ordering). A platform effect of the fitted evaluation's squash, the same before and after this track |
