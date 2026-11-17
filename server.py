import http.server
from pathlib import Path
import json
import random
import time
import torch
torch.set_num_threads(1)
import os
import sys

# Force current working directory to the directory of this file to make all relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.cheqqers_official import OfficialCheqqersEnv
from search.expecti_cheqqers import ExpectiCheqqersPlayer, eval_weights_from_environment
from search.opening_book import book_from_environment
from arena.move_budget import cheqqers_budget

# The tournament plays the fitted evaluation (fit of 13 Nov 2026 on 9,944 self-play games under the 11-Nov rules).
# Series of 13 Nov, C++ engine, 2.2 s a move, 200 colour-balanced pairs each: +96 Elo (95% CI +65..+129) over the
# default weights; adding three-square moves in the first 2 plies only gave -10 (-41..+20) on top of it
# (experiments/CONCLUDED_EXPERIMENTS.md, track 30). An explicit CHEQQERS_EVAL_WEIGHTS wins; set it empty for the
# default evaluation.
DEFAULT_CHEQQERS_EVAL_WEIGHTS = "config/cheqqers_eval_weights_2026-11-13.json"
os.environ.setdefault("CHEQQERS_EVAL_WEIGHTS", DEFAULT_CHEQQERS_EVAL_WEIGHTS)


def cheqqers_engine() -> str:
    """Which engine searches Cheqqers. The native one (native/cheqqers via src/search/native_cheqqers.py: the Python
    search's moves at the same depth, ~67x faster) is the engine; CHEQQERS_ENGINE=python is an explicit opt-in for
    experiments only. The Python search is never a fallback of the native one."""
    return "python" if os.getenv("CHEQQERS_ENGINE", "cpp").strip().lower() == "python" else "cpp"


def cheqqers_search_options() -> dict:
    """The search options both engines take from the environment."""
    return dict(
        eval_weights=eval_weights_from_environment(),  # None: the default evaluation
        opening_book=book_from_environment(),  # None: every move is searched
        # Quiescence on pending captures (Track-Q): off unless CHEQQERS_QUIESCENCE=1
        quiescence=os.getenv("CHEQQERS_QUIESCENCE", "0") == "1",
        max_capture_extension=int(os.getenv("CHEQQERS_CAPTURE_EXTENSION", 4)),
        # Three-square moves only in the first plies of the search (Track-Q step 2): unset = everywhere
        quantum_plies=(int(os.environ["CHEQQERS_QUANTUM_PLIES"])
                       if os.getenv("CHEQQERS_QUANTUM_PLIES", "").strip() else None),
    )


def _report_native_failure(data: dict, what: str) -> None:
    """A native search that raises is a bug: say so loudly and keep the payload to reproduce it."""
    folder = Path(os.getenv("CHEQQERS_FAILURE_DIR", "logs/cheqqers_native_failures"))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.perf_counter_ns()}.json"
        path.write_text(json.dumps({"error": what, "payload": data}))
        where = f"payload saved to {path}"
    except OSError as save_err:
        where = f"payload not saved ({save_err})"
    print(f"❌❌ NATIVE CHEQQERS SEARCH FAILED: {what}; {where}", flush=True)


def native_cheqqers_move(data: dict, legal_moves: list, player, t_req_start: float,
                         time_limit_override_ms: float | None = None):
    """The move the native engine answers with. If it raises (a bug), the best move of its last finished iteration
    when it has one, else legalMoves[0]: never the Python search. time_limit_override_ms (MCTS fallback only): search
    for that long instead of the request's budget (the time the MCTS left)."""
    from search import native_cheqqers

    try:
        env = OfficialCheqqersEnv.from_tournament_json(data)
        # The budget follows the limit the request states (settings.timeout): src/arena/move_budget.py
        time_limit_ms, max_depth = cheqqers_budget(data.get("settings"))
        if time_limit_override_ms is not None:
            time_limit_ms = time_limit_override_ms
        result = native_cheqqers.best_move(env, time_limit_ms, max_depth, available_moves=legal_moves,
                                           **cheqqers_search_options())
        if result.error:
            _report_native_failure(data, f"{result.error} (answered with the depth-{result.last_depth} move)")
        if result.move in legal_moves:
            elapsed_ms = (time.perf_counter() - t_req_start) * 1000.0
            print(f"🤖 [Cheqqers Official] size={env.board_size} q={env.quantumness} player={player} | engine=cpp | "
                  f"depth={result.last_depth}/{max_depth} | nodes={result.nodes} | move={result.move} | "
                  f"latency={elapsed_ms:.1f}ms", flush=True)
            return result.move
        _report_native_failure(data, f"native move {result.move} is not one of the legal moves")
    except Exception as err:
        import traceback
        _report_native_failure(data, f"{type(err).__name__}: {err} | {traceback.format_exc()}")
    return legal_moves[0]


# ── Network-guided MCTS (track 33; docs/specs/cheqqers_gpu_mcts.md, shipping rule) ─────────────────────────────────
# CHEQQERS_MCTS_MODEL=<model.pt>: Cheqqers moves come from the MCTS with that policy-value network on the GPU
# (search/gpu_mcts_cheqqers.py), within each request's budget (move_budget). CUDA missing, the network failing to load,
# or a warm-up search slower than the budget: the server says so and plays FW (the native search) as before.
# Unset: the network shipped in the repo (DEFAULT_CHEQQERS_MCTS_MODEL), so `uv run python server.py` plays the submitted
# agent with no configuration. Empty (CHEQQERS_MCTS_MODEL=): nothing here runs and the FW path is unchanged.
DEFAULT_CHEQQERS_MCTS_MODEL = "config/cheqqers_net.pt"
_MCTS_PLAYER = None
MCTS_MIN_EVALS = 5000  # warm-up floor at the arena budget (CHEQQERS_MCTS_MIN_EVALS overrides)
MCTS_FALLBACK_FLOOR_MS = 50.0  # the FW fallback after an MCTS error always gets at least this


def _mcts_settings_ok(env, data: dict) -> bool:
    """The network knows only the tournament configuration: 6x6, quantumness 2, 2 start rows. A payload carries the
    board itself (from_tournament_json leaves start_rows at 0), so start rows are checked only when the payload states
    them (settings.startRows)."""
    settings = data.get("settings") or {}
    rows = settings.get("startRows", data.get("startRows"))
    return env.board_size == 6 and int(env.quantumness) == 2 and (rows is None or int(rows) == 2)


def mcts_model_path() -> str:
    """The network for the MCTS: CHEQQERS_MCTS_MODEL if set (empty = MCTS off), else the shipped network. A relative
    path is resolved against the repo, so the server can be started from any directory."""
    path = os.environ.get("CHEQQERS_MCTS_MODEL", DEFAULT_CHEQQERS_MCTS_MODEL).strip()
    if path and not Path(path).is_absolute():
        path = str(Path(__file__).resolve().parent / path)
    return path


def setup_cheqqers_mcts() -> None:
    """Load and warm the MCTS player at start-up (before the warm-up moves and the public port), or fall back."""
    global _MCTS_PLAYER
    path = mcts_model_path()
    if not path:
        return
    device = os.getenv("CHEQQERS_MCTS_DEVICE", "").strip() or "cuda"
    try:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        import hashlib

        from search import native_cheqqers
        from search.gpu_mcts_cheqqers import get_mcts_player

        # the same construction path as the series agents (settings from CHEQQERS_MCTS_* or the panels' defaults);
        # it warms the player up (CUDA initialisation, graph capture)
        player = get_mcts_player(path, device=device)
        start = native_cheqqers.NativeEnv.new(6, 2, 2)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        budget_ms, _ = cheqqers_budget({"timeout": 3000})
        min_evals = int(os.getenv("CHEQQERS_MCTS_MIN_EVALS", MCTS_MIN_EVALS))
        t0 = time.perf_counter()
        result = player.search(start, None, time_ms=budget_ms)
        took = (time.perf_counter() - t0) * 1000.0
        if took > budget_ms or result.elapsed_ms > budget_ms:
            raise RuntimeError(f"warm-up search took {took:.0f} ms (search {result.elapsed_ms:.0f} ms), over the "
                               f"{budget_ms:.0f} ms budget")
        if result.evals < min_evals:  # a slow GPU: MCTS with this few evaluations is much weaker than FW
            raise RuntimeError(f"only {result.evals} network evaluations in {budget_ms:.0f} ms at the arena budget "
                               f"(floor {min_evals})")
        _MCTS_PLAYER = player
        print(f"✅ Cheqqers engine: MCTS with {path} (sha256 {sha}) on {device}: batch {player.batch}, c_puct "
              f"{player.c_puct}, fpu {player.fpu}; {result.evals} network evaluations in {took:.0f} ms at the arena "
              f"budget (floor {min_evals}); FW (native search) on any error and for any other board than 6x6, 2 start "
              f"rows, quantumness 2.", flush=True)
    except Exception as err:
        _MCTS_PLAYER = None
        print(f"⚠️ The Cheqqers MCTS ({path}) cannot be used ({type(err).__name__}: {err}): "
              f"Cheqqers falls back to FW (the native search).", flush=True)


def mcts_cheqqers_move(data: dict, legal_moves: list, player, t_req_start: float):
    """The MCTS move within the request's budget. Other settings than the tournament's (or a payload the engine
    cannot read): straight to FW before any MCTS work. On an MCTS error: FW with the time left (at least
    MCTS_FALLBACK_FLOOR_MS), so the reply stays inside the request's limit."""
    try:
        env = OfficialCheqqersEnv.from_tournament_json(data)
        ok = _mcts_settings_ok(env, data)
    except Exception:
        ok = False
    if not ok:
        return native_cheqqers_move(data, legal_moves, player, t_req_start)
    # The opening book (CHEQQERS_OPENING_BOOK; build-net: the network's own long search): a booked position is
    # answered from it; a position not in it, or a stored move that is not legal, is searched as before.
    book = book_from_environment()
    if book is not None:
        booked = book.lookup(env, legal_moves)
        if booked is not None:
            elapsed_ms = (time.perf_counter() - t_req_start) * 1000.0
            print(f"🤖 [Cheqqers Official] player={player} | engine=book | move={booked} | latency={elapsed_ms:.1f}ms",
                  flush=True)
            return booked
    time_limit_ms, _ = cheqqers_budget(data.get("settings"))
    try:
        result = _MCTS_PLAYER.search(env, legal_moves, time_ms=time_limit_ms)
        if result.move in legal_moves:
            elapsed_ms = (time.perf_counter() - t_req_start) * 1000.0
            print(f"🤖 [Cheqqers Official] player={player} | engine=mcts | evals={result.evals} | "
                  f"move={result.move} | latency={elapsed_ms:.1f}ms", flush=True)
            return result.move
        _report_native_failure(data, f"MCTS move {result.move} is not one of the legal moves (FW answers)")
    except Exception as err:
        import traceback
        _report_native_failure(data, f"MCTS: {type(err).__name__}: {err} | {traceback.format_exc()} (FW answers)")
    left_ms = time_limit_ms - (time.perf_counter() - t_req_start) * 1000.0
    return native_cheqqers_move(data, legal_moves, player, t_req_start,
                                time_limit_override_ms=max(MCTS_FALLBACK_FLOOR_MS, left_ms))
def refused_by_engine(env, move) -> bool:
    """Whether the organisers' engine would refuse `move` (QuantumLimitExceeded: more than 15 entangled squares).
    Their harness then leaves the position as it is with the same side to move, and asks again: a deterministic
    agent that chose it chooses it again, and the game never ends (seen 17 Nov 2026: 8,453 repeats of one move).
    Our simulator models the refusal the same way, so here a refused move is one after which the same side is to
    move with no capture sequence pending and the game not over."""
    child = env.clone()
    turn = child.turn
    child.apply_move(list(move))
    return child.turn == turn and child.multi_jump_square is None and not child.done


def playable_cheqqers_moves(data: dict, legal_moves: list) -> list:
    """`legal_moves` without the moves the engine would refuse; all of them if every move would be (or if the
    check itself fails: the request is answered as before)."""
    if len(legal_moves) <= 1:
        return legal_moves
    try:
        env = OfficialCheqqersEnv.from_tournament_json(data)
        keep = [m for m in legal_moves if not refused_by_engine(env, m)]
    except Exception as err:
        print(f"⚠️ [Cheqqers] refused-move check failed ({type(err).__name__}: {err}); all legal moves kept", flush=True)
        return legal_moves
    if keep and len(keep) < len(legal_moves):
        dropped = [m for m in legal_moves if m not in keep]
        print(f"ℹ️ [Cheqqers] dropped {dropped}: the engine would refuse them (entanglement limit)", flush=True)
    return keep or legal_moves


from arena.warmup import warm_up
try:
    from search.exact_tiqtaqtoe import ExactTiqTaqToePlayer
    from fractions import Fraction as _Fraction
    # TTT_EXACT=false disables the exact tables; TTT_EXACT_W is what a tie or draw is worth to us (0, 1/4, 1/2, 3/4, 1)
    exact_ttt_player = (ExactTiqTaqToePlayer(w=_Fraction(os.getenv("TTT_EXACT_W", "1/2")))
                        if os.getenv("TTT_EXACT", "true").lower() == "true" else None)
except Exception as _exact_err:
    print(f"⚠️ Exact TiqTaqToe tables unavailable ({_exact_err!r}); TiqTaqToe will play random legal moves.")
    exact_ttt_player = None
_ttt_random_warned = False  # the TiqTaqToe random-move fallback says so once


def random_tiqtaqtoe_move(legal_moves: list, reason: str):
    """The exact tables gave no legal move: a uniformly random legal move (never forfeit, never block)."""
    global _ttt_random_warned
    if not _ttt_random_warned:
        _ttt_random_warned = True
        print(f"⚠️ [TiqTaqToe] exact tables not used ({reason}): answering uniformly random legal moves "
              f"(this line is printed once).", flush=True)
    return random.choice(legal_moves)


# The optional value network of the Python Cheqqers search (CHEQQERS_ENGINE=python with CHEQQERS_NN_BLEND > 0; an
# experiment only), loaded on first use and reloaded when the file changes.
_cheqqers_value_nets = {}


def cheqqers_value_net(size: int):
    from models.mlp import MLPNet

    path = f"results/q_net_official_cheqqers_size{size}_q2_best.pth"
    mtime = os.path.getmtime(path)
    cached = _cheqqers_value_nets.get(size)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    num_active = 13 if size == 5 else (18 if size == 6 else (32 if size == 8 else 50))
    in_dim = num_active * 4
    out_dim = num_active * 4 + num_active * 6 + num_active * 6
    # Detect architecture from state_dict keys (decoupled/wdl MLP vs simple MLP)
    state_dict = torch.load(path, map_location="cpu")
    if any(k.startswith("shared_net.") for k in state_dict.keys()):
        is_wdl = ("value_head.2.weight" in state_dict and state_dict["value_head.2.weight"].shape[0] == 3)
        model = MLPNet(in_dim=in_dim, hidden_dims=[128, 128, 128], out_dim=out_dim, decoupled=True, wdl=is_wdl)
    else:
        model = MLPNet(in_dim=in_dim, hidden_dim=128, out_dim=out_dim)
    model.load_state_dict(state_dict)
    model.eval()
    _cheqqers_value_nets[size] = (mtime, model)
    print(f"Loaded official Phase 3 Cheqqers model for size {size} from {path}")
    return model


class SNNBackendHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    # Handle lowercase HTTP/2 method names forwarded by proxies/ngrok (prevents 501 on 'options /move')
    do_options = do_OPTIONS

    def do_GET(self):
        if self.path in ['/', '/health', '/move']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "status": "ok",
                "ai": "my-ai",
                "endpoint": "POST /move",
                "supports": ["tiqtaqtoe", "cheqqers"]
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/move':
            import time
            t_req_start = time.perf_counter()
            # Tournament /move endpoint
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            board = data.get('board', [])
            legalMoves = data.get('legalMoves', [])
            player = data.get('player', 'O')

            # 1. Fallback safety net if no moves
            if not legalMoves:
                self.send_success_response({"move": []})
                return

            # 2. Check if the game is Cheqqers
            is_cheqqers = False
            if len(board) > 0 and isinstance(board[0], dict) and 'piece' in board[0]:
                is_cheqqers = True

            if is_cheqqers and cheqqers_engine() == "cpp":
                legalMoves = playable_cheqqers_moves(data, legalMoves)
                if _MCTS_PLAYER is not None:
                    self.send_success_response({"move": mcts_cheqqers_move(data, legalMoves, player, t_req_start)})
                    return
                self.send_success_response({"move": native_cheqqers_move(data, legalMoves, player, t_req_start)})
                return

            if is_cheqqers:  # CHEQQERS_ENGINE=python: the Python search, for experiments
                try:
                    env = OfficialCheqqersEnv.from_tournament_json(data)
                    # The budget follows the limit the request states (settings.timeout): src/arena/move_budget.py
                    time_limit_ms, max_depth = cheqqers_budget(data.get("settings"))
                    official_cheq_path = f"results/q_net_official_cheqqers_size{env.board_size}_q2_best.pth"
                    cheq_model = None
                    # Pure search by default: over 40 games on 10 Nov 2026 the network-guided player (blend 0.25,
                    # depth 4) scored 13.5/40 against the same search without the network (Elo -117, 95% CI -251..-11).
                    # Set CHEQQERS_NN_BLEND=0.35 to get the old behaviour back.
                    nn_blend = float(os.getenv("CHEQQERS_NN_BLEND", 0.0))
                    if nn_blend > 0.0 and os.path.exists(official_cheq_path):
                        cheq_model = cheqqers_value_net(env.board_size)
                    player_expecti = ExpectiCheqqersPlayer(
                        model=cheq_model,
                        max_depth=max_depth,
                        time_limit_ms=time_limit_ms,
                        nn_blend=nn_blend,
                        **cheqqers_search_options(),
                    )
                    selected_move = player_expecti.get_move(env, available_moves=legalMoves)
                    if selected_move in legalMoves:
                        elapsed_ms = (time.perf_counter() - t_req_start) * 1000.0
                        print(f"🤖 [Cheqqers Official] size={env.board_size} q={env.quantumness} player={player} | engine=python | depth={max_depth} | move={selected_move} | latency={elapsed_ms:.1f}ms", flush=True)
                        self.send_success_response({"move": selected_move})
                        return
                    reason = f"move {selected_move} is not one of the legal moves"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    reason = f"{type(e).__name__}: {e}"
                print(f"⚠️ [Cheqqers Python search] failed ({reason}): answering a random legal move.", flush=True)
                self.send_success_response({"move": random.choice(legalMoves)})
                return

            # 3. Game is TiqTaqToe: the exact tables (quantumness 0-2, decoherence off), optimal play, no search.
            # No legal move from them (tables missing, unsupported settings, an error): a uniformly random legal move.
            if exact_ttt_player is None:
                reason = (f"TTT_EXACT={os.getenv('TTT_EXACT')}" if os.getenv("TTT_EXACT", "true").lower() != "true"
                          else "the exact player could not be loaded at start-up")
            else:
                try:
                    exact_move = exact_ttt_player.get_move(data)
                    reason = exact_ttt_player.last_reason
                except Exception as e:
                    exact_move = None
                    reason = f"ExactTiqTaqToePlayer failed: {e!r}"
                if exact_move is not None and exact_move in legalMoves:
                    elapsed_ms = (time.perf_counter() - t_req_start) * 1000.0
                    print(f"🤖 [TiqTaqToe Exact] player={player} | move={exact_move} | {exact_ttt_player.last_reason} | latency={elapsed_ms:.1f}ms", flush=True)
                    self.send_success_response({"move": exact_move})
                    return
                if exact_move is not None:
                    reason = f"table move {exact_move} is not one of the legal moves"
            self.send_success_response({"move": random_tiqtaqtoe_move(legalMoves, reason)})

        else:
            self.send_error(404, "Unknown endpoint: POST /move is the only one")

    def send_success_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def end_headers(self):
        # Add CORS support and disable caching for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    do_post = do_POST
    do_get = do_GET

def require_cheqqers_engine() -> None:
    """Start-up check: the native Cheqqers engine is built (if missing and a compiler exists) and loaded now, before
    the warm-up and before any port is bound. Without it the server does not start: there is no Python fallback."""
    if cheqqers_engine() == "python":
        print("⚠️ Cheqqers engine: Python (CHEQQERS_ENGINE=python, for experiments; not the submitted agent).")
        return
    if float(os.getenv("CHEQQERS_NN_BLEND", 0.0)) > 0.0:
        raise SystemExit("❌ CHEQQERS_NN_BLEND > 0 needs the value network, which only the Python search has: "
                         "unset it, or set CHEQQERS_ENGINE=python for that experiment. Not starting.")
    from search import native_cheqqers
    try:
        native_cheqqers.load()
    except Exception as err:
        raise SystemExit(f"❌ The native Cheqqers engine cannot be loaded, so the server does not start: {err}. "
                         f"   Build it with `bash scripts/build_native.sh` (needs g++), or ship the prebuilt "
                         f"{native_cheqqers.LIB_PATH}.") from err
    print(f"✅ Cheqqers engine: native ({native_cheqqers.LIB_PATH}).")


def run(port=None):
    if port is None:
        port = int(os.getenv('PORT', 8081))
    server_address = ('', port)
    http.server.HTTPServer.allow_reuse_address = True
    if exact_ttt_player is not None:
        if exact_ttt_player.preload():
            print("✅ Exact TiqTaqToe tables loaded for the tournament configuration (quantumness 2, decoherence off).")
        else:
            print("⚠️ Exact TiqTaqToe tables for quantumness 2 are missing (see exact/docs/results.md to solve them). "
                  "TiqTaqToe plays random legal moves until then.")
    # Read the Cheqqers evaluation weights now: a bad CHEQQERS_EVAL_WEIGHTS stops the server here, loudly, and
    # never reaches a move request.
    cheqqers_weights = eval_weights_from_environment()
    print("✅ Cheqqers evaluation: "
          + ("default weights." if cheqqers_weights is None
             else f"weights from {os.getenv('CHEQQERS_EVAL_WEIGHTS')}."))
    # The same for the opening book (CHEQQERS_OPENING_BOOK): read now, fail now.
    cheqqers_book = book_from_environment()
    require_cheqqers_engine()
    setup_cheqqers_mcts()  # MCTS on the GPU with the shipped network (or CHEQQERS_MCTS_MODEL); FW on failure or if empty
    arena_ms, arena_depth = cheqqers_budget({"timeout": 3000})
    print(f"✅ Cheqqers search: depth cap {arena_depth}, time from each request's settings.timeout "
          f"({arena_ms:.0f} ms at the arena's 3000 ms limit), quiescence "
          + ("on" if os.getenv("CHEQQERS_QUIESCENCE", "0") == "1" else "off")
          + f", extension budget {os.getenv('CHEQQERS_CAPTURE_EXTENSION', 4)}, three-square moves "
          + (f"in the first {os.environ['CHEQQERS_QUANTUM_PLIES']} plies of the search."
             if os.getenv("CHEQQERS_QUANTUM_PLIES", "").strip() else "at every ply."))
    print("✅ Cheqqers opening book: "
          + ("none, every move is searched." if cheqqers_book is None
             else f"{len(cheqqers_book)} positions from {os.getenv('CHEQQERS_OPENING_BOOK')}."))
    # Ready before reachable: one move of each game through this handler on a private port, then the public port
    # (src/arena/warmup.py). WARMUP=0 skips it, for tests that start many servers.
    if os.getenv("WARMUP", "1") != "0":
        warm_up(SNNBackendHandler)
    httpd = http.server.HTTPServer(server_address, SNNBackendHandler)
    print(f"✅ READY: listening on port {port}; every move path has answered once.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the move server.")
        httpd.server_close()

def main():
    run()

if __name__ == '__main__':
    main()
