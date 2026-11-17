"""The Cheqqers rules engine and search in C++ (native/cheqqers/), loaded through ctypes.

The native engine is a line-by-line port of `env/cheqqers_official.py` and of `ExpectiCheqqersPlayer` in
`search/expecti_cheqqers.py`: same legal moves in the same order, the same outcome probabilities to the last bit,
the same evaluation, and at a fixed depth the same move after the same number of nodes (tests/native/). It is
faster, not different.

    from search import native_cheqqers as nc
    result = nc.best_move(payload_or_env, time_limit_ms=2200, max_depth=12, quantum_plies=2)
    result.move, result.nodes, result.last_depth, result.last_score, result.root_scores

    env = nc.NativeEnv.from_env(python_env)          # or NativeEnv.from_payload(payload), NativeEnv.new(6, 2, 2)
    env.legal_moves(); env.move_outcomes([13, 10]); env.terminal; env.winner; env.eval_features(); env.to_python()

The library is `native/cheqqers/libcheqqers.so`. If it is missing (or older than its sources) and a C++ compiler is
on the PATH (and CHEQQERS_NATIVE_BUILD is not 0), `load()` builds it with `scripts/build_native.sh`; otherwise it
raises `NativeUnavailable`, and server.py refuses to start (this is the server's Cheqqers engine; the Python search
is not a fallback). CHEQQERS_NATIVE_LIB points at another library file. A payload is converted once per request:
the Python simulator reads it (`from_tournament_json`, so every rule of the reconstruction is the Python's), and its
state is copied across. Options the port does not have (a value network) raise `NativeUnsupported`; so does a
position beyond the port's fixed capacities before the search starts. If the search itself raises (a bug), it stops
and `best_move` returns the last finished iteration's move with `error` set.

Identity holds for the interpreter this repo runs (CPython 3.13): the Python's float `sum()` is compensated
(Neumaier) from 3.12 on and the port reproduces that, so under Python 3.11 the Python itself would differ.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.cheqqers_official import (  # noqa: E402
    EVAL_FEATURES,
    EvalWeights,
    GameState,
    GameType,
    JointSystem,
    OfficialCheqqersEnv,
    PieceColor,
    QuantumSquare,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native" / "cheqqers"
LIB_PATH = Path(os.getenv("CHEQQERS_NATIVE_LIB", str(NATIVE_DIR / "libcheqqers.so")))
# CHEQQERS_NATIVE_BUILD=0: never compile, only load what is there (a machine that must run the prebuilt library)
BUILD_ALLOWED = os.getenv("CHEQQERS_NATIVE_BUILD", "1").strip() != "0"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_native.sh"
SOURCES = ("engine.h", "engine.cpp", "search.h", "search.cpp", "api.cpp", "value_net.h", "value_net.cpp", "mcts.h",
           "mcts.cpp")
ABI_VERSION = 2
MAGIC = 0x43510001
NAN_ID = -(2**31)
NFEAT = len(EVAL_FEATURES)
MAX_OUTCOMES = 4096
MAX_MOVES = 4096


class NativeUnavailable(RuntimeError):
    """The native library cannot be built or loaded here."""


class NativeUnsupported(RuntimeError):
    """The request uses something the native engine does not do; use the Python engine for it."""


class _Opts(ctypes.Structure):
    _fields_ = [
        ("max_depth", ctypes.c_int32),
        ("time_limit_ms", ctypes.c_double),
        ("max_tt_size", ctypes.c_int32),
        ("max_capture_extension", ctypes.c_int32),
        ("star1", ctypes.c_int32),
        ("quiescence", ctypes.c_int32),
        ("quantum_plies", ctypes.c_int32),
        ("use_cache", ctypes.c_int32),
        ("has_weights", ctypes.c_int32),
        ("weights", ctypes.c_double * (NFEAT + 2)),
    ]


class _Result(ctypes.Structure):
    _fields_ = [
        ("chosen", ctypes.c_int32),
        ("nodes", ctypes.c_int64),
        ("last_depth", ctypes.c_int32),
        ("has_score", ctypes.c_int32),
        ("last_score", ctypes.c_double),
        ("unexpanded", ctypes.c_int32),
        ("elapsed_ms", ctypes.c_double),
        ("n_root", ctypes.c_int32),
        ("failed", ctypes.c_int32),
    ]


class MctsOpts(ctypes.Structure):
    """mcts.h MctsOpts (api.cpp CqMctsOptsC)."""
    _fields_ = [
        ("c_puct", ctypes.c_double),
        ("fpu_reduction", ctypes.c_double),
        ("dirichlet_alpha", ctypes.c_double),
        ("dirichlet_frac", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("max_nodes", ctypes.c_int64),
        ("max_moves", ctypes.c_int32),
        ("gumbel_m", ctypes.c_int32),  # > 0: Gumbel root + Sequential Halving (mcts.h)
        ("gumbel_n", ctypes.c_int32),
        ("c_visit", ctypes.c_double),
        ("c_scale", ctypes.c_double),
    ]


class MctsStats(ctypes.Structure):
    """api.cpp CqMctsStatsC."""
    _fields_ = [
        ("sims", ctypes.c_int64),
        ("evals", ctypes.c_int64),
        ("terminal_sims", ctypes.c_int64),
        ("collisions", ctypes.c_int64),
        ("nodes", ctypes.c_int64),
        ("overflow", ctypes.c_int64),
        ("max_depth", ctypes.c_int32),
        ("root_turn", ctypes.c_int32),
        ("root_value", ctypes.c_double),
        ("root_vnet", ctypes.c_double),
        ("done", ctypes.c_int32),
    ]


_lib: Optional[ctypes.CDLL] = None
_lock = threading.Lock()
_ERR = 512


def _compiler() -> Optional[str]:
    return shutil.which(os.getenv("CXX", "g++"))


def _stale() -> bool:
    if not LIB_PATH.exists():
        return True
    built = LIB_PATH.stat().st_mtime
    return any((NATIVE_DIR / name).exists() and (NATIVE_DIR / name).stat().st_mtime > built for name in SOURCES)


def build() -> Path:
    """Compile the library with scripts/build_native.sh. NativeUnavailable without a compiler or on failure."""
    if _compiler() is None or shutil.which("bash") is None:
        raise NativeUnavailable("no C++ compiler (g++ or $CXX) to build native/cheqqers")
    env = dict(os.environ, CQ_OUT=str(LIB_PATH))
    done = subprocess.run(["bash", str(BUILD_SCRIPT)], capture_output=True, text=True, timeout=600, env=env)
    if done.returncode != 0:
        raise NativeUnavailable(f"building native/cheqqers failed:\n{done.stderr[-2000:]}")
    return LIB_PATH


def load() -> ctypes.CDLL:
    """The loaded library, built first if it is missing or older than its sources and a compiler is present."""
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        if BUILD_ALLOWED and _stale() and _compiler() is not None and NATIVE_DIR.exists():
            build()
        if not LIB_PATH.exists():
            raise NativeUnavailable(f"{LIB_PATH} is missing and there is no compiler to build it")
        try:
            lib = ctypes.CDLL(str(LIB_PATH))
        except OSError as err:
            raise NativeUnavailable(f"cannot load {LIB_PATH}: {err}") from err
        if lib.cq_abi_version() != ABI_VERSION:
            raise NativeUnavailable(f"{LIB_PATH} has ABI {lib.cq_abi_version()}, this wrapper {ABI_VERSION}")
        p, i32, i64, dbl, cp = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_double, ctypes.c_char_p
        ip32, dp = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
        fp = ctypes.POINTER(ctypes.c_float)
        sigs = {
            "cq_env_new": (p, [i32, i32, i32, cp, i32]),
            "cq_env_import": (p, [ip32, i64, dp, i64, cp, i32]),
            "cq_env_free": (None, [p]),
            "cq_env_clone": (p, [p]),
            "cq_env_set_cache": (None, [p, i32]),
            "cq_env_export": (i64, [p, ip32, i64, dp, i64, ctypes.POINTER(ctypes.c_int64)]),
            "cq_state": (i32, [p]),
            "cq_turn": (i32, [p]),
            "cq_multi_jump": (i32, [p]),
            "cq_legal_moves": (i32, [p, ip32, i32, cp, i32]),
            "cq_move_outcomes": (i32, [p, ip32, i32, dp, ctypes.POINTER(p), i32, cp, i32]),
            "cq_apply_move": (i32, [p, ip32, i32, ip32, i32, i32, ctypes.c_uint64, cp, i32]),
            "cq_eval_features": (i32, [p, i32, dp, cp, i32]),
            "cq_evaluate": (i32, [p, i32, dp, dp, cp, i32]),
            "cq_best_move": (i32, [p, ip32, i32, ctypes.POINTER(_Opts), ctypes.POINTER(_Result), ip32, dp, i32, cp,
                                   i32]),
            "cq_selfplay": (i32, [p, i32, i32, i32, ctypes.POINTER(_Opts), i32, ctypes.c_uint64, cp, i32, cp, i32]),
        }
        for name, (res, args) in sigs.items():
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = args
        # the value network's entry points (value_net.h); a library built before them still loads, without a net
        optional = {
            "cq_net_new": (p, [ip32, i32, dp, i64, cp, i32]),
            "cq_net_free": (None, [p]),
            "cq_net_inputs": (i32, [p]),
            "cq_net_forward": (i32, [p, dp, i32, dp, cp, i32]),
            "cq_net_encode": (i32, [p, p, dp, dp, cp, i32]),
            "cq_net_set_residual": (None, [p, dbl]),
            "cq_selfplay_leaves": (i32, [p, i32, i32, i32, ctypes.POINTER(_Opts), p, i32, ctypes.c_uint64, cp, i32,
                                         dbl, cp, i32]),
            "cq_best_move_net": (i32, [p, ip32, i32, ctypes.POINTER(_Opts), p, ctypes.POINTER(_Result), ip32, dp,
                                       i32, cp, i32]),
            "cq_selfplay_net": (i32, [p, i32, i32, i32, ctypes.POINTER(_Opts), p, i32, ctypes.c_uint64, cp, i32, cp,
                                      i32]),
            # MCTS for a network run by the caller (mcts.h, track 33; search/gpu_mcts_cheqqers.py drives it)
            "cq_encode_position": (i32, [p, dp, cp, i32]),
            "cq_mcts_abi": (i32, []),
            "cq_mcts_new": (p, [p, ip32, i32, ctypes.POINTER(MctsOpts), cp, i32]),
            "cq_mcts_free": (None, [p]),
            "cq_mcts_in_dim": (i32, [p]),
            "cq_mcts_collect": (i32, [p, i32, i32, fp, ip32, ip32, ip32, cp, i32]),
            "cq_mcts_apply": (i32, [p, fp, fp, i32, i32, cp, i32]),
            "cq_mcts_collect_many": (i32, [ctypes.POINTER(p), i32, i32, i32, fp, ip32, ip32, ip32, ip32, i32, cp,
                                           i32]),
            "cq_mcts_apply_many": (i32, [ctypes.POINTER(p), i32, ip32, fp, fp, i32, i32, cp, i32]),
            "cq_mcts_root": (i32, [p, ip32, ip32, dp, dp, i32, ctypes.POINTER(MctsStats)]),
            "cq_mcts_done": (i32, [p]),
            "cq_mcts_gumbel": (i32, [p, dp, dp, ip32, i32]),
            # batched self-play for a network run by the caller (training/selfplay_pg_cheqqers.py)
            "cq_sp_new": (p, [i32, ctypes.c_uint64, i32, i32, i32, i32, i32, cp, i32]),
            # Gumbel self-play batches (training/gumbel_selfplay.py)
            "cq_gz_new": (p, [i32, ctypes.c_uint64, i32, i32, i32, ctypes.POINTER(MctsOpts), i32, cp, i32]),
            "cq_gz_free": (None, [p]),
            "cq_gz_set_frozen": (None, [p, ip32, i32]),
            "cq_gz_which": (None, [p, ip32]),
            "cq_gz_collect": (i32, [p, i32, i32, fp, ip32, ip32, ip32, ip32, i32, cp, i32]),
            "cq_gz_apply": (i32, [p, fp, fp, i32, i32, cp, i32]),
            "cq_gz_take": (i32, [p, fp, ip32, fp, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ip32,
                                 ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]),
            "cq_gz_status": (None, [p, ip32, ip32, ctypes.POINTER(ctypes.c_int64)]),
            "cq_gz_set_refill": (None, [p, ctypes.c_int64]),
            "cq_gz_set_fw": (None, [p, ip32, i32, dbl, i32, dp]),
            "cq_sp_free": (None, [p]),
            "cq_sp_encode": (i32, [p, fp, ip32, ip32, ip32, ip32, ip32, i32, cp, i32]),
            "cq_sp_apply": (i32, [p, ip32, i32, i32, cp, i32]),
            "cq_sp_results": (i32, [p, ip32, ip32]),
        }
        for name, (res, args) in optional.items():
            if hasattr(lib, name):
                fn = getattr(lib, name)
                fn.restype = res
                fn.argtypes = args
        _lib = lib
        return lib


def available() -> bool:
    try:
        load()
        return True
    except NativeUnavailable:
        return False


def _err() -> ctypes.Array:
    return ctypes.create_string_buffer(_ERR)


def _raise(buf: ctypes.Array, what: str) -> None:
    raise NativeUnsupported(f"{what}: {buf.value.decode(errors='replace')}")


def _ints(values: Sequence[int]) -> Any:
    arr = array("i", values)
    return (ctypes.c_int32 * len(arr)).from_buffer(arr) if len(arr) else (ctypes.c_int32 * 1)()


def _doubles(values: Sequence[float]) -> Any:
    arr = array("d", values)
    return (ctypes.c_double * len(arr)).from_buffer(arr) if len(arr) else (ctypes.c_double * 1)()


# ── The flat state (format in native/cheqqers/api.cpp) ─────────────────────────────────────────────────────────

_HASH_PART = re.compile(r"^(\d+)([WB])([KM])(.+)$")


def _pid_out(x: Any) -> int:
    return NAN_ID if isinstance(x, float) and math.isnan(x) else int(x)


def _pid_in(x: int) -> Any:
    return math.nan if x == NAN_ID else int(x)


def history_codes(board_hash: str, n_squares: int) -> list[int]:
    """`get_board_state_hash()` as the native key: turn, then per square 0 or 1 + presence*4 + colour*2 + crown,
    presence in thousandths as the string has it ("1" is 1000)."""
    parts = board_hash.split("|")
    codes = [int(parts[0][2:])] + [0] * n_squares
    for part in parts[1:]:
        m = _HASH_PART.match(part)
        if m is None:
            raise NativeUnsupported(f"unexpected board hash part {part!r}")
        i, colour, crown, occ = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        presence = 1000 if occ == "1" else int(round(float(occ) * 1000))
        codes[1 + i] = 1 + presence * 4 + (0 if colour == "W" else 2) + (1 if crown == "K" else 0)
    return codes


def history_string(codes: Sequence[int]) -> str:
    parts = [f"T:{codes[0]}"]
    for i, code in enumerate(codes[1:]):
        if code == 0:
            continue
        v = code - 1
        presence, colour_black, crown = v // 4, (v >> 1) & 1, v & 1
        occ = "1" if presence == 1000 else f"0.{presence:03d}"
        parts.append(f"{i}{'B' if colour_black else 'W'}{'K' if crown else 'M'}{occ}")
    return "|".join(parts)


def export_env(env: OfficialCheqqersEnv) -> tuple[list[int], list[float]]:
    """The Python position as the flat (ints, doubles) the native engine reads; also the identity tests' fingerprint
    of a position: two positions are the same exactly when their exports are (doubles compared bit for bit)."""
    n = env.total_squares
    ints: list[int] = [MAGIC, env.board_size, env.start_rows, int(env.game_type), int(env.turn), int(env.next_piece_id),
                       int(env.moves_since_take), -1 if env.multi_jump_square is None else int(env.multi_jump_square),
                       int(bool(env.just_crowned)), int(env.consecutive_queen_moves),
                       int(bool(env.disable_depth_calculation)), int(bool(env.force_all_moves)), int(env.state),
                       int(bool(env.cache_enabled)), n]
    dbl: list[float] = []
    for i in range(n):
        p = env.piece_properties.get(i)
        if p is None:
            ints.append(0)
            continue
        ids = [_pid_out(x) for x in p["pieceIds"]]
        ints += [1, int(p["color"]), int(bool(p["crowned"])), len(ids)] + ids
    jumped = [_pid_out(x) for x in env.jumped_piece_ids_this_turn]
    ints += [len(jumped)] + jumped
    square_index = {id(sq): i for i, sq in enumerate(env.squares)}
    order: list[Any] = []
    system_index: dict[int, int] = {}
    for sq in env.squares:
        system = sq.entangled_system
        if system is not None and id(system) not in system_index:
            system_index[id(system)] = len(order)
            order.append(system)
    ints.append(len(order))
    for system in order:
        ints.append(len(system.entities))
        ints += [square_index[id(e)] for e in system.entities]
        ints.append(len(system.joint_amplitudes))
        ints += [int(k) for k in system.joint_amplitudes]
    for sq in env.squares:
        system = sq.entangled_system
        ints.append(-1 if system is None else system_index[id(system)])
        ints.append(len(sq.amplitudes))
        ints += [int(k) for k in sq.amplitudes]
    ints.append(len(env.state_history))
    for board_hash, count in env.state_history.items():
        codes = history_codes(board_hash, n)
        ints += [int(count), len(codes)] + codes
    override = env._override_legal_moves
    ints += [0 if override is None else 1, 0 if override is None else len(override)]
    for m in override or []:
        ints += [len(m)] + [int(x) for x in m]
    for system in order:
        for a in system.joint_amplitudes.values():
            dbl += [a.real, a.imag]
    for sq in env.squares:
        for a in sq.amplitudes.values():
            dbl += [a.real, a.imag]
    return ints, dbl


def import_env(ints: Sequence[int], dbl: Sequence[float]) -> OfficialCheqqersEnv:
    """The inverse of export_env: a Python position from a flat state (e.g. one written by the native self-play)."""
    it = iter(ints)
    nxt = it.__next__
    if nxt() != MAGIC:
        raise ValueError("not a flat Cheqqers state")
    board_size, start_rows, game_type = nxt(), nxt(), nxt()
    env = OfficialCheqqersEnv(board_size=board_size, start_rows=0, quantumness=game_type)
    env.start_rows = start_rows
    env.turn = PieceColor(nxt())
    env.next_piece_id = nxt()
    env.moves_since_take = nxt()
    mj = nxt()
    env.multi_jump_square = None if mj == -1 else mj
    env.just_crowned = bool(nxt())
    env.consecutive_queen_moves = nxt()
    env.disable_depth_calculation = bool(nxt())
    env.force_all_moves = bool(nxt())
    env.state = GameState(nxt())
    env.cache_enabled = bool(nxt())
    n = nxt()
    for i in range(n):
        if nxt():
            colour, crowned, k = nxt(), nxt(), nxt()
            env.piece_properties[i] = {"color": PieceColor(colour), "crowned": bool(crowned),
                                       "pieceIds": [_pid_in(nxt()) for _ in range(k)]}
        else:
            env.piece_properties[i] = None
    env.jumped_piece_ids_this_turn = [_pid_in(nxt()) for _ in range(nxt())]
    env.squares = [QuantumSquare(i) for i in range(n)]
    systems = []
    for _ in range(nxt()):
        entities = [env.squares[nxt()] for _ in range(nxt())]
        keys = [nxt() for _ in range(nxt())]
        systems.append((JointSystem(entities, initialize_amplitudes=False), keys))
    square_keys = []
    for sq in env.squares:
        s = nxt()
        sq.entangled_system = None if s == -1 else systems[s][0]
        square_keys.append([nxt() for _ in range(nxt())])
    history = {}
    for _ in range(nxt()):
        count = nxt()
        history[history_string([nxt() for _ in range(nxt())])] = count
    env.state_history = history
    has_override, n_override = nxt(), nxt()
    override = [[nxt() for _ in range(nxt())] for _ in range(n_override)]
    env._override_legal_moves = override if has_override else None
    d = iter(dbl)
    for system, keys in systems:
        system.joint_amplitudes = {k: complex(next(d), next(d)) for k in keys}
    for sq, keys in zip(env.squares, square_keys):
        sq.amplitudes = {k: complex(next(d), next(d)) for k in keys}
    env.game_type = GameType(game_type)
    env.quantumness = int(env.game_type)
    env._invalidate_caches()
    return env


def _perspective(env_turn: int, perspective: Any) -> int:
    """`OfficialCheqqersEnv._resolve_perspective_color` as 0 (white) / 1 (black)."""
    if perspective is None:
        return env_turn
    if isinstance(perspective, PieceColor):
        return int(perspective)
    if isinstance(perspective, str):
        return 0 if perspective.strip().lower() in ("white", "w", "1", "x") else 1
    if isinstance(perspective, int):
        if perspective == 1 or perspective == 0:
            return 0
        if perspective == -1:
            return 1
    return env_turn


def _weights(w: Optional[EvalWeights]) -> Optional[Any]:
    if w is None:
        return None
    return _doubles(list(w.as_vector()) + [w.scale, w.logit_per_man])


def _move_ints(move: Any) -> list[int]:
    if isinstance(move, dict):
        move = OfficialCheqqersEnv.move_to_array(move)
    values = [int(x) for x in move]
    if len(values) not in (2, 3):
        raise NativeUnsupported(f"move {move!r} is not a tournament array")
    return values


class NativeEnv:
    """A position in the native engine. Freed with the object."""

    __slots__ = ("_h", "_lib")

    def __init__(self, handle: int, lib: ctypes.CDLL):
        if not handle:
            raise NativeUnsupported("no native position")
        self._h = ctypes.c_void_p(handle)
        self._lib = lib

    def __del__(self):
        h = getattr(self, "_h", None)
        if h is not None and h.value:
            self._lib.cq_env_free(h)
            self._h = None

    @classmethod
    def new(cls, board_size: int = 6, start_rows: Optional[int] = None, quantumness: int = 2) -> "NativeEnv":
        lib, err = load(), _err()
        h = lib.cq_env_new(board_size, -1 if start_rows is None else start_rows, int(quantumness), err, _ERR)
        if not h:
            _raise(err, "new position")
        return cls(h, lib)

    @classmethod
    def from_flat(cls, ints: Sequence[int], dbl: Sequence[float]) -> "NativeEnv":
        lib, err = load(), _err()
        h = lib.cq_env_import(_ints(ints), len(ints), _doubles(dbl), len(dbl), err, _ERR)
        if not h:
            _raise(err, "import position")
        return cls(h, lib)

    @classmethod
    def from_env(cls, env: OfficialCheqqersEnv) -> "NativeEnv":
        return cls.from_flat(*export_env(env))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NativeEnv":
        return cls.from_env(OfficialCheqqersEnv.from_tournament_json(payload))

    def export(self) -> tuple[list[int], list[float]]:
        nd = ctypes.c_int64()
        n = self._lib.cq_env_export(self._h, None, 0, None, 0, ctypes.byref(nd))
        ints, dbl = (ctypes.c_int32 * n)(), (ctypes.c_double * max(1, nd.value))()
        self._lib.cq_env_export(self._h, ints, n, dbl, nd.value, ctypes.byref(nd))
        return list(ints), list(dbl[: nd.value])

    def to_python(self) -> OfficialCheqqersEnv:
        return import_env(*self.export())

    def clone(self) -> "NativeEnv":
        return NativeEnv(self._lib.cq_env_clone(self._h), self._lib)

    def set_cache(self, on: bool) -> None:
        self._lib.cq_env_set_cache(self._h, int(bool(on)))

    @property
    def state(self) -> GameState:
        return GameState(self._lib.cq_state(self._h))

    @property
    def terminal(self) -> bool:
        return self.state != GameState.IN_PROGRESS

    @property
    def winner(self) -> Optional[int]:
        """As OfficialCheqqersEnv.winner: 1 white, -1 black, 0 draw, None in progress."""
        return {GameState.WHITE_WON: 1, GameState.BLACK_WON: -1, GameState.DRAW: 0}.get(self.state)

    @property
    def turn(self) -> PieceColor:
        return PieceColor(self._lib.cq_turn(self._h))

    @property
    def multi_jump_square(self) -> Optional[int]:
        mj = self._lib.cq_multi_jump(self._h)
        return None if mj < 0 else mj

    def legal_moves(self) -> list[list[int]]:
        buf, err = (ctypes.c_int32 * (4 * MAX_MOVES))(), _err()
        n = self._lib.cq_legal_moves(self._h, buf, MAX_MOVES, err, _ERR)
        if n < 0:
            _raise(err, "legal moves")
        return [list(buf[4 * i + 1: 4 * i + 1 + buf[4 * i]]) for i in range(n)]

    def move_outcomes(self, move: Any) -> list[tuple[float, "NativeEnv"]]:
        mv = _move_ints(move)
        probs, kids, err = (ctypes.c_double * MAX_OUTCOMES)(), (ctypes.c_void_p * MAX_OUTCOMES)(), _err()
        n = self._lib.cq_move_outcomes(self._h, _ints(mv), len(mv), probs, kids, MAX_OUTCOMES, err, _ERR)
        if n < 0:
            _raise(err, f"outcomes of {mv}")
        return [(probs[i], NativeEnv(kids[i], self._lib)) for i in range(n)]

    def apply_move(self, move: Any, forced_measurements: Optional[Sequence[int]] = None, seed: int = 0) -> int:
        mv = _move_ints(move)
        forced = list(forced_measurements or [])
        err = _err()
        n = self._lib.cq_apply_move(self._h, _ints(mv), len(mv), _ints(forced), len(forced),
                                    int(forced_measurements is not None), seed, err, _ERR)
        if n < 0:
            _raise(err, f"apply {mv}")
        return n

    def encode(self) -> list[float]:
        """The network input of this position (encode_position flattened, side to move's view; 144 on 6x6)."""
        if not hasattr(self._lib, "cq_encode_position"):
            raise NativeUnavailable(f"{LIB_PATH} has no cq_encode_position (rebuild it)")
        err = _err()
        buf = (ctypes.c_double * 1024)()
        n = self._lib.cq_encode_position(self._h, buf, err, _ERR)
        if n < 0:
            _raise(err, "encode")
        return list(buf[:n])

    def eval_features(self, perspective_color: Any = None) -> tuple[float, ...]:
        out, err = (ctypes.c_double * NFEAT)(), _err()
        if self._lib.cq_eval_features(self._h, _perspective(int(self.turn), perspective_color), out, err, _ERR) < 0:
            _raise(err, "eval features")
        return tuple(out)

    def evaluate_state(self, perspective_color: Any = None, weights: Optional[EvalWeights] = None) -> float:
        out, err = ctypes.c_double(), _err()
        if self._lib.cq_evaluate(self._h, _perspective(int(self.turn), perspective_color), _weights(weights),
                                 ctypes.byref(out), err, _ERR) < 0:
            _raise(err, "evaluate")
        return out.value


class ValueNet:
    """A value network (src/training/train_cheqqers_value_mlp.py's value_mlp.json) loaded into the native library.

    JSON: {"inputs": 144, "activation": "relu", "layers": [{"weight": [[out x in]], "bias": [out]}, ...]}; the
    output is the logit that the side to move wins. At the search's leaves it replaces the linear evaluation:
    900 tanh(logit / 2) for the side to move (the fitted evaluation's squash scale), negated for the other side;
    terminal positions keep +-1000 / 0."""

    def __init__(self, path: str | Path):
        import json

        self.path = Path(path)
        spec = json.loads(self.path.read_text())
        if spec.get("activation", "relu") != "relu":
            raise NativeUnsupported(f"value net {path}: activation {spec.get('activation')!r} (only relu)")
        layers = spec["layers"]
        dims = [int(spec["inputs"])]
        params: list[float] = []
        for layer in layers:
            w, b = layer["weight"], layer["bias"]
            if len(w) != len(b) or any(len(row) != dims[-1] for row in w):
                raise NativeUnsupported(f"value net {path}: layer shapes do not chain")
            for row in w:
                params.extend(float(v) for v in row)
            params.extend(float(v) for v in b)
            dims.append(len(b))
        self.dims = dims
        self.layers = [(list(map(list, layer["weight"])), list(layer["bias"])) for layer in layers]
        self._lib = load()
        if not hasattr(self._lib, "cq_net_new"):
            raise NativeUnavailable(f"{LIB_PATH} has no value network entry points (rebuild it)")
        err = _err()
        self._h = self._lib.cq_net_new(_ints(dims), len(dims) - 1, _doubles(params), len(params), err, _ERR)
        if not self._h:
            _raise(err, "value net")
        # {"residual": {"lambda": x}}: the leaf logit is the search weights' linear logit + x * the network's output
        # (the network was trained as a correction of the fitted linear evaluation; needs weights with logit_per_man)
        self.residual_lambda: Optional[float] = None
        if spec.get("residual") is not None:
            if not hasattr(self._lib, "cq_net_set_residual"):
                raise NativeUnavailable(f"{LIB_PATH} has no residual value network (rebuild it)")
            self.residual_lambda = float(spec["residual"]["lambda"])
            self._lib.cq_net_set_residual(self._h, self.residual_lambda)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h and self._lib is not None:
            self._lib.cq_net_free(h)

    @property
    def inputs(self) -> int:
        return self.dims[0]

    def forward(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """The C++ forward pass on given input rows -> logits."""
        flat = [float(v) for r in rows for v in r]
        if len(flat) != len(rows) * self.inputs:
            raise ValueError("input rows of the wrong length")
        out, err = (ctypes.c_double * max(1, len(rows)))(), _err()
        if self._lib.cq_net_forward(self._h, _doubles(flat), len(rows), out, err, _ERR) < 0:
            _raise(err, "value net forward")
        return list(out[: len(rows)])

    def encode(self, env: "NativeEnv") -> tuple[list[float], float]:
        """(the C++ input vector for this position, its logit), side to move's view."""
        x, logit, err = (ctypes.c_double * self.inputs)(), ctypes.c_double(), _err()
        if self._lib.cq_net_encode(self._h, env._h, x, ctypes.byref(logit), err, _ERR) < 0:
            _raise(err, "value net encode")
        return list(x), logit.value


_nets: dict[tuple[str, float], ValueNet] = {}


def resolve_value_net(spec: Any = None) -> Optional[ValueNet]:
    """The network to use: a ValueNet as given; a path (loaded once per path and modification time); None -> the
    path in CHEQQERS_VALUE_NET if set (the server's switch), else no network; "" or False -> no network."""
    if isinstance(spec, ValueNet):
        return spec
    if spec is None:
        spec = os.getenv("CHEQQERS_VALUE_NET", "").strip()
    if not spec:
        return None
    path = Path(spec).expanduser().resolve()
    key = (str(path), path.stat().st_mtime)
    with _lock:
        net = _nets.get(key)
    if net is None:
        net = ValueNet(path)
        with _lock:
            _nets[key] = net
    return net


def _opts(time_limit_ms: float, max_depth: int, eval_weights: Optional[EvalWeights], quiescence: bool,
          quantum_plies: Optional[int], max_capture_extension: int, max_tt_size: int, star1_pruning: bool,
          use_cache: bool) -> _Opts:
    opts = _Opts(max_depth=int(max_depth), time_limit_ms=float(time_limit_ms), max_tt_size=int(max_tt_size),
                 max_capture_extension=int(max_capture_extension), star1=int(bool(star1_pruning)),
                 quiescence=int(bool(quiescence)), quantum_plies=-1 if quantum_plies is None else int(quantum_plies),
                 use_cache=int(bool(use_cache)), has_weights=int(eval_weights is not None))
    if eval_weights is not None:
        for i, v in enumerate(list(eval_weights.as_vector()) + [eval_weights.scale, eval_weights.logit_per_man]):
            opts.weights[i] = float(v)
    return opts


@dataclass
class NativeSearchResult:
    move: Any  # the chosen element of the legal moves given (None if there were none)
    nodes: int = 0  # ExpectiCheqqersPlayer._nodes_visited
    last_depth: int = 0
    last_score: Optional[float] = None
    from_book: bool = False
    unexpanded: int = 0
    elapsed_ms: float = 0.0  # inside the native call
    root_scores: list[tuple[list[int], float]] = field(default_factory=list)  # last finished iteration, best first
    error: Optional[str] = None  # an iteration raised (a bug): `move` is the last finished iteration's best move


def best_move(
    payload_or_env: Any,
    time_limit_ms: float = 1500.0,
    max_depth: int = 8,
    *,
    available_moves: Optional[Sequence[Any]] = None,
    eval_weights: Optional[EvalWeights] = None,
    quiescence: bool = False,
    quantum_plies: Optional[int] = None,
    max_capture_extension: int = 4,
    max_tt_size: int = 65536,
    star1_pruning: bool = True,
    use_cache: bool = True,
    opening_book: Any = None,
    model: Any = None,
    nn_blend: float = 0.25,
    value_net: Any = None,
) -> NativeSearchResult:
    """`ExpectiCheqqersPlayer(max_depth=..., time_limit_ms=..., <options>).get_move(env, available_moves)` in C++.

    `payload_or_env`: a tournament /move payload (dict with "board"), an OfficialCheqqersEnv, or a NativeEnv. For a
    payload the legal moves default to its "legalMoves", as server.py passes them. A value network (`model` with
    `nn_blend` > 0) raises NativeUnsupported. `value_net`: a small MLP at the leaves instead of the linear evaluation
    (ValueNet or path; None = CHEQQERS_VALUE_NET if set; "" = none), see `resolve_value_net()`."""
    if model is not None and nn_blend > 0.0:
        raise NativeUnsupported("the native search has no value network")
    lib = load()
    net = resolve_value_net(value_net)
    py_env: Optional[OfficialCheqqersEnv] = None
    if isinstance(payload_or_env, NativeEnv):
        native = payload_or_env
    else:
        if isinstance(payload_or_env, dict):
            py_env = OfficialCheqqersEnv.from_tournament_json(payload_or_env)
            if available_moves is None and payload_or_env.get("legalMoves") is not None:
                available_moves = payload_or_env["legalMoves"]
        else:
            py_env = payload_or_env
        if hasattr(py_env, "eval_weights"):
            py_env.eval_weights = eval_weights  # as ExpectiCheqqersPlayer._begin does to the position it is given
        native = NativeEnv.from_env(py_env)
    raw = list(available_moves) if available_moves is not None else None
    if raw is not None:
        if len(raw) == 0:
            return NativeSearchResult(move=None)
        if len(raw) == 1:
            return NativeSearchResult(move=raw[0])
    if opening_book is not None:
        book_env = py_env if py_env is not None else native.to_python()
        legal = raw if raw is not None else book_env.get_legal_moves()
        if len(legal) > 1:
            booked = opening_book.lookup(book_env, legal)
            if booked is not None:
                return NativeSearchResult(move=booked, from_book=True)
    opts = _opts(time_limit_ms, max_depth, eval_weights, quiescence, quantum_plies, max_capture_extension,
                 max_tt_size, star1_pruning, use_cache)
    flat: list[int] = []
    if raw is not None:
        for m in raw:
            mv = _move_ints(m)
            flat += [len(mv)] + mv + [0] * (3 - len(mv))
    res, err = _Result(), _err()
    cap = MAX_MOVES
    root_moves, root_scores = (ctypes.c_int32 * (4 * cap))(), (ctypes.c_double * cap)()
    if net is None:
        rc = lib.cq_best_move(native._h, _ints(flat), -1 if raw is None else len(raw), ctypes.byref(opts),
                              ctypes.byref(res), root_moves, root_scores, cap, err, _ERR)
    else:
        rc = lib.cq_best_move_net(native._h, _ints(flat), -1 if raw is None else len(raw), ctypes.byref(opts),
                                  net._h, ctypes.byref(res), root_moves, root_scores, cap, err, _ERR)
    if rc < 0:
        _raise(err, "search")
    legal_list = raw if raw is not None else native.legal_moves()
    chosen = None if res.chosen < 0 or not legal_list else legal_list[res.chosen]
    scores = [(list(root_moves[4 * i + 1: 4 * i + 1 + root_moves[4 * i]]), root_scores[i]) for i in range(res.n_root)]
    return NativeSearchResult(move=chosen, nodes=res.nodes, last_depth=res.last_depth,
                              last_score=res.last_score if res.has_score else None, unexpanded=res.unexpanded,
                              elapsed_ms=res.elapsed_ms, root_scores=scores,
                              error=err.value.decode(errors="replace") if res.failed else None)


def selfplay(out_path: str | Path, games: int = 1, seed: int = 1, *, start: Any = None, board_size: int = 6,
             start_rows: Optional[int] = None, quantumness: int = 2, time_limit_ms: float = 200.0, max_depth: int = 4,
             max_plies: int = 300, write_state: bool = False, eval_weights: Optional[EvalWeights] = None,
             quiescence: bool = False, quantum_plies: Optional[int] = 2, max_capture_extension: int = 4,
             max_tt_size: int = 65536, value_net: Any = "", leaf_rate: float = 0.0) -> int:
    """Play `games` whole games in C++, both sides the native search with these options, measurement outcomes drawn
    with their probabilities (game k uses seed + k). Appends JSON lines to `out_path`: one per position searched
    (features, presences, pieces, legal moves, their root scores, the move played, depth, nodes; with
    `write_state` the flat state, which import_env reads) and one {"result", "state", "plies"} per game. `start`: an
    OfficialCheqqersEnv or NativeEnv to start from instead of the initial position. `value_net`: a ValueNet or path
    for both sides' leaves ("" = none; None = CHEQQERS_VALUE_NET). `leaf_rate` > 0: each searched leaf is also
    written with this probability as {"leaf", "turn", "d", "cp", "x"} (network input) after its position's line.
    Returns the positions written."""
    lib = load()
    net = resolve_value_net(value_net)
    opts = _opts(time_limit_ms, max_depth, eval_weights, quiescence, quantum_plies, max_capture_extension,
                 max_tt_size, True, True)
    native = start if isinstance(start, NativeEnv) or start is None else NativeEnv.from_env(start)
    total = 0
    for k in range(games):
        err = _err()
        if leaf_rate > 0.0:
            if not hasattr(lib, "cq_selfplay_leaves"):
                raise NativeUnavailable(f"{LIB_PATH} has no leaf sampling (rebuild it)")
            n = lib.cq_selfplay_leaves(None if native is None else native._h, board_size,
                                       -1 if start_rows is None else start_rows, quantumness, ctypes.byref(opts),
                                       None if net is None else net._h, max_plies, (seed + k) & 0xFFFFFFFFFFFFFFFF,
                                       str(out_path).encode(), int(write_state), float(leaf_rate), err, _ERR)
        elif net is None:
            n = lib.cq_selfplay(None if native is None else native._h, board_size,
                                -1 if start_rows is None else start_rows, quantumness, ctypes.byref(opts), max_plies,
                                (seed + k) & 0xFFFFFFFFFFFFFFFF, str(out_path).encode(), int(write_state), err, _ERR)
        else:
            n = lib.cq_selfplay_net(None if native is None else native._h, board_size,
                                    -1 if start_rows is None else start_rows, quantumness, ctypes.byref(opts), net._h,
                                    max_plies, (seed + k) & 0xFFFFFFFFFFFFFFFF, str(out_path).encode(),
                                    int(write_state), err, _ERR)
        if n < 0:
            _raise(err, "self-play")
        total += n
    return total
