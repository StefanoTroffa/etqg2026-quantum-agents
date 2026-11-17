"""MCTS serving with FW fallback (server.py CHEQQERS_MCTS_MODEL; track 33, docs/specs/cheqqers_gpu_mcts.md).

- empty: no MCTS player, the FW path is the one that answers; unset: the network shipped in the repo;
- set but CUDA missing (or the network failing to load): the server says so and falls back to FW;
- set with a usable device: the start-up search fits the budget and the move path answers a legal MCTS move;
- a warm-up with fewer network evaluations than the floor (a slow GPU) falls back to FW;
- an MCTS error mid-move answers FW's legal move, and FW gets only the time left (the reply stays in the limit);
- other settings than 6x6 / 2 start rows / quantumness 2 go to FW before any MCTS work;
- the server's own warm-up (arena/warmup.py) goes through the MCTS path when it is enabled.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
torch = pytest.importorskip("torch")


@pytest.fixture
def server_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CHEQQERS_ENGINE", raising=False)
    sys.path.insert(0, str(REPO))
    import server

    monkeypatch.setattr(server, "_MCTS_PLAYER", None)
    return server


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(REPO / "src"))
    from models.cheqqers_policy_value import CheqqersPolicyValue, model_config

    torch.manual_seed(0)
    model = CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64)
    path = tmp_path_factory.mktemp("m") / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model_config(model)}, path)
    return path


def test_empty_means_no_mcts(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", "")
    server_module.setup_cheqqers_mcts()
    assert server_module._MCTS_PLAYER is None


def test_unset_means_the_shipped_network(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
    """`uv run python server.py` with no configuration plays the submitted agent: the network in the repo, found from
    any working directory, and the file is a loadable policy-value network."""
    monkeypatch.delenv("CHEQQERS_MCTS_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    path = Path(server_module.mcts_model_path())
    assert path == REPO / server_module.DEFAULT_CHEQQERS_MCTS_MODEL and path.is_file()
    sys.path.insert(0, str(REPO / "src"))
    from search import native_cheqqers
    from search.gpu_mcts_cheqqers import MctsPlayer

    start = native_cheqqers.NativeEnv.new(6, 2, 2)
    result = MctsPlayer(str(path), device="cpu", batch=16, half=False).search(start, None, time_ms=300.0)
    assert result.evals > 0 and list(result.move) in [list(m) for m in start.legal_moves()]


def test_missing_cuda_falls_back_to_fw(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                      tiny_model: Path, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(tiny_model))
    monkeypatch.delenv("CHEQQERS_MCTS_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    server_module.setup_cheqqers_mcts()
    assert server_module._MCTS_PLAYER is None
    assert "falls back to FW" in capsys.readouterr().out


def test_bad_network_falls_back_to_fw(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a model")
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(bad))
    monkeypatch.setenv("CHEQQERS_MCTS_DEVICE", "cpu")
    server_module.setup_cheqqers_mcts()
    assert server_module._MCTS_PLAYER is None
    assert "falls back to FW" in capsys.readouterr().out


def test_usable_mcts_answers_legal_moves(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                         tiny_model: Path) -> None:
    from arena.warmup import start_payloads

    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(tiny_model))
    monkeypatch.setenv("CHEQQERS_MCTS_DEVICE", "cpu")
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "400")
    monkeypatch.setenv("CHEQQERS_MCTS_MIN_EVALS", "1")
    server_module.setup_cheqqers_mcts()
    assert server_module._MCTS_PLAYER is not None
    payload = dict(start_payloads())["cheqqers"]
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", 0.0)
    assert move in payload["legalMoves"]


def test_mcts_error_answers_fw_move(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                    tmp_path: Path) -> None:
    from arena.warmup import start_payloads

    class Broken:
        def search(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated MCTS bug")

    monkeypatch.setenv("CHEQQERS_FAILURE_DIR", str(tmp_path))
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "200")
    monkeypatch.setattr(server_module, "_MCTS_PLAYER", Broken())
    payload = dict(start_payloads())["cheqqers"]
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", 0.0)
    assert move in payload["legalMoves"] and any(tmp_path.iterdir())


def test_too_few_warm_up_evaluations_fall_back_to_fw(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                     tiny_model: Path, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(tiny_model))
    monkeypatch.setenv("CHEQQERS_MCTS_DEVICE", "cpu")
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "300")
    monkeypatch.setenv("CHEQQERS_MCTS_MIN_EVALS", "100000000")
    server_module.setup_cheqqers_mcts()
    assert server_module._MCTS_PLAYER is None
    assert "network evaluations" in capsys.readouterr().out


def test_a_late_mcts_error_leaves_fw_only_the_time_left(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    from arena.warmup import start_payloads

    class LateBroken:
        def search(self, *args: object, **kwargs: object) -> None:
            time.sleep(1.0)
            raise RuntimeError("simulated CUDA error late in the move")

    monkeypatch.setenv("CHEQQERS_FAILURE_DIR", str(tmp_path))
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "1200")
    monkeypatch.setattr(server_module, "_MCTS_PLAYER", LateBroken())
    payload = dict(start_payloads())["cheqqers"]
    t0 = time.perf_counter()
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", t0)
    took = time.perf_counter() - t0
    assert move in payload["legalMoves"]
    assert took < 1.6, f"{took:.2f} s: FW took the full budget again after the MCTS used 1 s"


@pytest.mark.parametrize("size, quantumness, rows", [(8, 2, None), (6, 1, None), (6, 2, 1)])
def test_other_settings_go_to_fw_before_any_mcts_work(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                      size: int, quantumness: int, rows: int | None) -> None:
    from env.cheqqers_official import OfficialCheqqersEnv

    class MustNotRun:
        def search(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("the MCTS was asked about another configuration")

    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "200")
    monkeypatch.setattr(server_module, "_MCTS_PLAYER", MustNotRun())
    payload = OfficialCheqqersEnv(board_size=size, quantumness=quantumness).to_tournament_payload(timeout_ms=3000)
    if rows is not None:
        payload["settings"]["startRows"] = rows
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", time.perf_counter())
    assert move in payload["legalMoves"]


def test_the_tournament_payload_goes_to_the_mcts(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not turn the MCTS off for the tournament's own payloads (start position and mid-game)."""
    from arena.warmup import start_payloads

    seen = []

    class Recorder:
        def search(self, env: object, legal: list, time_ms: float) -> object:
            seen.append(time_ms)

            class R:
                move = legal[0]
                evals = 1
            return R()

    monkeypatch.setattr(server_module, "_MCTS_PLAYER", Recorder())
    payload = dict(start_payloads())["cheqqers"]
    assert server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", time.perf_counter()) \
        == payload["legalMoves"][0]
    assert len(seen) == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_warm_up_goes_through_the_mcts(tiny_model: Path) -> None:
    env = dict(os.environ, PORT=str(_free_port()), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
               CHEQQERS_MCTS_MODEL=str(tiny_model), CHEQQERS_MCTS_DEVICE="cpu", CHEQQERS_MCTS_MIN_EVALS="1",
               CHEQQERS_TIME_LIMIT_MS="300", CHEQQERS_NATIVE_BUILD="0")
    env.pop("CHEQQERS_ENGINE", None)
    proc = subprocess.Popen([sys.executable, "server.py"], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines: list[str] = []
    try:
        deadline = time.time() + 240
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "READY" in line:
                break
    finally:
        proc.kill()
        proc.wait(timeout=30)
    out = "".join(lines)
    assert "READY" in out, out[-3000:]
    warm = out[: out.index("READY")]
    assert "Cheqqers engine: MCTS" in warm and "engine=mcts" in warm, warm[-3000:]


def test_server_plays_the_panels_configuration(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                               tiny_model: Path) -> None:
    """The server's MCTS uses the settings the panels measured (gpu_mcts_cheqqers.PLAY_*), not the class defaults."""
    from search.gpu_mcts_cheqqers import PLAY_BATCH, PLAY_C_PUCT, PLAY_FPU

    for var in ("CHEQQERS_MCTS_BATCH", "CHEQQERS_MCTS_CPUCT", "CHEQQERS_MCTS_FPU"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(tiny_model))
    monkeypatch.setenv("CHEQQERS_MCTS_DEVICE", "cpu")
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "300")
    monkeypatch.setenv("CHEQQERS_MCTS_MIN_EVALS", "1")
    server_module.setup_cheqqers_mcts()
    p = server_module._MCTS_PLAYER
    assert (p.batch, p.c_puct, p.fpu) == (PLAY_BATCH, PLAY_C_PUCT, PLAY_FPU)


def test_series_agent_and_server_build_the_same_player(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                       tiny_model: Path) -> None:
    """The hub's mcts= agent and the server construct the MCTS through the same function with the same settings."""
    from arena.leiden_sparring_hub import _build_agent_from_spec
    from search import gpu_mcts_cheqqers as g

    for var in ("CHEQQERS_MCTS_BATCH", "CHEQQERS_MCTS_CPUCT", "CHEQQERS_MCTS_FPU"):
        monkeypatch.delenv(var, raising=False)
    built = []
    real = g.get_mcts_player
    monkeypatch.setattr(g, "get_mcts_player", lambda *a, **k: built.append(real(*a, **k)) or built[-1])
    _build_agent_from_spec(f"M:expecti_heuristic@ms=300,mcts={tiny_model}", "cheqqers")
    monkeypatch.setenv("CHEQQERS_MCTS_MODEL", str(tiny_model))
    monkeypatch.setenv("CHEQQERS_MCTS_DEVICE", "cpu")
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "300")
    monkeypatch.setenv("CHEQQERS_MCTS_MIN_EVALS", "1")
    server_module.setup_cheqqers_mcts()
    assert len(built) == 2 and server_module._MCTS_PLAYER is built[1]
    assert [(p.batch, p.c_puct, p.fpu) for p in built] == [g.play_settings()] * 2


def test_booked_position_answered_from_the_book_without_search(server_module: ModuleType,
                                                               monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                               capsys: pytest.CaptureFixture) -> None:
    from arena.warmup import start_payloads
    from env.cheqqers_official import OfficialCheqqersEnv
    from search.opening_book import OpeningBook

    class MustNotSearch:
        def search(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("a booked position must not be searched")

    payload = dict(start_payloads())["cheqqers"]
    env = OfficialCheqqersEnv.from_tournament_json(payload)
    stored = list(payload["legalMoves"][-1])  # not the first legal move, so a fallback would show
    path = tmp_path / "book.json"
    OpeningBook({OpeningBook.key(env): stored}).save(path)
    monkeypatch.setenv("CHEQQERS_OPENING_BOOK", str(path))
    monkeypatch.setenv("CHEQQERS_FAILURE_DIR", str(tmp_path / "failures"))
    monkeypatch.setattr(server_module, "_MCTS_PLAYER", MustNotSearch())
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", 0.0)
    assert list(move) == stored
    assert "engine=book" in capsys.readouterr().out
    assert not (tmp_path / "failures").exists()


def test_position_not_in_the_book_is_searched(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    from arena.warmup import start_payloads
    from search.gpu_mcts_cheqqers import MctsResult
    from search.opening_book import OpeningBook

    payload = dict(start_payloads())["cheqqers"]
    calls = []

    class Recording:
        def search(self, env: object, legal: list, time_ms: float = 0.0) -> MctsResult:
            calls.append(time_ms)
            return MctsResult(move=legal[0])

    path = tmp_path / "book.json"
    OpeningBook({"some other position": [0, 1]}).save(path)
    monkeypatch.setenv("CHEQQERS_OPENING_BOOK", str(path))
    monkeypatch.setenv("CHEQQERS_TIME_LIMIT_MS", "200")
    monkeypatch.setattr(server_module, "_MCTS_PLAYER", Recording())
    move = server_module.mcts_cheqqers_move(payload, payload["legalMoves"], "W", 0.0)
    assert move == payload["legalMoves"][0] and len(calls) == 1


def test_moves_the_engine_would_refuse_are_dropped(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """A move the organisers' engine refuses (entanglement limit) leaves the same side to move; a deterministic agent
    repeats it forever (17 Nov 2026). With the limit lowered to four squares (as in the harness compliance test), a
    random game reaches a position where some legal moves are refused: the server keeps only the others; when every
    move given is refused it keeps them all; at the real limit nothing is dropped there."""
    import random
    from env.cheqqers_official import JointSystem, OfficialCheqqersEnv

    monkeypatch.setattr(JointSystem, "max_dimension_limit", 16)
    rng = random.Random(1)
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
    payload = refused = None
    for _ in range(80):
        legal = env.get_legal_moves()
        refused = [m for m in legal if server_module.refused_by_engine(env, m)]
        if refused and len(refused) < len(legal):
            payload = env.to_tournament_payload()
            break
        env.apply_move(list(rng.choice(legal)))
    assert payload is not None, "no position with a refused move in this game"
    legal = payload["legalMoves"]
    kept = server_module.playable_cheqqers_moves(payload, legal)
    assert kept and all(m not in kept for m in refused) and len(kept) == len(legal) - len(refused)
    assert server_module.playable_cheqqers_moves(payload, refused) == refused
    monkeypatch.setattr(JointSystem, "max_dimension_limit", 32768)
    assert server_module.playable_cheqqers_moves(payload, legal) == legal
