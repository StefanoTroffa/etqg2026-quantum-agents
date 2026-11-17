"""The native Cheqqers engine is the server's engine (server.py, src/search/native_cheqqers.py).

- Without the native library the server refuses to start, before the warm-up and before binding its port.
- A native search that raises mid-move (a bug) is logged with the payload saved, and the reply is still a legal
  move (the last finished iteration's, else legalMoves[0]); the Python search is never asked.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_server_refuses_to_start_without_the_native_library(tmp_path):
    port = _free_port()
    env = dict(os.environ, PORT=str(port), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
               CHEQQERS_NATIVE_LIB=str(tmp_path / "missing" / "libcheqqers.so"), CHEQQERS_NATIVE_BUILD="0")
    env.pop("CHEQQERS_ENGINE", None)
    done = subprocess.run([sys.executable, "server.py"], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=180)
    out = done.stdout + done.stderr
    assert done.returncode != 0, out
    assert "native Cheqqers engine cannot be loaded" in out
    assert "READY" not in out and "Warm-up" not in out
    with socket.socket() as s:  # nothing ever listened there
        assert s.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture
def server_module(monkeypatch) -> ModuleType:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CHEQQERS_ENGINE", raising=False)
    sys.path.insert(0, str(REPO))
    import server

    return server


def test_a_native_failure_mid_move_answers_a_legal_move_without_the_python_search(server_module, monkeypatch, tmp_path):
    from arena.warmup import start_payloads
    from search import expecti_cheqqers, native_cheqqers

    def broken(*args, **kwargs):
        raise RuntimeError("simulated native bug")

    def python_search_must_not_run(*args, **kwargs):
        raise AssertionError("the Python search was asked")

    monkeypatch.setattr(native_cheqqers, "best_move", broken)
    monkeypatch.setattr(expecti_cheqqers.ExpectiCheqqersPlayer, "get_move", python_search_must_not_run)
    monkeypatch.setenv("CHEQQERS_FAILURE_DIR", str(tmp_path))
    payload = dict(start_payloads())["cheqqers"]
    move = server_module.native_cheqqers_move(payload, payload["legalMoves"], "white", 0.0)
    assert move == payload["legalMoves"][0]
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert "simulated native bug" in record["error"] and record["payload"] == payload


def test_the_native_engine_is_the_default(server_module, monkeypatch):
    assert server_module.cheqqers_engine() == "cpp"
    monkeypatch.setenv("CHEQQERS_ENGINE", "python")
    assert server_module.cheqqers_engine() == "python"
