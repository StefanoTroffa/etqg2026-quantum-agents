"""The server warms up before it opens its public port (src/arena/warmup.py)."""

import http.server
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from arena.warmup import start_payloads, warm_up

REPO = Path(__file__).resolve().parents[2]


def _move_handler(reply_move):
    """A request handler that answers every /move with reply_move(request body)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"move": reply_move(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    return type("MoveHandler", (http.server.BaseHTTPRequestHandler,),
                {"do_POST": do_POST, "log_message": lambda self, *args: None})


def test_start_payloads_are_the_tournament_positions():
    payloads = dict(start_payloads())
    assert payloads["cheqqers"]["settings"] == {"boardSize": 6, "quantumness": 2, "timeout": 3000}
    assert payloads["cheqqers"]["legalMoves"]
    assert payloads["tiqtaqtoe"]["settings"]["quantumness"] == 2
    assert payloads["tiqtaqtoe"]["legalMoves"]


def test_warm_up_accepts_legal_moves():
    times = warm_up(_move_handler(lambda body: body["legalMoves"][0]), log=lambda _: None)
    assert set(times) == {"cheqqers", "tiqtaqtoe"}


def test_warm_up_rejects_an_illegal_move():
    with pytest.raises(RuntimeError, match="not one of the legal moves"):
        warm_up(_move_handler(lambda body: [99, 98]), log=lambda _: None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_answers_its_first_move_within_the_limit_once_listening():
    """Start the real server; from the first moment its port accepts a connection, a Cheqqers move comes back
    within the arena's 3000 ms limit."""
    port = _free_port()
    env = dict(os.environ, PORT=str(port), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    env.pop("CHEQQERS_EVAL_WEIGHTS", None)  # the server's own default must apply
    server = subprocess.Popen([sys.executable, "server.py"], cwd=REPO, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            assert server.poll() is None, server.stdout.read()
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("server did not listen within 120 s")
        body = dict(start_payloads())["cheqqers"]
        request = urllib.request.Request(f"http://127.0.0.1:{port}/move", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=10) as reply:
            move = json.loads(reply.read())["move"]
        assert time.perf_counter() - started < 3.0
        assert move in body["legalMoves"]
    finally:
        server.terminate()
        output, _ = server.communicate(timeout=10)
    # The tournament default is the fitted evaluation, a file the repo ships (not under the ignored results/).
    assert "weights from config/cheqqers_eval_weights_2026-11-13.json" in output
