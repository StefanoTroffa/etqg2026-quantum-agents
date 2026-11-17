"""The agent switch: one address for the arena connector, behind it whichever move server a file names.

Stopping the connector to change servers forfeits the game the arena starts at that moment (three games on
13 November 2026). With the switch the connector stays up and the next request goes to the new server.
"""

import http.server
import json
import threading
import urllib.error
import urllib.request
from typing import Iterator

import pytest

from arena.agent_switch import serve


class Fixed(http.server.BaseHTTPRequestHandler):
    """A stand-in move server: answers every POST with its own name and the body it was sent."""

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        answer = json.dumps({"server": self.server.label, "path": self.path, "sent": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def do_GET(self) -> None:  # noqa: N802
        answer = json.dumps({"server": self.server.label, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *args) -> None:
        pass


def start(server: http.server.HTTPServer) -> http.server.HTTPServer:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def setup(tmp_path) -> Iterator[dict]:
    """Two stand-in move servers and a switch pointed at the first: {"switch", "target", "urls"}."""
    backends = []
    for label in ("first", "second"):
        backend = http.server.HTTPServer(("127.0.0.1", 0), Fixed)
        backend.label = label
        backends.append(start(backend))
    target = tmp_path / "target.txt"
    target.write_text(f"http://127.0.0.1:{backends[0].server_address[1]}\n")
    switch = start(serve(0, target))
    yield {"switch": f"http://127.0.0.1:{switch.server_address[1]}", "target": target,
           "urls": [f"http://127.0.0.1:{b.server_address[1]}" for b in backends]}
    for server in backends + [switch]:
        server.shutdown()


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


# ── Forwarding and switching ───────────────────────────────────────────────────────────────────────────────────


class TestAgentSwitch:
    def test_request_reaches_the_server_the_file_names_with_its_path_and_body(self, setup):
        answer = post(setup["switch"] + "/move", {"game": "cheqqers", "ply": 3})
        assert answer == {"server": "first", "path": "/move", "sent": {"game": "cheqqers", "ply": 3}}

    def test_next_request_goes_to_the_new_server_when_the_file_changes(self, setup):
        assert post(setup["switch"] + "/move", {})["server"] == "first"
        setup["target"].write_text(setup["urls"][1] + "\n")
        assert post(setup["switch"] + "/move", {})["server"] == "second"

    def test_health_is_forwarded_too(self, setup):
        with urllib.request.urlopen(setup["switch"] + "/health", timeout=5) as response:
            assert json.loads(response.read()) == {"server": "first", "path": "/health"}

    def test_switch_answers_502_when_the_named_server_is_down(self, setup):
        setup["target"].write_text("http://127.0.0.1:9\n")
        with pytest.raises(urllib.error.HTTPError) as error:
            post(setup["switch"] + "/move", {})
        assert error.value.code == 502

    def test_switch_keeps_the_old_server_when_the_file_is_empty_or_not_an_address(self, setup):
        assert post(setup["switch"] + "/move", {})["server"] == "first"
        setup["target"].write_text("\n")
        assert post(setup["switch"] + "/move", {})["server"] == "first"
        setup["target"].write_text("second server please\n")
        assert post(setup["switch"] + "/move", {})["server"] == "first"
