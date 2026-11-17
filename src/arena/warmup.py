"""Play one move of each game through the server's own /move path before the server opens its public port.

The organisers start our server themselves and their arena calls it as soon as a game begins, so a listening
server has to be a ready one: every import, table, compiled library and network the move path touches is loaded
and has answered once. `warm_up` binds the handler to a private loopback port, sends it the tournament's start
positions over HTTP exactly as the arena does, checks that every reply is a legal move, and closes that port.
Only then does `server.run` bind the public one. A warm-up that fails stops the server loudly instead of letting
the first tournament game find out.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.request
from typing import Any, Callable

from env.cheqqers_official import OfficialCheqqersEnv
from env.tiqtaqtoe_official import OfficialTiqTaqToeEnv


def start_payloads() -> list[tuple[str, dict[str, Any]]]:
    """(name, request body) for the tournament start positions: Cheqqers 6x6 quantumness 2 with the arena's
    3000 ms limit, and TiqTaqToe in the tournament configuration (quantumness 2, decoherence off)."""
    cheqqers = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2).to_tournament_payload(timeout_ms=3000)
    tiqtaqtoe = OfficialTiqTaqToeEnv(quantumness=2, decoherence=0).to_tournament_payload()
    return [("cheqqers", cheqqers), ("tiqtaqtoe", tiqtaqtoe)]


def warm_up(handler: type[http.server.BaseHTTPRequestHandler],
            payloads: list[tuple[str, dict[str, Any]]] | None = None,
            log: Callable[[str], None] = print) -> dict[str, float]:
    """Send each payload to `handler` on a private loopback port; return {name: reply seconds}.

    Raises RuntimeError if a reply is not HTTP 200 with a legal move from the payload's `legalMoves`."""
    payloads = start_payloads() if payloads is None else payloads
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    times: dict[str, float] = {}
    try:
        for name, body in payloads:
            data = json.dumps(body).encode("utf-8")
            request = urllib.request.Request(f"http://127.0.0.1:{port}/move", data=data,
                                             headers={"Content-Type": "application/json"}, method="POST")
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as reply:
                status, text = reply.status, reply.read().decode("utf-8")
            times[name] = time.perf_counter() - started
            move = json.loads(text).get("move")
            if status != 200 or move not in body.get("legalMoves", []):
                raise RuntimeError(f"warm-up {name}: HTTP {status}, move {move!r} is not one of the legal moves")
            log(f"✅ Warm-up {name}: legal move {move} in {times[name] * 1000:.0f} ms.")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    return times
