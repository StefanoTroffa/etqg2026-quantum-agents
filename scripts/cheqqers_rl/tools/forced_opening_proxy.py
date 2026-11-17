"""forced_opening_proxy.py: a /move proxy in front of an opponent that forces white's FIRST move.

    uv run python scripts/cheqqers_rl/tools/forced_opening_proxy.py --port 9100 --target http://127.0.0.1:8082 \
        --moves 13,10 13,11 [--log proxy.log]

At the real start position (by the board: see is_start) it answers the next move of --moves in turn
(cycling, one per game), and forwards every other request unchanged to --target/move. Used to test our agent as
black against a fixed white opening, with the opponent playing normally afterwards. Standard library only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

START_PIECES = ["black"] * 6 + [None] * 6 + ["white"] * 6


def is_start(data: dict) -> bool:
    """The real start position: black men on 0-5, white men on 12-17, nothing in between, nothing crowned or
    superposed. (Not the legal-move list: a later position can have the same white moves.)"""
    board = data.get("board") or []
    if len(board) != 18 or str(data.get("player", "")).lower() != "white":
        return False
    pieces = [sq.get("piece") for sq in board]
    calm = all(not sq.get("crowned") and float(sq.get("probability", 1.0) if sq.get("piece") else 1.0) >= 0.999
               for sq in board)
    return pieces == START_PIECES and calm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--moves", nargs="+", required=True, help="forced first moves, e.g. 13,10 13,11")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    forced = itertools.cycle([[int(x) for x in m.split(",")] for m in args.moves])
    lock = threading.Lock()

    def log(line: str) -> None:
        if args.log:
            with open(args.log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def _send(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            try:
                with urllib.request.urlopen(args.target + self.path, timeout=5) as r:
                    self._send(r.status, r.read())
            except Exception as err:
                self._send(502, json.dumps({"error": str(err)}).encode())

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                data = json.loads(raw)
                legal = data.get("legalMoves") or []
                if is_start(data):
                    with lock:
                        move = next(forced)
                    if move in legal:
                        log(f"forced first move {move}")
                        self._send(200, json.dumps({"move": move}).encode())
                        return
            except Exception:
                pass
            req = urllib.request.Request(args.target + "/move", data=raw, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    self._send(r.status, r.read())
            except Exception as err:
                self._send(502, json.dumps({"error": str(err)}).encode())

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
