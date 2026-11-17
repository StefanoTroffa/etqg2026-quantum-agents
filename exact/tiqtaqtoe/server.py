"""The tournament's move endpoint, answered from the solved tables.

POST /move takes the payload fetchRemoteAiMove (harness 680-) sends and answers {"move": m}, m one of its legalMoves:
the stored policy's move (solve.choose) at draw weight w for the side to move. GET /health says it is up; OPTIONS
answers the browser's CORS preflight.

Every request with a legal move gets one, so a request never forfeits on our side. The move comes from the tables when
they can answer: the settings' configuration is solved (manual measurement on, as the harness always has it), the
decoherence counters are known (decoherence off, or the payload carries decoherenceCounters), the state is in the
tables, and our legal moves are the payload's. Otherwise it is a random legal move, logged with the reason.

Run from exact/: python3 -m tiqtaqtoe.server [--port 8080] [--w 1/2] [--preload Q D]
"""

import argparse
import json
import random
import sys
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tiqtaqtoe import compact, solve
from tiqtaqtoe import engine as E
from tiqtaqtoe.solve import DATA, W
from tiqtaqtoe.teacher import Teacher


class MoveServer:
    """Chooses moves from the tables in table_dir at draw weight w, the mover's own (a draw worth w of a win). Tables
    load on first use per configuration and stay loaded."""

    def __init__(self, table_dir=DATA, w=Fraction(1, 2), rng=None):
        if w not in W:
            raise ValueError(f"draw weight {w} is not one of {', '.join(map(str, W))}")
        self.table_dir, self.w, self.rng = table_dir, Fraction(w), rng or random.Random()
        self._teachers = {}

    def teacher(self, cfg):
        """The Teacher of cfg, loaded once. RuntimeError if its tables are missing or stale."""
        if cfg not in self._teachers:
            self._teachers[cfg] = Teacher(cfg, self.table_dir)
        return self._teachers[cfg]

    def choose(self, payload):
        """(move, how): a move of payload["legalMoves"] as listed there, and "table" or "fallback: <reason>".
        ValueError if there is no legal move to give."""
        legal = payload.get("legalMoves") if isinstance(payload, dict) else None
        if not isinstance(legal, list) or not legal:
            raise ValueError("the payload lists no legalMoves")
        try:
            return self._from_tables(payload, legal), "table"
        except _Fallback as why:
            return self.rng.choice(legal), f"fallback: {why}"
        except Exception as err:  # a malformed payload or a bug: still answer, never forfeit
            return self.rng.choice(legal), f"fallback: {type(err).__name__}: {err}"

    def _from_tables(self, payload, legal):
        settings = payload.get("settings") or {}
        cfg = compact.Config(int(settings.get("quantumness", -1)), int(settings.get("decoherence", 0)), True)
        if cfg.decoherence and payload.get("decoherenceCounters") is None:
            raise _Fallback("decoherence on but the payload has no decoherenceCounters")
        try:
            t = self.teacher(cfg)
        except RuntimeError as err:
            raise _Fallback(f"no tables for {cfg.name}: {err}") from None
        e = E.from_round_state(payload, counters=(0,) * 9 if not cfg.decoherence else None)
        by_ours = {}
        for m in legal:
            by_ours[E.move_from_harness(m)] = m
        x_w = self.w if e.turn == 1 else 1 - self.w  # the tables score for X
        try:
            cands = [(m, entry) for m, _, entry in t.move_values(e, x_w)]
        except ValueError as err:  # OutOfConfiguration included
            raise _Fallback(str(err)) from None
        if sorted(by_ours) != sorted(m for m, _ in cands):
            raise _Fallback("legalMoves differ from ours")
        move, _ = solve.choose(e.turn, cands, x_w)
        return by_ours[move]


class _Fallback(Exception):
    """The tables cannot answer this request; the message says why."""


def make_http_server(host, port, mover):
    """A ThreadingHTTPServer on (host, port) answering with mover (a MoveServer)."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok", "ai": "exact-teacher", "endpoint": "POST /move",
                                 "supports": ["tiqtaqtoe"]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/move":
                return self._send(404, {"error": "not found"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                move, how = mover.choose(payload)
            except (ValueError, json.JSONDecodeError) as err:
                return self._send(400, {"error": str(err)})
            print(f"[exact-teacher] player={payload.get('player')} legal={len(payload['legalMoves'])} "
                  f"-> {json.dumps(move)} ({how})", file=sys.stderr, flush=True)
            self._send(200, {"move": move})

        def log_message(self, *args):  # one line per move from do_POST is enough
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    """Serve POST /move from the solved tables."""
    parser = argparse.ArgumentParser(prog="python3 -m tiqtaqtoe.server", description=main.__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--w", type=Fraction, default=Fraction(1, 2), help="own draw weight, one of 0 1/4 1/2 3/4 1")
    parser.add_argument("--preload", nargs=2, type=int, default=(2, 0), metavar=("Q", "D"),
                        help="load these tables before serving (default: the tournament's Q2, decoherence off)")
    args = parser.parse_args(argv)
    try:
        mover = MoveServer(DATA, args.w)
        mover.teacher(compact.Config(args.preload[0], args.preload[1], True))
    except (ValueError, RuntimeError) as err:
        parser.error(str(err))
    httpd = make_http_server(args.host, args.port, mover)
    print(f"[exact-teacher] POST http://{args.host}:{args.port}/move  (w = {args.w})", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
