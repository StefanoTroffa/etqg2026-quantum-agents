"""One address for the arena connector, behind it whichever move server a file names.

    uv run python src/arena/agent_switch.py --port 8090 --target-file /tmp/cheqqers_target.txt

The file holds the base address of a move server, for example `http://127.0.0.1:8080`. Every request to the switch
is forwarded there with its path and body; the file is read again for each request, so writing another address
into it sends the next move to another server while the connector stays connected. Stopping the connector to
change servers forfeits the game the arena starts at that moment (three games on 13 November 2026).

A file that is empty or does not hold an http address leaves the last good address in use. If the named server
does not answer, the switch answers 502 and the connector's own handling of a failed move applies.
"""

from __future__ import annotations

import argparse
import http.server
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence


class _Switch(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], target_file: Path, timeout_s: float):
        super().__init__(address, _Forward)
        self.target_file = Path(target_file)
        self.timeout_s = timeout_s
        self.last_good: Optional[str] = None

    def target(self) -> Optional[str]:
        """The address in the file if it is one, else the last one that was."""
        try:
            text = self.target_file.read_text().strip()
        except OSError:
            text = ""
        if text.startswith("http://") or text.startswith("https://"):
            if text != self.last_good:
                print(f"switch: forwarding to {text}", file=sys.stderr, flush=True)
            self.last_good = text.rstrip("/")
        return self.last_good


class _Forward(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, body: Optional[bytes]) -> None:
        base = self.server.target()
        if base is None:
            self._answer(502, b'{"error": "the switch has no move server address"}')
            return
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")} if body is not None else {}
        request = urllib.request.Request(base + self.path, data=body, headers=headers,
                                         method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.server.timeout_s) as response:
                self._answer(response.status, response.read(), response.headers.get("Content-Type"))
        except urllib.error.HTTPError as error:
            self._answer(error.code, error.read(), error.headers.get("Content-Type"))
        except (urllib.error.URLError, OSError) as error:
            self._answer(502, f'{{"error": "move server unreachable: {type(error).__name__}"}}'.encode())

    def _answer(self, status: int, payload: bytes, content_type: Optional[str] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        self._forward(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def do_GET(self) -> None:  # noqa: N802
        self._forward(None)

    def log_message(self, *args) -> None:
        pass


def serve(port: int, target_file: str | Path, timeout_s: float = 10.0) -> _Switch:
    """The switch, bound to 127.0.0.1:`port` (0: any free port) and not yet serving: call serve_forever()."""
    return _Switch(("127.0.0.1", port), Path(target_file), timeout_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the move server")
    args = parser.parse_args(argv)
    switch = serve(args.port, args.target_file, args.timeout)
    print(f"switch: listening on http://127.0.0.1:{switch.server_address[1]}, reading {args.target_file}",
          file=sys.stderr, flush=True)
    try:
        switch.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
