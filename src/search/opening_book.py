"""An opening book for Cheqqers: the moves a long search chooses in the first positions of a game (Track-O).

The book stores, for every position its side can meet in its first decisions, the move a search much longer than
a game's time limit chose there. At move time it is a lookup; a position that is not in the book, or a stored
move that is not legal, goes to the search as before. The book is made by our own search and evaluation and has
to be rebuilt when either changes.

    uv run python src/search/opening_book.py build --own-moves 3 --seconds 20 --workers 7 --out config/cheqqers_opening_book.json
    uv run python src/search/opening_book.py show  config/cheqqers_opening_book.json
    uv run python src/search/opening_book.py check config/cheqqers_opening_book.json --agent-ms 2200 --judge-seconds 60

`build` walks the game from the start once for each colour: where the book's side is to move it searches and
follows its own move through every measurement outcome; where the other side is to move it follows every legal
move. `check` walks the same positions, asks the agent at its game time limit for a move, and where that differs
from the book's has a longer search value both.

A position is keyed by the simulator's state hash: the side to move, every square's colour, crown and presence,
and the square of an open capture sequence. Two positions with the same presences and different entanglement
share a key (known limit).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.cheqqers_official import EvalWeights, OfficialCheqqersEnv, PieceColor  # noqa: E402


def same_move(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether two move arrays name the same move: equal, or a three-square move with its two targets (split) or
    its two sources (merge) listed in the other order."""
    a, b = [int(x) for x in a], [int(x) for x in b]
    if a == b:
        return True
    if len(a) == 3 and len(b) == 3:
        return (a[0] == b[0] and {a[1], a[2]} == {b[1], b[2]}) or (a[2] == b[2] and {a[0], a[1]} == {b[0], b[1]})
    return False


class OpeningBook:
    """{state hash: move as an array of squares}. `settings` records how the book was made."""

    def __init__(self, moves: Optional[dict[str, list[int]]] = None, settings: Optional[dict[str, Any]] = None):
        self.moves: dict[str, list[int]] = dict(moves or {})
        self.settings: dict[str, Any] = dict(settings or {})

    def __len__(self) -> int:
        return len(self.moves)

    @staticmethod
    def key(env: OfficialCheqqersEnv) -> str:
        return env.get_state_hash()

    def lookup(self, env: OfficialCheqqersEnv, legal_moves: Sequence[Any]) -> Optional[Any]:
        """The member of `legal_moves` that is the book's move for this position, or None: the position is not in
        the book, or the stored move is not among the legal ones."""
        stored = self.moves.get(self.key(env))
        if stored is None:
            return None
        for move in legal_moves:
            if isinstance(move, (list, tuple)) and same_move(move, stored):
                return move
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"settings": self.settings, "moves": self.moves}

    @classmethod
    def from_dict(cls, data: Any) -> "OpeningBook":
        if not isinstance(data, dict) or not isinstance(data.get("moves"), dict):
            raise ValueError('not an opening book: expected {"settings": ..., "moves": {state hash: move}}')
        return cls({str(k): [int(x) for x in v] for k, v in data["moves"].items()}, data.get("settings"))

    @classmethod
    def from_json(cls, path: str | Path) -> "OpeningBook":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")


_BOOK_BY_PATH: dict[str, OpeningBook] = {}


def book_from_environment() -> Optional[OpeningBook]:
    """The book named by the environment variable CHEQQERS_OPENING_BOOK, read once per path; None when the variable
    is unset or empty. A file that is missing or is not a book raises: call this once at start-up."""
    path = os.getenv("CHEQQERS_OPENING_BOOK", "").strip()
    if not path:
        return None
    if path not in _BOOK_BY_PATH:
        _BOOK_BY_PATH[path] = OpeningBook.from_json(path)
    return _BOOK_BY_PATH[path]


# ── Walking the first positions of a game ──────────────────────────────────────────────────────────────────────


def start_position() -> OfficialCheqqersEnv:
    """The tournament's start position: 6x6, two rows of men, quantumness 2."""
    return OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)


def _children(env: OfficialCheqqersEnv, moves: Sequence[Sequence[int]]) -> list[OfficialCheqqersEnv]:
    """Every position that follows `env` by one of `moves` and any measurement outcome, games in progress only."""
    out = []
    for move in moves:
        for probability, child in env.get_move_outcomes(list(move)):
            if probability > 1e-9 and not child.done:
                out.append(child)
    return out


def _unique(positions: Sequence[OfficialCheqqersEnv]) -> list[OfficialCheqqersEnv]:
    seen: dict[str, OfficialCheqqersEnv] = {}
    for env in positions:
        seen.setdefault(OpeningBook.key(env), env)
    return list(seen.values())


def _to_own_turn(positions: Sequence[OfficialCheqqersEnv], side: PieceColor) -> list[OfficialCheqqersEnv]:
    """Follow every legal move of the other side, through its capture sequences, until `side` is to move."""
    ours: list[OfficialCheqqersEnv] = []
    waiting = list(positions)
    for _ in range(12):  # a capture sequence on this board is far shorter
        ours += [e for e in waiting if e.turn == side]
        theirs = _unique([e for e in waiting if e.turn != side])
        if not theirs:
            break
        waiting = [child for e in theirs for child in _children(e, e.get_legal_moves())]
    return _unique(ours)


def walk(side: PieceColor, own_moves: int,
         choose: Callable[[list[OfficialCheqqersEnv]], dict[str, list[int]]]) -> list[list[OfficialCheqqersEnv]]:
    """The positions in which `side` makes each of its first `own_moves` decisions, decision by decision.
    `choose` is given the positions of one decision and returns {key: move} for those it has a move for; the walk
    follows that move, and every legal move where there is none (a single legal move, a position left out)."""
    levels = []
    frontier = [start_position()]
    for _ in range(own_moves):
        ours = _to_own_turn(frontier, side)
        if not ours:
            break
        levels.append(ours)
        chosen = choose(ours)
        frontier = []
        for env in ours:
            move = chosen.get(OpeningBook.key(env))
            frontier += _children(env, [move] if move is not None else env.get_legal_moves())
    return levels


# ── Building ───────────────────────────────────────────────────────────────────────────────────────────────────


def _search_job(job: tuple) -> tuple[str, Optional[list[int]]]:
    """(key, the long search's move) for one position; no move if the search fails."""
    env, seconds, depth, weights = job
    from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move

    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=seconds * 1000.0, eval_weights=weights)
    try:
        move = player.get_move(env, available_moves=env.get_legal_moves())
    except Exception as err:  # noqa: BLE001 - one position must not cost the book
        print(f"search failed at {OpeningBook.key(env)}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        return OpeningBook.key(env), None
    return OpeningBook.key(env), [int(x) for x in canonicalize_cheqqers_move(move)]


def _run(function: Callable, jobs: list, workers: int) -> list:
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(function, jobs, chunksize=1)
    return [function(job) for job in jobs]


def build(own_moves: int, seconds: float, depth: int = 12, workers: int = 1,
          weights: Optional[EvalWeights] = None,
          on_level: Optional[Callable[[str, int, int, "OpeningBook"], None]] = None) -> OpeningBook:
    """The book for the first `own_moves` decisions of either colour: each position searched for `seconds` or to
    `depth`, whichever ends first, one decision at a time on `workers` processes. A position with a single legal
    move is not stored. `on_level(colour, decision, positions searched, book so far)` is called after each."""
    book = OpeningBook({}, {"own_moves": own_moves, "seconds": seconds, "depth": depth,
                            "weights": weights.to_dict() if weights is not None else "default"})
    for side in (PieceColor.WHITE, PieceColor.BLACK):
        decision = 0

        def choose(positions: list[OfficialCheqqersEnv]) -> dict[str, list[int]]:
            nonlocal decision
            decision += 1
            todo = [e for e in positions if OpeningBook.key(e) not in book.moves and len(e.get_legal_moves()) > 1]
            for key, move in _run(_search_job, [(e, seconds, depth, weights) for e in todo], workers):
                if move is not None:
                    book.moves[key] = move
            if on_level is not None:
                on_level(side.name.lower(), decision, len(todo), book)
            return book.moves

        walk(side, own_moves, choose)
    return book


# ── Building with the network agent's own long search (16 Nov 2026) ───────────────────────────────────────────
# The FW build above stores FW's choices, and the network agent at 2.2 s already beats FW: an FW book would
# replace good moves with worse ones. This build stores what the network's MCTS chooses with `seconds` instead of
# 2.2 s, and follows the opponent's likely replies (all of them in its first `opp_all` turns, then the `opp_top`
# most visited by an `opp_seconds` search) and every measurement outcome.

_NET_PLAYER = None


def _net_init(model: str, device: str) -> None:
    global _NET_PLAYER
    from search.gpu_mcts_cheqqers import get_mcts_player
    _NET_PLAYER = get_mcts_player(model, device=device)


def _net_job(job: tuple) -> tuple[str, Optional[list[int]], list[list[int]], dict[str, Any]]:
    """(key, the search's move, its moves by visits, stats) for one position."""
    env, seconds = job
    try:
        r = _NET_PLAYER.search(env, env.get_legal_moves(), time_ms=seconds * 1000.0)
    except Exception as err:  # a failed search leaves the position out (the agent searches it at play time)
        return OpeningBook.key(env), None, [], {"error": f"{type(err).__name__}: {err}"}
    ranked = [list(m) for m, _v, _q, _p in sorted(r.visits, key=lambda row: -row[1])]
    return OpeningBook.key(env), list(r.move), ranked, {"evals": r.evals, "value": round(float(r.root_value), 4)}


def arena_positions(records: str | Path, team: str, max_ply: int) -> list[OfficialCheqqersEnv]:
    """The positions where `team` was to move before ply `max_ply` in the arena's game records (the recorded moves
    and measurement outcomes replayed), unique by key: the lines real opponents steer into."""
    import copy
    out: list[OfficialCheqqersEnv] = []
    for line in Path(records).read_text().splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        names = {side: (game.get("players", {}).get(side) or {}).get("name") for side in ("white", "black")}
        if team not in names.values():
            continue
        ours = "white" if names["white"] == team else "black"
        settings = game.get("settings") or {}
        env = OfficialCheqqersEnv(board_size=int(settings.get("boardSize", 6)),
                                  start_rows=int(settings.get("startRows", 2)),
                                  quantumness=int(settings.get("quantumness", 2)))
        for ply, move in enumerate(game["moves"][:max_ply]):
            if move["by"] == ours and len(env.get_legal_moves()) > 1:
                out.append(copy.deepcopy(env))
            try:
                env.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
            except Exception:
                break  # a record the simulator cannot follow: keep what came before
    return _unique(out)


def build_net(model: str, own_moves: int, seconds: float, opp_seconds: float, opp_all: int, opp_top: int,
              workers: int = 1, device: str = "cuda", out: Optional[Path] = None,
              log: Callable[[str], None] = lambda line: None,
              extra: Sequence[OfficialCheqqersEnv] = ()) -> OpeningBook:
    """The network book for the first `own_moves` decisions of either colour (see the note above). With `out`, the
    book is saved after every level and an existing file is resumed (its positions are not searched again)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    book = OpeningBook({}, {"kind": "network", "model": str(model), "own_moves": own_moves, "seconds": seconds,
                            "opp_seconds": opp_seconds, "opp_all": opp_all, "opp_top": opp_top})
    if out is not None and Path(out).exists():
        old = OpeningBook.from_json(out)
        book.moves.update(old.moves)
        log(f"resuming {out}: {len(old)} positions already in the book")
    replies: dict[str, list[list[int]]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_net_init, initargs=(str(model), device)) as pool:
        def search(positions: list[OfficialCheqqersEnv], secs: float) -> list[tuple]:
            return list(pool.map(_net_job, [(e, secs) for e in positions]))

        for side in (PieceColor.WHITE, PieceColor.BLACK):
            frontier = [start_position()]
            for decision in range(1, own_moves + 1):
                # the opponent's turns up to our next decision: all replies early, the likeliest later
                ours: list[OfficialCheqqersEnv] = []
                waiting = list(frontier)
                for _ in range(12):
                    ours += [e for e in waiting if e.turn == side]
                    theirs = _unique([e for e in waiting if e.turn != side])
                    if not theirs:
                        break
                    if decision <= opp_all:
                        moves_of = {OpeningBook.key(e): e.get_legal_moves() for e in theirs}
                    else:
                        todo = [e for e in theirs if OpeningBook.key(e) not in replies and len(e.get_legal_moves()) > 1]
                        for key, _move, ranked, _st in search(todo, opp_seconds):
                            replies[key] = ranked[:opp_top]
                        moves_of = {OpeningBook.key(e): (replies.get(OpeningBook.key(e)) or e.get_legal_moves())
                                    for e in theirs}
                    waiting = [c for e in theirs for c in _children(e, moves_of[OpeningBook.key(e)])]
                ours = _unique(ours)
                if not ours:
                    break
                todo = [e for e in ours if OpeningBook.key(e) not in book.moves and len(e.get_legal_moves()) > 1]
                log(f"{side.name.lower()} decision {decision}: {len(ours)} positions, {len(todo)} to search "
                    f"at {seconds:g} s")
                for key, move, _ranked, st in search(todo, seconds):
                    if move is not None:
                        book.moves[key] = move
                    else:
                        log(f"  search failed: {st.get('error')}")
                if out is not None:
                    book.save(out)
                frontier = []
                for env in ours:
                    move = book.moves.get(OpeningBook.key(env))
                    frontier += _children(env, [move] if move is not None else env.get_legal_moves())
        todo = [e for e in _unique(list(extra)) if OpeningBook.key(e) not in book.moves]
        if extra:
            log(f"arena lines: {len(extra)} positions, {len(todo)} to search at {seconds:g} s")
        for key, move, _ranked, st in search(todo, seconds):
            if move is not None:
                book.moves[key] = move
        if out is not None and todo:
            book.save(out)
    return book


# ── Checking a book against the agent it is for ────────────────────────────────────────────────────────────────


def _check_job(job: tuple) -> Optional[dict[str, Any]]:
    """For one book position: the agent's move at its game time limit and, where it is not the book's, the value
    of both under the judge. None where the agent plays the book move."""
    env, book_move, agent_ms, judge_seconds, weights = job
    from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move

    agent = ExpectiCheqqersPlayer(max_depth=12, time_limit_ms=agent_ms, eval_weights=weights)
    agent_move = [int(x) for x in canonicalize_cheqqers_move(agent.get_move(env, available_moves=env.get_legal_moves()))]
    if same_move(agent_move, book_move):
        return None
    judge = ExpectiCheqqersPlayer(max_depth=14, time_limit_ms=judge_seconds * 1000.0, eval_weights=weights)
    depth, values = judge.root_values(env)
    value = {tuple(k): v for k, v in values.items()}
    find = lambda move: next((v for k, v in value.items() if same_move(k, move)), None)  # noqa: E731
    return {"key": OpeningBook.key(env), "book": book_move, "agent": agent_move, "judge_depth": depth,
            "book_value": find(book_move), "agent_value": find(agent_move)}


def check(book: OpeningBook, agent_ms: float, judge_seconds: float, workers: int = 1,
          weights: Optional[EvalWeights] = None) -> dict[str, Any]:
    """{"positions", "agent_differs", "judged", "book_better", "agent_better", "equal", "mean_gain", "rows"}: over
    the book's positions, how often the agent at `agent_ms` a move plays another move, and for those the value of
    the book's move minus the agent's under a search of `judge_seconds` (in evaluation units; 25 is one man)."""
    jobs = []
    for side in (PieceColor.WHITE, PieceColor.BLACK):
        for level in walk(side, int(book.settings.get("own_moves", 0)), lambda positions: book.moves):
            jobs += [(env, book.moves[OpeningBook.key(env)], agent_ms, judge_seconds, weights)
                     for env in level if OpeningBook.key(env) in book.moves]
    rows = [row for row in _run(_check_job, jobs, workers) if row is not None]
    judged = [r for r in rows if r["book_value"] is not None and r["agent_value"] is not None]
    gains = [r["book_value"] - r["agent_value"] for r in judged]
    return {"positions": len(jobs), "agent_differs": len(rows), "judged": len(judged),
            "book_better": sum(g > 1e-9 for g in gains), "agent_better": sum(g < -1e-9 for g in gains),
            "equal": sum(abs(g) <= 1e-9 for g in gains),
            "mean_gain": sum(gains) / len(gains) if gains else 0.0, "rows": rows}


# ── Command line ───────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    default_workers = max(1, (os.cpu_count() or 2) // 2 - 1)
    b = sub.add_parser("build", help="search the first positions of the game and write the book")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--own-moves", type=int, default=3)
    b.add_argument("--seconds", type=float, default=20.0)
    b.add_argument("--depth", type=int, default=12)
    b.add_argument("--workers", type=int, default=default_workers)
    b.add_argument("--weights", type=Path, default=None)
    n = sub.add_parser("build-net", help="the book from the network agent's own long search")
    n.add_argument("--model", default="config/cheqqers_net.pt")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--own-moves", type=int, default=4)
    n.add_argument("--seconds", type=float, default=20.0)
    n.add_argument("--opp-seconds", type=float, default=1.0)
    n.add_argument("--opp-all", type=int, default=1, help="our decisions before which every opponent reply is followed")
    n.add_argument("--opp-top", type=int, default=2)
    n.add_argument("--workers", type=int, default=2)
    n.add_argument("--device", default="cuda")
    n.add_argument("--arena", type=Path, default=None, help="arena game records (JSONL): add our positions from them")
    n.add_argument("--team", default="", help="our team's name in --arena")
    n.add_argument("--arena-plies", type=int, default=24)
    s = sub.add_parser("show", help="print a book's size and settings")
    s.add_argument("file", type=Path)
    c = sub.add_parser("check", help="compare the book's moves with the agent's under a longer search")
    c.add_argument("file", type=Path)
    c.add_argument("--agent-ms", type=float, default=2200.0)
    c.add_argument("--judge-seconds", type=float, default=60.0)
    c.add_argument("--workers", type=int, default=default_workers)
    c.add_argument("--weights", type=Path, default=None)
    c.add_argument("--out", type=Path, default=None, help="write the positions where the agent differs here")
    args = parser.parse_args(argv)

    if args.command == "show":
        book = OpeningBook.from_json(args.file)
        print(f"{len(book)} positions; settings {json.dumps(book.settings)}")
        return 0

    if args.command == "build-net":
        extra = arena_positions(args.arena, args.team, args.arena_plies) if args.arena else []
        book = build_net(args.model, args.own_moves, args.seconds, args.opp_seconds, args.opp_all, args.opp_top,
                         args.workers, args.device, args.out, log=lambda line: print(line, file=sys.stderr, flush=True),
                         extra=extra)
        book.settings.update({"arena_positions": len(extra), "arena_plies": args.arena_plies if args.arena else 0})
        book.save(args.out)
        print(f"{len(book)} positions written to {args.out}")
        return 0

    weights = EvalWeights.from_json(args.weights) if args.weights else None

    if args.command == "check":
        report = check(OpeningBook.from_json(args.file), args.agent_ms, args.judge_seconds, args.workers, weights)
        if args.out:
            args.out.write_text("".join(json.dumps(row) + "\n" for row in report["rows"]))
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
        return 0

    def on_level(colour: str, decision: int, searched: int, book: OpeningBook) -> None:
        book.save(args.out)  # a build that is interrupted leaves the decisions finished so far
        print(f"{colour}, decision {decision}: {searched} positions searched, {len(book)} in the book",
              file=sys.stderr, flush=True)

    book = build(args.own_moves, args.seconds, args.depth, args.workers, weights, on_level)
    book.save(args.out)
    print(f"{len(book)} positions written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
