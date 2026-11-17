"""The organisers' arena, read from its public record file (`<arena address>/games.jsonl`, one game per line).

    uv run python src/arena/arena_records.py summary <games.jsonl> [--team "<name>"]
    uv run python src/arena/arena_records.py verify  <games.jsonl>
    uv run python src/arena/arena_records.py review  <games.jsonl> --team "<name>" [--only-losses] [--judge-seconds 20]

`summary` prints, for every pairing, the results by side; for every player the time the arena measured per move
(network included) and the share of three-square moves; and the forfeits. With `--team` also that team's games
one by one. `verify` replays every Cheqqers game through `OfficialCheqqersEnv` from the recorded moves and
measurement outcomes and reports the games in which a move is not legal there, measures differently, or ends
otherwise than the arena recorded: the check that the simulator plays the arena's rules (critical workflow CW6).
`review` goes through a team's games and, at every move where it had a choice, has a search longer than a game
allows value every legal move: what the played move gave away against the best one, in evaluation units (25 is
one man). Large losses at few moves say the agent is out-searched; small losses in lost games say the evaluation
itself misjudges, since the judge shares it.

A record: {"id", "game", "rulesVersion", "players": {"white": {"name", "house"?}, "black": {...}}, "result":
"white" | "black" | "draw", "reason": "normal" | "timeout" | ..., "forfeitBy", "finalHash",
"moves": [{"by", "move", "outcomes", "ms"}]}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.cheqqers_official import GameState, OfficialCheqqersEnv, PieceColor  # noqa: E402

RESULT = {GameState.WHITE_WON: "white", GameState.BLACK_WON: "black", GameState.DRAW: "draw"}


def load(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Replay ─────────────────────────────────────────────────────────────────────────────────────────────────────


class ReplayMismatch(AssertionError):
    """A recorded game the simulator does not reproduce; the message says where and how."""


def replay(game: dict[str, Any]) -> OfficialCheqqersEnv:
    """The position after the game's recorded moves and measurement outcomes. ReplayMismatch if it is not the
    recorded side's move, the move is not one the simulator generates, or it measures another number of squares."""
    settings = game.get("settings") or {}
    env = OfficialCheqqersEnv(board_size=int(settings.get("boardSize", 6)), start_rows=int(settings.get("startRows", 2)),
                              quantumness=int(settings.get("quantumness", 2)))
    for ply, move in enumerate(game["moves"]):
        where = f"game {game.get('id')} ply {ply}"
        mover = "white" if env.turn == PieceColor.WHITE else "black"
        if mover != move["by"]:
            raise ReplayMismatch(f"{where}: the arena has {move['by']} to move, the simulator {mover}")
        if list(move["move"]) not in env.get_legal_moves():
            raise ReplayMismatch(f"{where}: move {move['move']} is not legal in the simulator")
        measured = env.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
        if measured != len(move["outcomes"]):
            raise ReplayMismatch(f"{where}: the arena measured {len(move['outcomes'])} squares, the simulator {measured}")
    return env


def verify(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Replay every game. {"games", "replayed", "same_result", "same_hash", "forfeits", "mismatches": [text]}:
    games ended by a forfeit are replayed but their result is not compared (the arena awards it)."""
    report: dict[str, Any] = {"games": len(games), "replayed": 0, "same_result": 0, "same_hash": 0, "forfeits": 0,
                              "mismatches": []}
    for game in games:
        try:
            env = replay(game)
        except ReplayMismatch as err:
            report["mismatches"].append(str(err))
            continue
        report["replayed"] += 1
        if game.get("reason", "normal") != "normal":
            report["forfeits"] += 1
            continue
        if RESULT.get(env.state) == game["result"]:
            report["same_result"] += 1
        else:
            report["mismatches"].append(
                f"game {game.get('id')}: the arena says {game['result']}, the simulator {RESULT.get(env.state, 'in progress')}")
        if "finalHash" in game and env.get_board_state_hash() == game["finalHash"]:
            report["same_hash"] += 1
    return report


# ── Summary ────────────────────────────────────────────────────────────────────────────────────────────────────


def _name(game: dict[str, Any], side: str) -> str:
    return str(game["players"][side]["name"])


def pairings(games: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """{(a, b) sorted by name: {"a_wins", "b_wins", "draws", "a_white_wins", "a_white_games", "a_black_wins",
    "a_black_games"}}."""
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for game in games:
        white, black = _name(game, "white"), _name(game, "black")
        a, b = sorted((white, black))
        row = out[(a, b)]
        winner = white if game["result"] == "white" else black if game["result"] == "black" else None
        row["a_wins" if winner == a else "b_wins" if winner == b else "draws"] += 1
        colour = "white" if white == a else "black"
        row[f"a_{colour}_games"] += 1
        row[f"a_{colour}_wins"] += winner == a
    return {pair: dict(row) for pair, row in out.items()}


def move_times(games: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per player: {"moves", "median_ms", "p95_ms", "max_ms", "three_square_share"} as the arena measured them."""
    times: dict[str, list[float]] = defaultdict(list)
    three: Counter = Counter()
    for game in games:
        for move in game["moves"]:
            who = _name(game, move["by"])
            times[who].append(float(move.get("ms", 0.0)))
            three[who] += len(move["move"]) == 3
    out = {}
    for who, values in times.items():
        values.sort()
        out[who] = {"moves": len(values), "median_ms": values[len(values) // 2],
                    "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))], "max_ms": values[-1],
                    "three_square_share": three[who] / len(values)}
    return out


def forfeits(games: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": g.get("id"), "reason": g.get("reason"), "by": _name(g, g["forfeitBy"]) if g.get("forfeitBy") else None,
             "plies": len(g["moves"])} for g in games if g.get("reason", "normal") != "normal"]


def team_games(games: Sequence[dict[str, Any]], team: str) -> list[dict[str, Any]]:
    """The team's games in order: {"id", "colour", "opponent", "result": "win" | "loss" | "draw", "reason", "plies",
    "slowest_ms"}."""
    rows = []
    for game in games:
        for colour, other in (("white", "black"), ("black", "white")):
            if _name(game, colour) != team:
                continue
            own = [float(m.get("ms", 0.0)) for m in game["moves"] if m["by"] == colour]
            result = "draw" if game["result"] == "draw" else "win" if game["result"] == colour else "loss"
            rows.append({"id": game.get("id"), "colour": colour, "opponent": _name(game, other), "result": result,
                         "reason": game.get("reason", "normal"), "plies": len(game["moves"]),
                         "slowest_ms": max(own, default=0.0)})
    return rows


# ── Review ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _review_job(job: tuple) -> Optional[dict[str, Any]]:
    """One decision: the judge's value of the played move and of the best one. None where there was no choice."""
    game, ply, judge_seconds, judge_depth = job
    from search.expecti_cheqqers import ExpectiCheqqersPlayer
    from search.opening_book import same_move

    env = replay({**game, "moves": game["moves"][:ply]})
    if len(env.get_legal_moves()) < 2:
        return None
    played = [int(x) for x in game["moves"][ply]["move"]]
    judge = ExpectiCheqqersPlayer(max_depth=judge_depth, time_limit_ms=judge_seconds * 1000.0)
    depth, values = judge.root_values(env)
    if not values:
        return None
    best, best_value = max(values.items(), key=lambda item: item[1])
    played_value = next((v for k, v in values.items() if same_move(k, played)), None)
    if played_value is None:
        return None
    return {"game": game.get("id"), "ply": ply, "by": game["moves"][ply]["by"], "ms": game["moves"][ply].get("ms"),
            "played": played, "best": [int(x) for x in best], "played_value": played_value,
            "best_value": best_value, "loss": best_value - played_value, "judge_depth": depth,
            "choices": len(values)}


def review(games: Sequence[dict[str, Any]], team: str, judge_seconds: float = 20.0, judge_depth: int = 14,
           workers: int = 1, only_losses: bool = False, against: Optional[str] = None) -> list[dict[str, Any]]:
    """For every move of `team` where it had a choice: {"game", "ply", "by", "ms", "played", "best", "played_value",
    "best_value", "loss", "judge_depth", "choices"}, values as the judge (a search of `judge_seconds` or
    `judge_depth`, default evaluation) sees them for the side to move. `only_losses` keeps the games the team lost
    on the board; `against` keeps one opponent. Games ended by a forfeit are left out."""
    jobs = []
    for game in games:
        if game.get("reason", "normal") != "normal":
            continue
        for colour, other in (("white", "black"), ("black", "white")):
            if _name(game, colour) != team or (against is not None and _name(game, other) != against):
                continue
            if only_losses and game["result"] != other:
                continue
            jobs += [(game, ply, judge_seconds, judge_depth) for ply, m in enumerate(game["moves"]) if m["by"] == colour]
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            rows = pool.map(_review_job, jobs, chunksize=1)
    else:
        rows = [_review_job(job) for job in jobs]
    return [row for row in rows if row is not None]


def review_summary(rows: Sequence[dict[str, Any]], man: float = 25.0) -> dict[str, Any]:
    """{"decisions", "mean_loss", "half_man_or_more", "by_ply": {"0-9" | "10-19" | "20+": {"decisions",
    "mean_loss", "half_man_or_more"}}, "judge_depth_median"}."""
    def part(selected: list[dict[str, Any]]) -> dict[str, Any]:
        losses = [r["loss"] for r in selected]
        return {"decisions": len(losses), "mean_loss": sum(losses) / len(losses) if losses else 0.0,
                "half_man_or_more": sum(x >= man / 2 for x in losses)}

    rows = list(rows)
    depths = sorted(r["judge_depth"] for r in rows)
    out = part(rows)
    out["by_ply"] = {"0-9": part([r for r in rows if r["ply"] < 10]),
                     "10-19": part([r for r in rows if 10 <= r["ply"] < 20]),
                     "20+": part([r for r in rows if r["ply"] >= 20])}
    out["judge_depth_median"] = depths[len(depths) // 2] if depths else 0
    return out


# ── Command line ───────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary", help="results by pairing, move times, forfeits")
    s.add_argument("file")
    s.add_argument("--team", default=None, help="also list this team's games")
    v = sub.add_parser("verify", help="replay every game through the simulator")
    v.add_argument("file")
    r = sub.add_parser("review", help="what each of a team's moves gave away, as a longer search sees it")
    r.add_argument("file")
    r.add_argument("--team", required=True)
    r.add_argument("--against", default=None, help="only the games against this opponent")
    r.add_argument("--only-losses", action="store_true")
    r.add_argument("--judge-seconds", type=float, default=20.0)
    r.add_argument("--judge-depth", type=int, default=14)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", default=None, help="write one JSON line per decision here")
    args = parser.parse_args(argv)
    games = load(args.file)

    if args.command == "verify":
        report = verify(games)
        print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}))
        for line in report["mismatches"][:20]:
            print("MISMATCH", line)
        return 1 if report["mismatches"] else 0

    if args.command == "review":
        rows = review(games, args.team, args.judge_seconds, args.judge_depth, args.workers, args.only_losses,
                      args.against)
        if args.out:
            Path(args.out).write_text("".join(json.dumps(row) + "\n" for row in rows))
        by_game: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_game[row["game"]].append(row)
        for game_id, game_rows in by_game.items():
            marks = " ".join(f"{r['ply']}:{r['loss']:.0f}" for r in sorted(game_rows, key=lambda r: r["ply"]))
            print(f"  {game_id}: {marks}")
        print(json.dumps(review_summary(rows)))
        return 0

    print(f"{len(games)} games, rules {sorted({g.get('rulesVersion') for g in games})}")
    for (a, b), row in sorted(pairings(games).items()):
        print(f"  {a} vs {b}: {row.get('a_wins', 0)} - {row.get('b_wins', 0)}, {row.get('draws', 0)} draws | {a} as white "
              f"{row.get('a_white_wins', 0)}/{row.get('a_white_games', 0)}, as black "
              f"{row.get('a_black_wins', 0)}/{row.get('a_black_games', 0)}")
    for who, t in sorted(move_times(games).items()):
        print(f"  {who}: {t['moves']} moves, median {t['median_ms']:.0f} ms, 95% {t['p95_ms']:.0f} ms, "
              f"max {t['max_ms']:.0f} ms, three-square moves {100 * t['three_square_share']:.0f}%")
    for f in forfeits(games):
        print(f"  forfeit {f['id']}: {f['reason']} by {f['by']} after {f['plies']} plies")
    if args.team:
        for row in team_games(games, args.team):
            print(f"  {row['id']} as {row['colour']} vs {row['opponent']}: {row['result']} ({row['reason']}), "
                  f"{row['plies']} plies, slowest {row['slowest_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
