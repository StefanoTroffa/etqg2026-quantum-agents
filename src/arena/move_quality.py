"""Move quality: what an agent's choices give away on recorded positions, as a much longer search sees them.

    uv run python src/arena/move_quality.py judge <games.jsonl> --out judged.jsonl [--team "<name>"] [--seconds 60]
    uv run python src/arena/move_quality.py compare judged.jsonl <games.jsonl> \\
        --agent "Plain:expecti_heuristic@depth=12,ms=2200" --agent "Q4:expecti_heuristic@depth=12,ms=2200,q=1"

`judge` takes every position of the recorded games in which the side to move had a choice (with `--team`, only
that team's moves) and has the default search value every legal move with a full window for `--seconds`. It
appends to `--out` and skips what is already there, so it can be stopped and resumed. `compare` lets each agent
(a spec of `leiden_sparring_hub.py`) choose a move in the same positions at its own time limit and reports what
the choice gave away against the judge's best move: mean loss, how many decisions cost half a man or more and
two men or more (25 evaluation units are one man), and the depth the agent finished.

Why: a series between two versions of the agent needs hundreds of games to see 30 Elo. This reads hundreds of
decisions in minutes. Its limit: the judge is our own search and evaluation with more time. It sees what a
deeper look reveals (tactics); it cannot see what the evaluation itself misjudges, and it favours agents that
think like it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from arena.arena_records import load, replay  # noqa: E402

MAN = 25.0  # evaluation units


def decisions(games: Sequence[dict[str, Any]],
              keep: Optional[Callable[[dict[str, Any], int], bool]] = None) -> list[dict[str, Any]]:
    """Every position of the games in which the side to move had more than one legal move, in order:
    {"id", "ply", "game": the game cut before that ply}. `keep(game, ply)` selects among them."""
    out = []
    for game in games:
        env = replay({**game, "moves": []})
        for ply, move in enumerate(game["moves"]):
            if len(env.get_legal_moves()) > 1 and (keep is None or keep(game, ply)):
                out.append({"id": game.get("id"), "ply": ply, "game": {**game, "moves": game["moves"][:ply]}})
            env.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))
    return out


# ── The judge ──────────────────────────────────────────────────────────────────────────────────────────────────


def _judge_job(job: tuple) -> dict[str, Any]:
    decision, seconds, depth = job
    from search.expecti_cheqqers import ExpectiCheqqersPlayer

    env = replay(decision["game"])
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=seconds * 1000.0)
    finished, values = player.root_values(env)
    return {"id": decision["id"], "ply": decision["ply"], "judge_depth": finished, "judge_seconds": seconds,
            "judge_max_depth": depth, "values": [[[int(x) for x in move], value] for move, value in values.items()]}


def _run(function: Callable, jobs: list, workers: int) -> list:
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(function, jobs, chunksize=1)
    return [function(job) for job in jobs]


def judge(positions: Sequence[dict[str, Any]], seconds: float = 60.0, depth: int = 14, workers: int = 1,
          cache_path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """For each decision {"id", "ply", "judge_depth", "judge_seconds", "judge_max_depth", "values": [[move,
    value], ...]}: the value of every legal move for the side to move, by the default search with a full window,
    `seconds` or `depth` at most. Rows already in `cache_path` with the same seconds and depth are not searched
    again; new ones are appended there."""
    known: dict[tuple, dict[str, Any]] = {}
    if cache_path is not None and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("judge_seconds") == seconds and row.get("judge_max_depth") == depth:
                    known[(row["id"], row["ply"])] = row
    todo = [d for d in positions if (d["id"], d["ply"]) not in known]
    fresh = _run(_judge_job, [(d, seconds, depth) for d in todo], workers)
    if cache_path is not None and fresh:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as out:
            for row in fresh:
                out.write(json.dumps(row) + "\n")
    for row in fresh:
        known[(row["id"], row["ply"])] = row
    return [known[(d["id"], d["ply"])] for d in positions]


# ── An agent against the judge ─────────────────────────────────────────────────────────────────────────────────


def _agent_job(job: tuple) -> dict[str, Any]:
    row, game_prefix, spec = job
    from arena.leiden_sparring_hub import _build_agent_from_spec
    from search.expecti_cheqqers import canonicalize_cheqqers_move
    from search.opening_book import same_move

    env = replay(game_prefix)
    player = _build_agent_from_spec(spec, "cheqqers").player
    move = [int(x) for x in canonicalize_cheqqers_move(player.get_move(env, available_moves=env.get_legal_moves()))]
    best = max(value for _, value in row["values"])
    chosen = next((value for judged, value in row["values"] if same_move(judged, move)), None)
    return {"id": row["id"], "ply": row["ply"], "move": move, "depth": player.last_depth,
            "loss": (best - chosen) if chosen is not None else None}


def agent_losses(judged: Sequence[dict[str, Any]], games: Sequence[dict[str, Any]], spec: str,
                 workers: int = 1) -> list[dict[str, Any]]:
    """For each judged decision {"id", "ply", "move", "depth", "loss"}: the move the agent built from `spec`
    chooses there, the depth it finished, and the judge's best value minus the value of that move."""
    by_id = {game.get("id"): game for game in games}
    jobs = [(row, {**by_id[row["id"]], "moves": by_id[row["id"]]["moves"][:row["ply"]]}, spec)
            for row in judged if row["values"]]
    return [r for r in _run(_agent_job, jobs, workers) if r["loss"] is not None]


def played_losses(judged: Sequence[dict[str, Any]], games: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same for the moves that were played in the recorded games."""
    from search.opening_book import same_move

    by_id = {game.get("id"): game for game in games}
    out = []
    for row in judged:
        if not row["values"]:
            continue
        move = [int(x) for x in by_id[row["id"]]["moves"][row["ply"]]["move"]]
        chosen = next((value for judged_move, value in row["values"] if same_move(judged_move, move)), None)
        if chosen is not None:
            out.append({"id": row["id"], "ply": row["ply"], "move": move, "depth": 0,
                        "loss": max(value for _, value in row["values"]) - chosen})
    return out


# ── A deeper judge where the agents differ ─────────────────────────────────────────────────────────────────────


def disagreements(table: dict[str, Sequence[dict[str, Any]]]) -> dict[tuple, list[list[int]]]:
    """{(id, ply): the distinct moves chosen there}, for the decisions at which the agents of `table` (name ->
    rows of agent_losses) did not all choose the same move. Only there can a judge tell them apart, so that is
    where a longer judgement is worth its time."""
    from search.opening_book import same_move

    chosen: dict[tuple, list[list[int]]] = {}
    for rows in table.values():
        for row in rows:
            moves = chosen.setdefault((row["id"], row["ply"]), [])
            if not any(same_move(row["move"], known) for known in moves):
                moves.append(row["move"])
    return {key: moves for key, moves in chosen.items() if len(moves) > 1}


def _judge_moves_job(job: tuple) -> dict[str, Any]:
    key, moves, game_prefix, seconds, depth = job
    from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move
    from search.opening_book import same_move

    env = replay(game_prefix)
    finder = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=seconds * 1000.0)
    best = [int(x) for x in canonicalize_cheqqers_move(finder.get_move(env, available_moves=env.get_legal_moves()))]
    subset = list(moves) + ([] if any(same_move(best, m) for m in moves) else [best])
    judge = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=seconds * 1000.0 * len(subset))
    finished, values = judge.root_values(env, moves=subset)
    return {"id": key[0], "ply": key[1], "judge_depth": finished, "judge_seconds": seconds,
            "finder_depth": finder.last_depth, "finder_move": best,
            "values": [[[int(x) for x in move], value] for move, value in values.items()]}


def judge_moves(differing: dict[tuple, list[list[int]]], games: Sequence[dict[str, Any]], seconds: float = 120.0,
                depth: int = 14, workers: int = 1) -> list[dict[str, Any]]:
    """For each decision of `differing`: the default search first looks for its own best move for `seconds`,
    then values that move and the agents' moves with a full window, all at one depth (`seconds` per move). Rows
    like judge()'s, with "values" for those moves only, plus the finder's move and depth."""
    by_id = {game.get("id"): game for game in games}
    jobs = [(key, moves, {**by_id[key[0]], "moves": by_id[key[0]]["moves"][:key[1]]}, seconds, depth)
            for key, moves in differing.items()]
    return _run(_judge_moves_job, jobs, workers)


def losses_under(verdicts: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """An agent's rows (of agent_losses) with their loss taken from `verdicts` instead, for the decisions the
    verdicts cover."""
    from search.opening_book import same_move

    by_key = {(v["id"], v["ply"]): v for v in verdicts}
    out = []
    for row in rows:
        verdict = by_key.get((row["id"], row["ply"]))
        if verdict is None or not verdict["values"]:
            continue
        chosen = next((value for move, value in verdict["values"] if same_move(move, row["move"])), None)
        if chosen is not None:
            out.append({**row, "loss": max(value for _, value in verdict["values"]) - chosen})
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """{"decisions", "mean_loss", "half_man_or_more", "two_men_or_more", "mean_depth"}."""
    losses = [row["loss"] for row in rows]
    count = len(losses)
    return {"decisions": count, "mean_loss": sum(losses) / count if count else 0.0,
            "half_man_or_more": sum(x >= MAN / 2 for x in losses),
            "two_men_or_more": sum(x >= 2 * MAN for x in losses),
            "mean_depth": sum(row["depth"] for row in rows) / count if count else 0.0}


# ── Command line ───────────────────────────────────────────────────────────────────────────────────────────────


def _team_filter(team: Optional[str]) -> Optional[Callable[[dict[str, Any], int], bool]]:
    if team is None:
        return None
    return lambda game, ply: game["players"][game["moves"][ply]["by"]]["name"] == team


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge", help="value every legal move of the recorded decisions with a long search")
    j.add_argument("games")
    j.add_argument("--out", required=True)
    j.add_argument("--team", default=None, help="only this team's decisions")
    j.add_argument("--seconds", type=float, default=60.0)
    j.add_argument("--depth", type=int, default=14)
    j.add_argument("--workers", type=int, default=6)
    c = sub.add_parser("compare", help="what each agent's choices give away against the judge")
    c.add_argument("judged")
    c.add_argument("games")
    c.add_argument("--agent", action="append", default=[], help="agent spec; repeat for several")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--out", default=None, help="write every agent's decisions here, one JSON line each")
    c.add_argument("--rejudge-seconds", type=float, default=0.0,
                   help="judge the decisions where the agents differ again, this long per move")
    c.add_argument("--rejudge-out", default=None, help="write those verdicts here")
    args = parser.parse_args(argv)
    games = [g for g in load(args.games) if g.get("reason", "normal") == "normal"]

    if args.command == "judge":
        rows = judge(decisions(games, _team_filter(args.team)), args.seconds, args.depth, args.workers, args.out)
        depths = sorted(row["judge_depth"] for row in rows)
        print(json.dumps({"decisions": len(rows), "judge_depth_median": depths[len(depths) // 2] if depths else 0}))
        return 0

    judged = [json.loads(line) for line in Path(args.judged).read_text().splitlines() if line.strip()]
    table = {"as played": played_losses(judged, games)}
    for spec in args.agent:
        table[spec.split(":", 1)[0]] = agent_losses(judged, games, spec, args.workers)
    if args.out:
        with open(args.out, "w") as out:
            for name, rows in table.items():
                for row in rows:
                    out.write(json.dumps({"agent": name, **row}) + "\n")
    for name, rows in table.items():
        s = summarize(rows)
        print(f"{name:>12}: {s['decisions']} decisions, mean loss {s['mean_loss']:.2f}, half a man or more "
              f"{s['half_man_or_more']}, two men or more {s['two_men_or_more']}, mean depth {s['mean_depth']:.2f}")
    if args.rejudge_seconds > 0:
        agents = {name: rows for name, rows in table.items() if name != "as played"}
        differing = disagreements(agents)
        verdicts = judge_moves(differing, games, args.rejudge_seconds, workers=args.workers)
        if args.rejudge_out:
            Path(args.rejudge_out).write_text("".join(json.dumps(v) + "\n" for v in verdicts))
        depths = sorted(v["judge_depth"] for v in verdicts)
        print(f"where the agents differ: {len(differing)} decisions, judged again at {args.rejudge_seconds:.0f} s "
              f"a move, median depth {depths[len(depths) // 2] if depths else 0}")
        for name, rows in agents.items():
            s = summarize(losses_under(verdicts, rows))
            print(f"{name:>12}: {s['decisions']} decisions, mean loss {s['mean_loss']:.2f}, half a man or more "
                  f"{s['half_man_or_more']}, two men or more {s['two_men_or_more']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
