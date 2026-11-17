"""book_check.py: a readable check of a network opening book, and how far a network agrees with it.

    uv run python scripts/cheqqers_rl/tools/book_check.py <book.json> --models a.pt [b.pt ...] --ms 500 \
        [--arena games.jsonl --team NAME --arena-plies 24] --out-md report.md --out-jsonl agreement.jsonl

For every book position (found again by walking the tree from the real start: our moves from the book, every
opponent reply and measurement outcome; and, with --arena, along the recorded games) it writes the line that leads
there, the book's move, and what each model plays with a search of --ms. The report groups positions by colour and
our decision number and says per model how often it plays the book's move. One JSON line per model goes to
--out-jsonl (agreement per colour and decision), so successive checkpoints give a curve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from env.cheqqers_official import OfficialCheqqersEnv, PieceColor  # noqa: E402
from search.opening_book import OpeningBook, _children, _unique, same_move, start_position  # noqa: E402

SQUARES = """Squares (black starts on 0-5 at the top, white on 12-17 at the bottom; white moves first):
 .  0  .  1  .  2
 3  .  4  .  5  .
 .  6  .  7  .  8
 9  . 10  . 11  .
 . 12  . 13  . 14
15  . 16  . 17  ."""


def fmt(move) -> str:
    m = list(move)
    if len(m) == 2:
        return f"{m[0]}-{m[1]}"
    return f"{m[0]}/{m[1]}/{m[2]}"


def tree_positions(book: OpeningBook, own_moves: int) -> dict[str, dict]:
    """{key: {"env", "side", "decision", "line"}} for the book's tree: our moves from the book, every opponent
    reply and outcome (the build's --opp-all 4)."""
    found: dict[str, dict] = {}
    for side in (PieceColor.WHITE, PieceColor.BLACK):
        frontier = [(start_position(), [])]
        for decision in range(1, own_moves + 1):
            ours, waiting = [], frontier
            for _ in range(12):
                ours += [(e, l) for e, l in waiting if e.turn == side]
                theirs = {}
                for e, l in waiting:
                    if e.turn != side:
                        theirs.setdefault(OpeningBook.key(e), (e, l))
                if not theirs:
                    break
                waiting = []
                for e, l in theirs.values():
                    for mv in e.get_legal_moves():
                        for p, c in e.get_move_outcomes(list(mv)):
                            if p > 1e-9 and not c.done:
                                waiting.append((c, l + [fmt(mv) + (f"({p:.2f})" if p < 0.999 else "")]))
            seen = {}
            for e, l in ours:
                seen.setdefault(OpeningBook.key(e), (e, l))
            frontier = []
            for k, (e, l) in seen.items():
                mv = book.moves.get(k)
                if mv is not None and k not in found:
                    found[k] = {"env": e, "side": side.name.lower(), "decision": decision, "line": l, "src": "tree"}
                for c_mv in ([mv] if mv is not None else e.get_legal_moves()):
                    for p, c in e.get_move_outcomes(list(c_mv)):
                        if p > 1e-9 and not c.done:
                            frontier.append((c, l + ["*" + fmt(c_mv) + (f"({p:.2f})" if p < 0.999 else "")]))
    return found


def arena_lines(book: OpeningBook, records: Path, team: str, max_ply: int, found: dict[str, dict]) -> None:
    for raw in records.read_text().splitlines():
        if not raw.strip():
            continue
        g = json.loads(raw)
        names = {s: (g.get("players", {}).get(s) or {}).get("name") for s in ("white", "black")}
        if team not in names.values():
            continue
        ours = "white" if names["white"] == team else "black"
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        line, decision = [], 0
        for ply, m in enumerate(g["moves"][:max_ply]):
            if m["by"] == ours:
                decision += 1
                k = OpeningBook.key(env)
                if k in book.moves and k not in found:
                    import copy
                    found[k] = {"env": copy.deepcopy(env), "side": ours, "decision": decision, "line": list(line),
                                "src": f"arena {g.get('id')}"}
            try:
                env.apply_move(list(m["move"]), forced_measurements=list(m["outcomes"]))
            except Exception:
                break
            line.append(("*" if m["by"] == ours else "") + fmt(m["move"]) + ("!" if m["outcomes"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book", type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ms", type=float, default=500.0)
    ap.add_argument("--own-moves", type=int, default=4)
    ap.add_argument("--arena", type=Path, default=None)
    ap.add_argument("--team", default="")
    ap.add_argument("--arena-plies", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--tree-only", action="store_true", help="skip the arena positions (faster)")
    args = ap.parse_args()

    from search.gpu_mcts_cheqqers import get_mcts_player
    book = OpeningBook.from_json(args.book)
    found = tree_positions(book, args.own_moves)
    if args.arena and not args.tree_only:
        arena_lines(book, args.arena, args.team, args.arena_plies, found)
    keys = sorted(found, key=lambda k: (found[k]["side"] != "white", found[k]["src"] != "tree", found[k]["decision"]))
    plays: dict[str, dict[str, list]] = {}
    summary = []
    for model in args.models:
        player = get_mcts_player(model, device=args.device)
        t0 = time.time()
        agree = defaultdict(lambda: [0, 0])
        plays[model] = {}
        evals = []
        for k in keys:
            e = found[k]["env"]
            r = player.search(e, e.get_legal_moves(), time_ms=args.ms)
            evals.append(int(r.evals))
            ok = same_move(r.move, book.moves[k])
            plays[model][k] = [fmt(r.move), ok]
            grp = (found[k]["side"], found[k]["src"] == "tree" and f"move {found[k]['decision']}" or "arena lines")
            agree[grp][0 if ok else 1] += 1
        rows = {f"{s} {d}": {"agree": a, "differ": b, "share": round(a / max(1, a + b), 3)}
                for (s, d), (a, b) in sorted(agree.items())}
        tot_a = sum(v["agree"] for v in rows.values()); tot = sum(v["agree"] + v["differ"] for v in rows.values())
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "model": str(model), "ms": args.ms, "positions": tot,
               "agree_share": round(tot_a / max(1, tot), 3), "by_group": rows, "seconds": round(time.time() - t0),
               "evals_median": sorted(evals)[len(evals) // 2] if evals else 0}
        # a search that barely ran returns its first candidate: the agreement would be meaningless (P3)
        rec["valid"] = rec["evals_median"] >= 1000
        if not rec["valid"]:
            print(f"WARNING: median {rec['evals_median']} network evaluations a search: too few, not a measurement",
                  file=sys.stderr)
        summary.append(rec)
        print(json.dumps(rec), flush=True)
        if args.out_jsonl:
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")

    names = [Path(m).name for m in args.models]
    md = [f"# Opening book check: `{args.book.name}` ({len(book)} positions)", "",
          f"Book settings: `{json.dumps(book.settings)}`", "", "```", SQUARES, "```", "",
          "Moves: `a-b` slide or jump, `a/b/c` a three-square move (split or merge). In a line, `*` marks our move, "
          "`(p)` the probability of that measurement outcome, `!` a recorded move that measured squares.", "",
          f"## Agreement with the book (search of {args.ms:g} ms)", "",
          "| model | positions | plays the book's move |", "|---|---|---|"]
    for rec in summary:
        md.append(f"| {Path(rec['model']).name} | {rec['positions']} | **{rec['agree_share']:.0%}**"
                  + ("" if rec["valid"] else f" (INVALID: {rec['evals_median']} evaluations a search)") + " |")
    md += ["", "| group | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for grp in summary[0]["by_group"]:
        md.append(f"| {grp} | " + " | ".join(f"{r['by_group'].get(grp, {}).get('share', 0):.0%} "
                                              f"({r['by_group'].get(grp, {}).get('agree', 0)}/"
                                              f"{r['by_group'].get(grp, {}).get('agree', 0) + r['by_group'].get(grp, {}).get('differ', 0)})"
                                              for r in summary) + " |")
    current = None
    for k in keys:
        f = found[k]
        grp = f"{f['side']}, " + (f"our move {f['decision']}" if f["src"] == "tree" else "arena lines")
        if grp != current:
            md += ["", f"## {grp}", "", "| line from the start | book | " + " | ".join(names) + " | source |",
                   "|---|---|" + "---|" * len(names) + "---|"]
            current = grp
        cells = [f"{mv}" + ("" if ok else " ✗") for mv, ok in (plays[m][k] for m in args.models)]
        md.append(f"| {' '.join(f['line']) or '(start)'} | **{fmt(book.moves[k])}** | " + " | ".join(cells)
                  + f" | {f['src']} |")
    args.out_md.write_text("\n".join(md) + "\n")
    print(f"report: {args.out_md} ({len(keys)} positions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
