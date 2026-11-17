"""Exact solve of the compact model: the reachable states by potential level, then the backward solve over them.

compact.potential() rises on every move, so the game graph has no repeats and a solve can walk the levels once,
children before parents. reach() collects the reachable non-terminal canonical states level by level and checks that
the potential rises on every move, which is the proof. solve() then walks the levels downwards and, for every state
and each draw weight in W, stores the outcome distribution and the expected remaining plies under the stored policy
(choose()). A long stage checkpoints: one pickle per stage in a directory, stamped with rules_hash() and the
configuration, written whole and moved into place, deleted when the stage completes and kept when it fails or is
stopped early, so that a rerun resumes. The command line, `python3 -m tiqtaqtoe.solve`, keeps each checkpoint until
the result it protects is written and takes the stamp once, when it starts.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from tiqtaqtoe import compact
from tiqtaqtoe.engine import CONTINUE, DRAW, O_WINS, TIE, X_WINS

DATA = Path(__file__).resolve().parents[1] / "results" / "data"  # generated bulk, gitignored
RESULTS = DATA.parent  # the committed summaries, results/<configuration>.json


def rules_hash():
    """The first 16 hex digits of the sha256 of compact.py and solve.py as they are on disk: a checkpoint or table made
    by other rules or solver code carries another hash, so it is never reused."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("compact.py", "solve.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


# --- checkpoints ------------------------------------------------------------------------------------------------


def _stamp(cfg):
    """What ties a checkpoint or a saved count to the rules code and the configuration that made it: (rules_hash(),
    tuple(cfg)). The settings are a plain tuple, equal to cfg, so the stamp itself names no project class. A file is
    unpickled whole before its stamp is compared, though. reach's checkpoint holds only plain data, so a stale one
    always loads and is ignored; the solve checkpoint holds Entry objects and reach.pkl a Reach, so one written when
    those classes had another shape raises at load instead of being ignored."""
    return (rules_hash(), tuple(cfg))


def _checkpoint_path(ckpt_dir, cfg, stage):
    """Where a stage keeps its checkpoint; None, meaning checkpointing is off, without a directory."""
    return None if ckpt_dir is None else Path(ckpt_dir) / f"{cfg.name}-{stage}.pkl"


def _tmp(path):
    """The file a save writes before moving it onto path."""
    return path.with_name(path.name + ".tmp")


def _save_checkpoint(path, payload):
    """Pickle payload to path atomically: write a temporary file beside it, then os.replace it, so a kill mid-write
    leaves the previous checkpoint whole. Does nothing when path is None."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_tmp(path), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(_tmp(path), path)


def _load_checkpoint(path, stamp):
    """The payload saved at path, or None if there is none or its "stamp" is another: a stale checkpoint is ignored and
    left in place for the next save to replace (but see _stamp: the file is unpickled first)."""
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            saved = pickle.load(f)
    except FileNotFoundError:
        return None
    return saved if saved["stamp"] == stamp else None


def _delete_checkpoint(path):
    """Remove the checkpoint and whatever a killed save left beside it."""
    if path is not None:
        path.unlink(missing_ok=True)
        _tmp(path).unlink(missing_ok=True)


# --- reachable states by potential level ------------------------------------------------------------------------


class RepeatError(RuntimeError):
    """A move whose child is not above its parent in potential: the game graph may repeat a state."""


class Reach(NamedTuple):
    """levels: potential -> sorted canonical keys of the reachable non-terminal states, in ascending potential.
    complete: False when the cap stopped the search, which makes the levels a lower bound. seconds: wall time of the
    call (a resumed call counts only its own part)."""

    levels: dict
    complete: bool
    seconds: float


def _expand(model, cfg, key, p, levels):
    """File the non-terminal children of the state with this key, at potential p, under their own potentials in
    levels (sets of keys). RepeatError if a child's potential is not above p."""
    potential, canonical = model.potential, model.canonical
    s = model.decode(key)
    for move in model.moves(s, cfg):
        for _, child, winner in model.apply(s, move, cfg):
            if winner != CONTINUE:
                continue
            q = potential(child)
            if q <= p:
                raise RepeatError(f"potential does not rise: state {s!r} (key {key}, potential {p}), move {move!r} "
                                  f"leads to a state of potential {q}, so the game graph may repeat")
            bucket = levels.get(q)
            if bucket is None:
                bucket = levels[q] = set()
            bucket.add(canonical(child)[0])


def _expand_level(model, cfg, p, levels, cap):
    """Expand every key of level p in ascending order, turning the level into a sorted list. False if the cap stopped
    it first, checked before each key."""
    keys = levels[p] = sorted(levels[p])
    for key in keys:
        if cap is not None and sum(map(len, levels.values())) >= cap:
            return False
        _expand(model, cfg, key, p, levels)
    return True


def _as_lists(levels):
    """levels by ascending potential, each as a sorted list."""
    return {p: keys if isinstance(keys, list) else sorted(keys) for p, keys in sorted(levels.items())}


def reach(cfg, cap=None, ckpt_dir=None, model=compact, on_level=None, keep=False, stamp=None):
    """The reachable non-terminal canonical states of cfg by potential level. The search starts from the empty board
    with X to move (the colour swap in the key makes that cover O-first rounds too) and takes the levels in ascending
    order; since every move raises the potential, a level is complete when its turn comes, and a move that does not
    raises RepeatError. A key is expanded by decoding it and applying every move.

    After each level it expands, the call saves the levels so far and the next level to expand (with ckpt_dir), then
    calls on_level(potential). A rerun resumes from a checkpoint of the same rules code and cfg and ignores any other.
    The search stops, with complete False, once cap keys are held, expanded or not. The checkpoint is deleted only
    when the search completes: a cap stop or an exception keeps it, so a rerun with a larger cap, or none, resumes
    instead of recounting. keep keeps it on completion too, for a caller that deletes it once it has saved the count
    (the command line does), so that a kill in between does not lose the count. stamp is what the checkpoint carries,
    by default _stamp(cfg) taken here; a long run passes the one it took at its start, so that an edit of the code on
    disk meanwhile does not relabel the work of the code that is running."""
    started = time.perf_counter()
    stamp = _stamp(cfg) if stamp is None else stamp
    path = _checkpoint_path(ckpt_dir, cfg, "reach")
    saved = _load_checkpoint(path, stamp)
    if saved is None:
        start = model.initial(1)
        cursor = model.potential(start)
        levels = {cursor: {model.canonical(start)[0]}}
    else:
        levels, cursor = saved["levels"], saved["next"]
    # levels: every key found so far by potential; a level already expanded is a sorted list, a level to come a set
    complete = True
    while cursor is not None:
        p = cursor
        if not _expand_level(model, cfg, p, levels, cap):
            complete = False
            break
        cursor = min((q for q in levels if q > p), default=None)
        _save_checkpoint(path, {"stamp": stamp, "levels": levels, "next": cursor})
        if on_level is not None:
            on_level(p)
    if complete and not keep:
        _delete_checkpoint(path)
    return Reach(_as_lists(levels), complete, time.perf_counter() - started)


# --- the backward solve -----------------------------------------------------------------------------------------

W = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))  # draw weights; W[4 - i] = 1 - W[i]
ZERO, ONE = Fraction(0), Fraction(1)


class Entry(NamedTuple):
    """What a state is worth at one draw weight: under the stored policy, the probabilities of an X win, an O win, a
    tie and a draw, and the expected number of plies left to the result."""

    x: Fraction
    o: Fraction
    tie: Fraction
    draw: Fraction
    plies: Fraction


# A finished game by winner code: that result for sure, no plies left.
TERMINAL = {X_WINS: Entry(ONE, ZERO, ZERO, ZERO, ZERO), O_WINS: Entry(ZERO, ONE, ZERO, ZERO, ZERO),
            TIE: Entry(ZERO, ZERO, ONE, ZERO, ZERO), DRAW: Entry(ZERO, ZERO, ZERO, ONE, ZERO)}

# ponytail: one process-wide dict, never emptied, so a process that solves many configurations keeps every entry it
# made; scope it to one solve() if memory ever shows it. Interning makes the many equal entries of a table one object
# each (few distinct ones in millions of rows), which Q2 needs to fit in memory.
_INTERN = {}


def value(e, w):
    """X's payoff under the entry at draw weight w: an X win 1, an O win 0, a tie or a draw w."""
    return e.x + w * (e.tie + e.draw)


def flip(e):
    """The entry of the colour-swapped game: X and O exchanged."""
    return e._replace(x=e.o, o=e.x)


def entry_at(table, s, wi):
    """The entry of state s at W[wi]. The table holds one state per class, in the orientation canonical() picks. If that
    is the colour swap of s, s is the stored game with X and O exchanged, so its entry is the flip of the stored one at
    draw weight 1 - w, which is W[4 - wi]. KeyError if the class is not in the table."""
    key, swapped = compact.canonical(s)
    return flip(table[key][4 - wi]) if swapped else table[key][wi]


def move_entry(table, s, move, cfg, wi):
    """The entry of playing `move` in s at W[wi]: the entries of its outcomes (TERMINAL for a finished game), weighted
    by their probabilities, one ply longer. The outcomes are summed one by one, never collected by state: two outcomes
    can share a canonical state, and each of them counts."""
    parts = [(p, TERMINAL[win] if win != CONTINUE else entry_at(table, child, wi))
             for p, child, win in compact.apply(s, move, cfg)]
    x, o, tie, draw, plies = (sum(p * e[i] for p, e in parts) for i in range(5))
    return Entry(x, o, tie, draw, 1 + plies)


def choose(mover, cands, w):
    """The stored policy: (move, entry) out of cands, [(move, entry)] in move order, for the mover (1 X, 2 O) at draw
    weight w. Keep the moves with the mover's best value (X the highest, O the lowest). Of those prefer, in order,
    (1) the fewest expected plies if that value is better for the mover than w, the most if it is worse, the fewest if
    it equals w; (2) the highest probability of the mover's own outright win; (3) the first in move order."""
    sign = 1 if mover == 1 else -1  # the mover maximises sign * value
    best = max(sign * value(e, w) for _, e in cands)
    top = [c for c in cands if sign * value(c[1], w) == best]
    stretch = best < sign * w  # worse than w for the mover: make the game last
    # fields 0 and 1 of an entry are the X and O wins, so [mover - 1] is the mover's own
    return min(top, key=lambda c: (-c[1].plies if stretch else c[1].plies, -c[1][mover - 1]))


def _solve_state(table, cfg, key):
    """The five entries of the state with this key, one per draw weight, from the table of the levels above it."""
    s = compact.decode(key)
    moves = compact.moves(s, cfg)
    if not moves:
        raise RuntimeError(f"state {s!r} (key {key}) is not finished but has no legal move")
    row = []
    for wi, w in enumerate(W):
        _, e = choose(s.turn, [(m, move_entry(table, s, m, cfg, wi)) for m in moves], w)
        row.append(_INTERN.setdefault(e, e))
    return tuple(row)


def solve(cfg, reach, ckpt_dir=None, on_level=None, keep=False, stamp=None):
    """{key: five entries, in W order} for every key of reach, a complete count of cfg (ValueError if the cap stopped
    it). The levels go in descending potential, so every child is in the table before its parents. As in reach(),
    after each level the call saves the table so far and the next level to solve (with ckpt_dir), then calls
    on_level(potential); a rerun resumes from a checkpoint with the same stamp and ignores one with another (see
    _stamp), and the checkpoint is deleted when the solve completes, unless keep: a caller that has results of its own
    to write keeps it until they are written, so that a kill in between does not lose the solve. stamp is as in
    reach(): by default _stamp(cfg) taken here, or the one a long run took at its start."""
    if not reach.complete:
        held = sum(map(len, reach.levels.values()))
        raise ValueError(f"the count is incomplete (its cap stopped it at {held} states): count to the end first")
    stamp = _stamp(cfg) if stamp is None else stamp
    path = _checkpoint_path(ckpt_dir, cfg, "solve")
    saved = _load_checkpoint(path, stamp)
    table, cursor = ({}, max(reach.levels, default=None)) if saved is None else (saved["table"], saved["next"])
    while cursor is not None:
        p = cursor
        for key in reach.levels[p]:
            table[key] = _solve_state(table, cfg, key)
        cursor = max((q for q in reach.levels if q < p), default=None)
        _save_checkpoint(path, {"stamp": stamp, "table": table, "next": cursor})
        if on_level is not None:
            on_level(p)
    if not keep:
        _delete_checkpoint(path)
    return table


def write_tables(cfg, table, out_dir=DATA, rules=None):
    """One pickle per draw weight, each moved into place whole: out_dir/<name>/w=<numerator>-<denominator>.pkl holding
    {"config", "w", "rules_hash", "entries": {key: Entry}}. The paths, in W order. All five carry the same hash:
    `rules`, the hash of the code that made the table, by default rules_hash() read once, here."""
    rules = rules_hash() if rules is None else rules
    paths = []
    for wi, w in enumerate(W):
        path = Path(out_dir) / cfg.name / f"w={w.numerator}-{w.denominator}.pkl"
        entries = {key: row[wi] for key, row in table.items()}
        _save_checkpoint(path, {"config": cfg, "w": w, "rules_hash": rules, "entries": entries})
        paths.append(path)
    return paths


# --- the command line -------------------------------------------------------------------------------------------


def _show(e):
    """An entry as a string of its five numbers, e.g. (0, 0, 0, 1, 9)."""
    return "(" + ", ".join(map(str, e)) + ")"


def _progress(stage):
    """An on_level callback that reports each finished level on stderr."""
    return lambda p: print(f"{stage}: level {p} done", file=sys.stderr, flush=True)


def _count(cfg, cap, stamp, ckpt_dir):
    """The count of cfg: the one saved in results/data/<name>/reach.pkl if its stamp matches, else a checkpointed run
    of reach(), saved there when it completes (a capped one leaves only reach's own checkpoint). reach's checkpoint is
    deleted only once the count is saved, or found saved: a kill before then loses nothing."""
    path = DATA / cfg.name / "reach.pkl"
    saved = _load_checkpoint(path, stamp)
    if saved is None:
        counted = reach(cfg, cap=cap, ckpt_dir=ckpt_dir, on_level=_progress("count"), keep=True, stamp=stamp)
        if not counted.complete:
            return counted
        _save_checkpoint(path, {"stamp": stamp, "reach": counted})
    else:
        counted = saved["reach"]
    _delete_checkpoint(_checkpoint_path(ckpt_dir, cfg, "reach"))
    return counted


def _summary(cfg, counted, table, seconds, rules):
    """The dict written to results/<name>.json, for the code with hash `rules`; the X-first root entry per w only when
    the states were solved."""
    settings = {"quantumness": cfg.quantumness, "decoherence": cfg.decoherence, "manual_measurement": cfg.manual}
    root = {} if table is None else {"root": {str(w): _show(entry_at(table, compact.initial(1), wi))
                                              for wi, w in enumerate(W)}}
    return {"config": settings, "rules_hash": rules, "states": sum(map(len, counted.levels.values())),
            "levels": {str(p): len(keys) for p, keys in counted.levels.items()}, "complete": counted.complete,
            "seconds": round(seconds, 2), **root}


def main(argv=None):
    """Count the reachable states of a configuration and, unless --count-only, solve them at the five draw weights and
    write the tables under results/data/<name>/. Writes and prints the summary results/<name>.json, whose seconds are
    the count's own (as recorded when it ran) plus this run's solve. A capped count (--cap) writes its summary with
    complete false and solves nothing; the same command without the cap resumes it. The stamp, and so the hash in
    every file written, is taken once, at the start; each checkpoint is deleted only after the result it protects is
    written: reach's after reach.pkl, the solve's after the tables and the summary."""
    parser = argparse.ArgumentParser(prog="python3 -m tiqtaqtoe.solve", description=main.__doc__)
    parser.add_argument("--quantumness", type=int, required=True)
    parser.add_argument("--decoherence", type=int, required=True)
    parser.add_argument("--manual-off", action="store_true", help="manual measurement off (default: on)")
    parser.add_argument("--cap", type=int, help="stop counting once this many states are held")
    parser.add_argument("--count-only", action="store_true", help="count the states, do not solve them")
    args = parser.parse_args(argv)
    try:
        cfg = compact.Config(args.quantumness, args.decoherence, not args.manual_off)
    except ValueError as err:
        parser.error(str(err))
    stamp, ckpt_dir = _stamp(cfg), DATA / "checkpoints"
    counted = _count(cfg, args.cap, stamp, ckpt_dir)
    table, seconds = None, counted.seconds
    if not counted.complete:
        print("the count stopped at its cap: run again with a larger --cap, or none, to resume it", file=sys.stderr)
    elif not args.count_only:
        started = time.perf_counter()
        table = solve(cfg, counted, ckpt_dir, _progress("solve"), keep=True, stamp=stamp)
        write_tables(cfg, table, DATA, rules=stamp[0])
        seconds += time.perf_counter() - started
    text = json.dumps(_summary(cfg, counted, table, seconds, stamp[0]), indent=2) + "\n"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{cfg.name}.json").write_text(text)
    if table is not None:
        _delete_checkpoint(_checkpoint_path(ckpt_dir, cfg, "solve"))  # the summary is the last result written
    print(text, end="")


if __name__ == "__main__":
    # Run main() from the module's normal import, not from this __main__ copy: the classes defined here would pickle as
    # __main__.Entry and __main__.Reach, which no other process (the teacher) can load.
    from tiqtaqtoe.solve import main as _main

    _main()
