"""solve.py (plan Tasks 4 and 5): the reachable states by potential level, the checkpoints that make a killed count or
solve resumable, and the backward solve with its tables."""

import contextlib
import hashlib
import io
import itertools
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tiqtaqtoe import compact as C
from tiqtaqtoe import engine as E
from tiqtaqtoe import solve as S

Q0 = C.Config(0, 0, True)


def walk(steps):
    """A model whose states are their own potentials and whose one move goes from each state to steps[state]."""
    return SimpleNamespace(initial=lambda first=1: 0, moves=lambda s, cfg: ["step"],
                           apply=lambda s, m, cfg: [(Fraction(1), steps[s], E.CONTINUE)],
                           canonical=lambda s: (s, False), decode=lambda k: k, potential=lambda s: s)


class Reach(unittest.TestCase):
    def test_q0_reaches_627_non_terminal_classes(self):
        r = S.reach(Q0)
        self.assertTrue(r.complete)
        self.assertEqual(sum(map(len, r.levels.values())), 627)

    def test_a_repeat_raises(self):
        loop = SimpleNamespace(initial=lambda first=1: 0, moves=lambda s, cfg: ["x"],
                               apply=lambda s, m, cfg: [(Fraction(1), 1 - s, E.CONTINUE)],
                               canonical=lambda s: (s, False), decode=lambda k: k, potential=lambda s: 0)
        with self.assertRaises(S.RepeatError):
            S.reach(Q0, model=loop)

    def test_interrupted_reach_resumes_to_the_same_levels(self):
        full = S.reach(Q0).levels
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == 9:
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.reach(Q0, ckpt_dir=Path(d), on_level=stop)
            self.assertEqual(S.reach(Q0, ckpt_dir=Path(d)).levels, full)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_levels_ascend_from_the_empty_board(self):
        r = S.reach(Q0)
        start = C.initial(1)
        self.assertEqual(r.levels[C.potential(start)], [C.canonical(start)[0]])
        self.assertEqual(list(r.levels), sorted(r.levels))
        for p, keys in r.levels.items():
            self.assertEqual(keys, sorted(set(keys)))
            self.assertEqual({C.potential(C.decode(k)) for k in keys}, {p})
        self.assertIsInstance(r.seconds, float)

    def test_a_run_killed_after_its_last_level_resumes_without_expanding_anything(self):
        full = S.reach(Q0).levels
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == max(full):
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.reach(Q0, ckpt_dir=Path(d), on_level=stop)
            fails_if_asked_to_expand = walk({})
            self.assertEqual(S.reach(Q0, ckpt_dir=Path(d), model=fails_if_asked_to_expand).levels, full)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_repeat_names_the_state_the_move_and_both_potentials(self):
        with self.assertRaises(S.RepeatError) as caught:
            S.reach(Q0, model=walk({0: 5, 5: 2}))
        for part in ("state 5", "move 'step'", "potential 5", "potential 2"):
            self.assertIn(part, str(caught.exception))

    def test_a_failed_run_keeps_its_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(S.RepeatError):
                S.reach(Q0, ckpt_dir=Path(d), model=walk({0: 5, 5: 2}))
            self.assertEqual(len(list(Path(d).iterdir())), 1)

    def test_keep_leaves_the_finished_checkpoint_for_the_caller_to_delete(self):
        full = S.reach(Q0).levels
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.reach(Q0, ckpt_dir=Path(d), keep=True).levels, full)
            [path] = Path(d).iterdir()
            self.assertIsNone(S._load_checkpoint(path, S._stamp(Q0))["next"])
            fails_if_asked_to_expand = walk({})  # a rerun from the finished checkpoint expands nothing
            self.assertEqual(S.reach(Q0, ckpt_dir=Path(d), model=fails_if_asked_to_expand, keep=True).levels, full)
            S._delete_checkpoint(path)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_given_stamp_is_what_the_checkpoint_carries(self):
        mine = ("f" * 16, tuple(Q0))
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.reach(Q0, ckpt_dir=Path(d), on_level=stop, stamp=mine)
            [path] = Path(d).iterdir()
            self.assertIsNotNone(S._load_checkpoint(path, mine))
            self.assertIsNone(S._load_checkpoint(path, S._stamp(Q0)))

    def test_a_cap_stops_the_search_with_what_was_found(self):
        r = S.reach(Q0, cap=100)
        self.assertFalse(r.complete)
        self.assertTrue(100 <= sum(map(len, r.levels.values())) < 110)  # checked per state, not per level
        self.assertEqual(list(r.levels), sorted(r.levels))
        self.assertTrue(all(keys == sorted(keys) for keys in r.levels.values()))
        self.assertTrue(S.reach(Q0, cap=10_000).complete)

    def test_a_capped_run_keeps_its_checkpoint_and_a_rerun_resumes_from_it(self):
        full = S.reach(Q0).levels
        stamp = (S.rules_hash(), Q0)
        for rerun_cap in (None, 10_000):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "q0-d0-m1-reach.pkl"
                self.assertFalse(S.reach(Q0, cap=100, ckpt_dir=Path(d)).complete)
                self.assertEqual(list(Path(d).iterdir()), [path])
                resume_at = S._load_checkpoint(path, stamp)["next"]
                self.assertGreater(resume_at, min(full))  # some levels were saved, so a rerun is not a recount
                expanded = []
                r = S.reach(Q0, cap=rerun_cap, ckpt_dir=Path(d), on_level=expanded.append)
                self.assertTrue(r.complete)
                self.assertEqual(r.levels, full)
                self.assertEqual(expanded, [p for p in sorted(full) if p >= resume_at])
                self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_larger_cap_resumes_and_keeps_the_checkpoint_again(self):
        full = S.reach(Q0).levels
        stamp = (S.rules_hash(), Q0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q0-d0-m1-reach.pkl"
            S.reach(Q0, cap=100, ckpt_dir=Path(d))
            first = S._load_checkpoint(path, stamp)["next"]
            expanded = []
            self.assertFalse(S.reach(Q0, cap=300, ckpt_dir=Path(d), on_level=expanded.append).complete)
            second = S._load_checkpoint(path, stamp)["next"]
            self.assertGreater(second, first)  # the second call saved the levels it finished
            self.assertEqual(expanded, [p for p in sorted(full) if first <= p < second])
            expanded.clear()
            self.assertEqual(S.reach(Q0, ckpt_dir=Path(d), on_level=expanded.append).levels, full)
            self.assertEqual(expanded, [p for p in sorted(full) if p >= second])
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_on_level_runs_after_each_save_with_the_next_level_recorded(self):
        full = S.reach(Q0).levels
        stamp = (S.rules_hash(), Q0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ck" / "q0-d0-m1-reach.pkl"  # the directory is made by the first save
            seen = []

            def note(p):
                saved = S._load_checkpoint(path, stamp)
                seen.append((p, saved["next"], saved["levels"][p]))
                if saved["next"] is not None:  # a level is complete when its turn comes
                    self.assertEqual(sorted(saved["levels"][saved["next"]]), full[saved["next"]])

            S.reach(Q0, ckpt_dir=path.parent, on_level=note)
            self.assertEqual(list(path.parent.iterdir()), [])
        ps = sorted(full)
        self.assertEqual([(p, nxt) for p, nxt, _ in seen], list(zip(ps, ps[1:] + [None])))
        self.assertTrue(all(keys == full[p] for p, _, keys in seen))


class Checkpoints(unittest.TestCase):
    def test_a_checkpoint_of_other_rules_or_settings_is_ignored_not_deleted(self):
        full = S.reach(Q0).levels
        stamp = (S.rules_hash(), Q0)
        for other in (("0" * 16, Q0), (S.rules_hash(), C.Config(1, 0, True))):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "q0-d0-m1-reach.pkl"
                with open(path, "wb") as f:
                    pickle.dump({"stamp": other, "levels": {0: [1]}, "next": None}, f)  # used, it would end the run
                self.assertIsNone(S._load_checkpoint(path, stamp))
                self.assertTrue(path.exists())
                self.assertEqual(S.reach(Q0, ckpt_dir=Path(d)).levels, full)
                self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_checkpoint_names_no_project_class(self):
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.reach(Q0, ckpt_dir=Path(d), on_level=stop)
            [path] = Path(d).iterdir()
            # so a stale one still unpickles, and is ignored, after a class it would need has changed
            self.assertNotIn(b"tiqtaqtoe", path.read_bytes())

    def dead_save(self, path, payload):
        """A save killed while writing: it leaves a half-written temporary file and raises."""
        def torn(obj, f, protocol=None):
            f.write(b"half a pickle")
            raise OSError("killed")

        with mock.patch.object(S.pickle, "dump", torn), self.assertRaises(OSError):
            S._save_checkpoint(path, payload)

    def test_a_save_killed_midway_leaves_the_previous_checkpoint_whole(self):
        stamp = (S.rules_hash(), Q0)
        old = {"stamp": stamp, "levels": {0: [1]}, "next": 3}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.pkl"
            S._save_checkpoint(path, old)
            self.dead_save(path, {**old, "next": 6})
            self.assertEqual(S._load_checkpoint(path, stamp), old)

    def test_deleting_a_checkpoint_clears_what_a_dead_save_left(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.pkl"
            S._save_checkpoint(path, {"stamp": None})
            self.dead_save(path, {"stamp": None})
            self.assertEqual(len(list(Path(d).iterdir())), 2)  # the checkpoint and the half-written temporary file
            S._delete_checkpoint(path)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_rules_hash_covers_compact_and_solve(self):
        here = Path(S.__file__).parent
        text = (here / "compact.py").read_bytes() + (here / "solve.py").read_bytes()
        self.assertRegex(S.rules_hash(), "^[0-9a-f]{16}$")
        self.assertEqual(S.rules_hash(), hashlib.sha256(text).hexdigest()[:16])

    def test_one_stamp_helper_with_no_project_class_in_it(self):
        self.assertEqual(S._stamp(Q0), (S.rules_hash(), (0, 0, True)))
        self.assertIs(type(S._stamp(Q0)[1]), tuple)


class Solve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reach = S.reach(Q0)
        cls.table = S.solve(Q0, cls.reach)

    def test_q0_is_a_draw_at_every_w(self):
        root = [S.entry_at(self.table, C.initial(), i) for i in range(5)]
        self.assertEqual([S.value(e, w) for e, w in zip(root, S.W)], list(S.W))
        self.assertEqual(root[2], S.Entry(0, 0, 0, 1, 9))

    def test_lookup_maps_colour_swapped_states(self):
        for key in self.table:
            s = C.decode(key)
            for i in range(5):
                self.assertEqual(S.entry_at(self.table, C.recolour(s), i), S.flip(S.entry_at(self.table, s, 4 - i)))

    def test_tie_break_prefers_faster_wins_and_slower_losses(self):
        half = Fraction(1, 2)
        fast, slow = S.Entry(1, 0, 0, 0, 3), S.Entry(1, 0, 0, 0, 5)
        self.assertEqual(S.choose(1, [("a", slow), ("b", fast)], half), ("b", fast))
        self.assertEqual(S.choose(2, [("a", fast), ("b", slow)], half), ("b", slow))
        draw3, draw5 = S.Entry(0, 0, 0, 1, 3), S.Entry(0, 0, 0, 1, 5)
        self.assertEqual(S.choose(1, [("a", draw5), ("b", draw3)], half), ("b", draw3))
        tie, coin = S.Entry(0, 0, 1, 0, 4), S.Entry(half, half, 0, 0, 4)
        self.assertEqual(S.choose(1, [("a", tie), ("b", coin)], half), ("b", coin))
        self.assertEqual(S.choose(1, [("a", tie), ("b", tie)], half), ("a", tie))

    def test_interrupted_solve_resumes_to_the_same_table(self):
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == 12:
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.solve(Q0, self.reach, ckpt_dir=Path(d), on_level=stop)
            self.assertEqual(S.solve(Q0, self.reach, ckpt_dir=Path(d)), self.table)
            self.assertEqual(list(Path(d).iterdir()), [])


# --- beyond the plan's four tests: what the rulings and the self-review ask for ---------------------------------


def all_positions(first):
    """{state: [outcomes of each legal move, as C.apply lists them]} for every position reachable from the empty Q0
    board with `first` to move: the game graph on raw states, with no canonical keys or colour swaps in it."""
    graph, todo = {}, [C.initial(first)]
    while todo:
        s = todo.pop()
        if s not in graph:
            graph[s] = [C.apply(s, m, Q0) for m in C.moves(s, Q0)]
            todo += [child for outs in graph[s] for _, child, win in outs if win == E.CONTINUE]
    return graph


def minimax(graph, w):
    """{state: X's value at draw weight w} by plain recursion: X win 1, O win 0, tie or draw w, X maximises."""
    pay = {E.X_WINS: Fraction(1), E.O_WINS: Fraction(0), E.TIE: w, E.DRAW: w}
    memo = {}

    def go(s):
        if s not in memo:
            options = [sum(p * (go(child) if win == E.CONTINUE else pay[win]) for p, child, win in outs)
                       for outs in graph[s]]
            memo[s] = max(options) if s.turn == 1 else min(options)
        return memo[s]

    for s in graph:
        go(s)
    return memo


class Entries(unittest.TestCase):
    def test_terminal_entries_value_flip_and_the_grid(self):
        self.assertEqual(S.W, tuple(Fraction(i, 4) for i in range(5)))
        self.assertEqual(S.TERMINAL, {E.X_WINS: S.Entry(1, 0, 0, 0, 0), E.O_WINS: S.Entry(0, 1, 0, 0, 0),
                                      E.TIE: S.Entry(0, 0, 1, 0, 0), E.DRAW: S.Entry(0, 0, 0, 1, 0)})
        e = S.Entry(Fraction(1, 4), Fraction(1, 8), Fraction(1, 8), Fraction(1, 2), 5)
        self.assertEqual(S.value(e, Fraction(3, 4)), Fraction(23, 32))  # 1/4 + 3/4 x (1/8 + 1/2)
        self.assertEqual(S.flip(e), S.Entry(Fraction(1, 8), Fraction(1, 4), Fraction(1, 8), Fraction(1, 2), 5))

    def test_choose_keeps_only_the_best_value_before_it_looks_at_length(self):
        half = Fraction(1, 2)
        x_wins, o_wins, draw = S.Entry(1, 0, 0, 0, 9), S.Entry(0, 1, 0, 0, 3), S.Entry(0, 0, 0, 1, 1)
        self.assertEqual(S.choose(1, [("a", draw), ("b", x_wins)], half), ("b", x_wins))
        self.assertEqual(S.choose(2, [("a", draw), ("b", x_wins), ("c", o_wins)], half), ("c", o_wins))
        # a loser stretches the game out whichever side it is, so X losing slowly beats X losing fast
        lose3, lose5 = S.Entry(0, 1, 0, 0, 3), S.Entry(0, 1, 0, 0, 5)
        self.assertEqual(S.choose(1, [("a", lose3), ("b", lose5)], half), ("b", lose5))

    def test_choose_breaks_equal_length_by_the_movers_own_wins(self):
        # at w = 0 the value is x alone, so equal values leave O's own wins free to differ; at w = 1 it is 1 - o, so
        # equal values leave X's own wins free: the mover's outright win then decides, after the length
        tie, o_half, x_half = S.Entry(0, 0, 1, 0, 4), S.Entry(0, Fraction(1, 2), 0, Fraction(1, 2), 4), \
            S.Entry(Fraction(1, 2), 0, Fraction(1, 2), 0, 4)
        self.assertEqual(S.choose(2, [("a", tie), ("b", o_half)], Fraction(0)), ("b", o_half))
        self.assertEqual(S.choose(1, [("a", tie), ("b", x_half)], Fraction(1)), ("b", x_half))

    def test_outcomes_that_share_a_canonical_key_each_count(self):
        cfg = C.Config(1, 0, True)
        [(_, pair, _)] = C.apply(C.initial(), ("q", 0, 8), cfg)  # X's pair on the corners 0 and 8, O to move
        outcomes = C.apply(pair, ("m", 0), cfg)  # measured: X on 0 or on 8, half each
        keys = {C.canonical(child)[0] for _, child, _ in outcomes}
        self.assertEqual((len(outcomes), len(keys)), (2, 1))  # two outcomes, one table row
        child = S.Entry(Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), 0, 6)  # its own flip, so the colour is moot
        table = {keys.pop(): (child,) * 5}
        self.assertEqual(S.move_entry(table, pair, ("m", 0), cfg, 2),
                         S.Entry(Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), 0, 7))

    def test_a_missing_state_is_a_key_error(self):
        with self.assertRaises(KeyError):
            S.entry_at({}, C.initial(), 2)


class Tables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reach = S.reach(Q0)
        cls.table = S.solve(Q0, cls.reach)

    def test_every_value_is_the_minimax_value_of_the_raw_game(self):
        for first in (1, 2):
            graph = all_positions(first)
            for i, w in enumerate(S.W):
                for s, want in minimax(graph, w).items():
                    self.assertEqual(S.value(S.entry_at(self.table, s, i), w), want, (first, w, s))

    def test_every_entry_is_a_distribution_and_the_rows_are_shared(self):
        rows = [e for row in self.table.values() for e in row]
        self.assertEqual({len(row) for row in self.table.values()}, {5})
        for e in rows:
            self.assertEqual(e.x + e.o + e.tie + e.draw, 1)
            self.assertGreaterEqual(e.plies, 1)
            self.assertTrue(all(type(v) is Fraction for v in e))
        self.assertEqual(len({id(e) for e in rows}), len(set(rows)))  # one object per distinct entry

    def test_a_win_in_one_is_one_ply_at_every_w_and_flips_with_the_colours(self):
        s = C.CState((1, 1, 0, 2, 2, 0, 0, 0, 0), (), (), 1, (0,) * 9)  # X on 0 and 1, O on 3 and 4, X to move
        for i in range(5):
            self.assertEqual(S.entry_at(self.table, s, i), S.Entry(1, 0, 0, 0, 1))
            self.assertEqual(S.entry_at(self.table, C.recolour(s), i), S.Entry(0, 1, 0, 0, 1))

    def test_write_tables_makes_one_pickle_per_w(self):
        with tempfile.TemporaryDirectory() as d:
            paths = S.write_tables(Q0, self.table, Path(d))
            self.assertEqual([p.relative_to(d).as_posix() for p in paths],
                             [f"q0-d0-m1/w={n}.pkl" for n in ("0-1", "1-4", "1-2", "3-4", "1-1")])
            for i, (path, w) in enumerate(zip(paths, S.W)):
                self.assertEqual(pickle.loads(path.read_bytes()),
                                 {"config": Q0, "w": w, "rules_hash": S.rules_hash(),
                                  "entries": {key: row[i] for key, row in self.table.items()}})
            self.assertEqual(sorted(p.name for p in (Path(d) / "q0-d0-m1").iterdir()), sorted(p.name for p in paths))

    def test_write_tables_takes_the_hash_once_or_as_given(self):
        calls = itertools.count()

        def hashes(paths):
            return {pickle.loads(p.read_bytes())["rules_hash"] for p in paths}

        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "rules_hash", lambda: f"{next(calls):016x}"):
            self.assertEqual(hashes(S.write_tables(Q0, self.table, Path(d))), {"0" * 16})  # one for all five files
            self.assertEqual(hashes(S.write_tables(Q0, self.table, Path(d), rules="given")), {"given"})


class SolveRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reach = S.reach(Q0)
        cls.table = S.solve(Q0, cls.reach)

    def test_an_incomplete_count_is_refused(self):
        with self.assertRaises(ValueError):
            S.solve(Q0, S.reach(Q0, cap=100))

    def test_a_non_terminal_state_without_a_move_raises(self):
        lone = S.Reach({0: [C.canonical(C.initial())[0]]}, True, 0.0)
        with mock.patch.object(C, "moves", return_value=[]), self.assertRaises(RuntimeError):
            S.solve(Q0, lone)

    def test_a_stopped_solve_keeps_a_stamped_checkpoint_of_the_levels_done(self):
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == 12:
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.solve(Q0, self.reach, ckpt_dir=Path(d), on_level=stop)
            [path] = Path(d).iterdir()
            saved = S._load_checkpoint(path, S._stamp(Q0))
            self.assertEqual(path.name, "q0-d0-m1-solve.pkl")
            self.assertEqual(saved["next"], 9)
            self.assertEqual(set(saved["table"]), {k for p, keys in self.reach.levels.items() if p >= 12 for k in keys})

    def test_a_solve_killed_after_its_last_level_resumes_without_solving_anything(self):
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == min(self.reach.levels):
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.solve(Q0, self.reach, ckpt_dir=Path(d), on_level=stop)
            with mock.patch.object(C, "moves", side_effect=AssertionError("nothing is left to solve")):
                self.assertEqual(S.solve(Q0, self.reach, ckpt_dir=Path(d)), self.table)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_given_stamp_is_what_the_checkpoint_carries_and_what_a_rerun_needs(self):
        mine = ("f" * 16, tuple(Q0))
        with tempfile.TemporaryDirectory() as d:
            def stop(p):
                if p == 12:
                    raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                S.solve(Q0, self.reach, ckpt_dir=Path(d), on_level=stop, stamp=mine)
            [path] = Path(d).iterdir()
            self.assertEqual(S._load_checkpoint(path, mine)["next"], 9)
            self.assertIsNone(S._load_checkpoint(path, S._stamp(Q0)))
            solved = []
            self.assertEqual(S.solve(Q0, self.reach, ckpt_dir=Path(d), on_level=solved.append, stamp=mine), self.table)
            self.assertEqual(solved, [9, 6, 3, 0])  # it resumed from the checkpoint
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_keep_leaves_the_finished_checkpoint_for_the_caller_to_delete(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.solve(Q0, self.reach, ckpt_dir=Path(d), keep=True), self.table)
            [path] = Path(d).iterdir()
            self.assertIsNone(S._load_checkpoint(path, S._stamp(Q0))["next"])
            S._delete_checkpoint(path)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_checkpoint_of_other_rules_is_ignored_not_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q0-d0-m1-solve.pkl"
            path.write_bytes(pickle.dumps({"stamp": ("0" * 16, tuple(Q0)), "table": {}, "next": None}))  # if used: {}
            self.assertEqual(S.solve(Q0, self.reach, ckpt_dir=Path(d)), self.table)
            self.assertEqual(list(Path(d).iterdir()), [])


class Cli(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.results, self.data = tmp / "results", tmp / "results" / "data"
        self.enterContext(mock.patch.object(S, "DATA", self.data))
        self.enterContext(mock.patch.object(S, "RESULTS", self.results))

    def run_cli(self, *args, q="0"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            S.main(["--quantumness", q, "--decoherence", "0", *args])
        return json.loads(out.getvalue())

    def test_count_only_saves_the_count_and_a_rerun_reuses_it(self):
        summary = self.run_cli("--count-only")
        self.assertEqual((summary["states"], summary["complete"]), (627, True))
        self.assertNotIn("root", summary)
        saved = pickle.loads((self.data / "q0-d0-m1" / "reach.pkl").read_bytes())
        self.assertEqual(saved["stamp"], S._stamp(Q0))
        self.assertEqual(sum(map(len, saved["reach"].levels.values())), 627)
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])
        self.assertEqual(list((self.data / "q0-d0-m1").glob("w=*")), [])
        with mock.patch.object(S, "reach", side_effect=AssertionError("recounted")):
            self.assertEqual(self.run_cli("--count-only")["states"], 627)
            self.assertEqual(self.run_cli()["root"]["1/2"], "(0, 0, 0, 1, 9)")  # solves from the saved count
        self.assertEqual(len(list((self.data / "q0-d0-m1").glob("w=*"))), 5)

    def test_a_stale_count_file_is_recounted(self):
        self.run_cli("--count-only")
        path = self.data / "q0-d0-m1" / "reach.pkl"
        saved = pickle.loads(path.read_bytes())
        path.write_bytes(pickle.dumps({**saved, "stamp": ("0" * 16, tuple(Q0))}))
        with mock.patch.object(S, "reach", wraps=S.reach) as counted:
            self.assertEqual(self.run_cli("--count-only")["states"], 627)
        counted.assert_called_once()
        self.assertEqual(pickle.loads(path.read_bytes())["stamp"], S._stamp(Q0))

    def test_a_capped_count_writes_its_summary_and_no_tables_and_a_rerun_resumes(self):
        with mock.patch.object(S, "solve", side_effect=AssertionError("solved an incomplete count")):
            summary = self.run_cli("--cap", "100")
        self.assertFalse(summary["complete"])
        self.assertNotIn("root", summary)
        self.assertEqual(json.loads((self.results / "q0-d0-m1.json").read_text()), summary)
        self.assertFalse((self.data / "q0-d0-m1").exists())  # no count file, no tables
        self.assertEqual([p.name for p in (self.data / "checkpoints").iterdir()], ["q0-d0-m1-reach.pkl"])
        full = self.run_cli()
        self.assertEqual((full["complete"], full["states"]), (True, 627))
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])

    def test_a_full_run_writes_the_count_the_tables_and_the_summary(self):
        summary = self.run_cli()
        self.assertEqual(summary["config"], {"quantumness": 0, "decoherence": 0, "manual_measurement": True})
        self.assertEqual(summary["rules_hash"], S.rules_hash())
        self.assertEqual((summary["states"], summary["complete"]), (627, True))
        self.assertEqual(sum(summary["levels"].values()), 627)
        self.assertEqual(list(map(int, summary["levels"])), sorted(map(int, summary["levels"])))
        self.assertIsInstance(summary["seconds"], float)
        self.assertEqual(list(summary["root"]), ["0", "1/4", "1/2", "3/4", "1"])
        self.assertEqual(summary["root"]["1/2"], "(0, 0, 0, 1, 9)")
        self.assertEqual(json.loads((self.results / "q0-d0-m1.json").read_text()), summary)
        self.assertEqual(sorted(p.name for p in (self.data / "q0-d0-m1").iterdir()),
                         ["reach.pkl"] + [f"w={n}.pkl" for n in ("0-1", "1-1", "1-2", "1-4", "3-4")])
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])

    def test_a_failed_table_write_keeps_the_solve_checkpoint_and_the_rerun_only_writes(self):
        with mock.patch.object(S, "write_tables", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_cli()
        self.assertEqual([p.name for p in (self.data / "checkpoints").iterdir()], ["q0-d0-m1-solve.pkl"])
        with mock.patch.object(C, "moves", side_effect=AssertionError("solved again")):
            summary = self.run_cli()
        self.assertEqual(summary["root"]["1/2"], "(0, 0, 0, 1, 9)")
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])
        self.assertEqual(len(list((self.data / "q0-d0-m1").glob("w=*"))), 5)

    def test_a_failed_summary_write_keeps_the_solve_checkpoint_and_the_rerun_only_writes(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_cli()
        self.assertEqual([p.name for p in (self.data / "checkpoints").iterdir()], ["q0-d0-m1-solve.pkl"])
        self.assertEqual(len(list((self.data / "q0-d0-m1").glob("w=*.pkl"))), 5)  # the tables did get written
        with mock.patch.object(C, "moves", side_effect=AssertionError("solved again")):
            summary = self.run_cli()
        self.assertEqual(summary["root"]["1/2"], "(0, 0, 0, 1, 9)")
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])

    def test_a_failed_count_save_keeps_the_count_checkpoint_and_the_rerun_only_saves(self):
        real = S._save_checkpoint

        def full_disk(path, payload):
            if path.name == "reach.pkl":
                raise OSError("disk full")
            real(path, payload)

        with mock.patch.object(S, "_save_checkpoint", full_disk), self.assertRaises(OSError):
            self.run_cli("--count-only")
        self.assertEqual([p.name for p in (self.data / "checkpoints").iterdir()], ["q0-d0-m1-reach.pkl"])
        with mock.patch.object(C, "moves", side_effect=AssertionError("counted again")):
            self.assertEqual(self.run_cli("--count-only")["states"], 627)
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])
        self.assertTrue((self.data / "q0-d0-m1" / "reach.pkl").exists())

    def test_a_count_checkpoint_left_beside_a_saved_count_is_cleared_on_reuse(self):
        real = S._delete_checkpoint

        def killed_just_after_the_count_is_saved(path):
            if path.name.endswith("-reach.pkl"):
                raise KeyboardInterrupt
            real(path)

        with mock.patch.object(S, "_delete_checkpoint", killed_just_after_the_count_is_saved):
            with self.assertRaises(KeyboardInterrupt):
                self.run_cli("--count-only")
        self.assertTrue((self.data / "q0-d0-m1" / "reach.pkl").exists())
        self.assertEqual([p.name for p in (self.data / "checkpoints").iterdir()], ["q0-d0-m1-reach.pkl"])
        with mock.patch.object(S, "reach", side_effect=AssertionError("recounted")):
            self.assertEqual(self.run_cli("--count-only")["states"], 627)
        self.assertEqual(list((self.data / "checkpoints").iterdir()), [])

    def test_each_checkpoint_goes_only_after_the_result_it_protects_is_written(self):
        seen, real = [], S._delete_checkpoint

        def spy(path):
            tables = len(list((self.data / "q0-d0-m1").glob("w=*.pkl")))
            seen.append((path.name, (self.data / "q0-d0-m1" / "reach.pkl").exists(), tables,
                         (self.results / "q0-d0-m1.json").exists()))
            real(path)

        with mock.patch.object(S, "_delete_checkpoint", spy):
            self.run_cli()
        # (checkpoint, reach.pkl there, tables there, summary there) at the moment each was deleted
        self.assertEqual(seen, [("q0-d0-m1-reach.pkl", True, 0, False), ("q0-d0-m1-solve.pkl", True, 5, True)])

    @contextlib.contextmanager
    def changing_hash(self):
        """rules_hash() as if the code on disk were edited after every look: ...00, then ...01, ...02 and so on.
        Yields the list of looks taken so far."""
        looks = []

        def rules_hash():
            looks.append(None)
            return f"{len(looks) - 1:016x}"

        with mock.patch.object(S, "rules_hash", rules_hash):
            yield looks

    def test_one_hash_taken_at_the_start_stamps_every_artifact(self):
        start = "0" * 16
        with self.changing_hash() as looks:
            summary = self.run_cli()
        self.assertEqual(len(looks), 1)  # the whole run looks at the code once, when it starts
        self.assertEqual(summary["rules_hash"], start)
        self.assertEqual(json.loads((self.results / "q0-d0-m1.json").read_text())["rules_hash"], start)
        saved = pickle.loads((self.data / "q0-d0-m1" / "reach.pkl").read_bytes())
        self.assertEqual(saved["stamp"], (start, tuple(Q0)))
        tables = list((self.data / "q0-d0-m1").glob("w=*.pkl"))
        self.assertEqual(len(tables), 5)
        self.assertEqual({pickle.loads(p.read_bytes())["rules_hash"] for p in tables}, {start})

    def stamp_of_the_checkpoint_left_by_a_run_stopped_after(self, stage, level):
        def progress(name):
            def note(p):
                if name == stage and p == level:
                    raise KeyboardInterrupt
            return note

        with self.changing_hash(), mock.patch.object(S, "_progress", progress):
            with self.assertRaises(KeyboardInterrupt):
                self.run_cli()
        [path] = (self.data / "checkpoints").iterdir()
        self.assertEqual(path.name, f"q0-d0-m1-{'reach' if stage == 'count' else 'solve'}.pkl")
        return pickle.loads(path.read_bytes())["stamp"]

    def test_a_count_checkpoint_carries_the_hash_from_the_start_of_the_run(self):
        self.assertEqual(self.stamp_of_the_checkpoint_left_by_a_run_stopped_after("count", 9), ("0" * 16, tuple(Q0)))

    def test_a_solve_checkpoint_carries_the_hash_from_the_start_of_the_run(self):
        self.assertEqual(self.stamp_of_the_checkpoint_left_by_a_run_stopped_after("solve", 12), ("0" * 16, tuple(Q0)))

    def test_manual_off_names_the_configuration(self):
        summary = self.run_cli("--manual-off", "--count-only")
        self.assertEqual(summary["config"]["manual_measurement"], False)
        self.assertTrue((self.results / "q0-d0-m0.json").exists())

    def test_a_setting_outside_the_model_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            self.run_cli(q="3")

    def test_pickles_written_by_python_dash_m_load_in_another_process(self):
        with tempfile.TemporaryDirectory() as d:  # a copy of the package, so that its results/ is the copy's
            shutil.copytree(Path(S.__file__).parent, Path(d) / "tiqtaqtoe",
                            ignore=shutil.ignore_patterns("__pycache__"))

            def run(*args):
                proc = subprocess.run([sys.executable, *args], cwd=d, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                return proc.stdout

            run("-m", "tiqtaqtoe.solve", "--quantumness", "0", "--decoherence", "0")
            load = ("import pickle; from pathlib import Path; "
                    "d = pickle.loads(Path('results/data/q0-d0-m1/%s').read_bytes()); print(type(%s))")
            entry = run("-c", load % ("w=1-2.pkl", "next(iter(d['entries'].values()))"))
            self.assertEqual(entry.strip(), "<class 'tiqtaqtoe.solve.Entry'>")
            self.assertEqual(run("-c", load % ("reach.pkl", "d['reach']")).strip(), "<class 'tiqtaqtoe.solve.Reach'>")


if __name__ == "__main__":
    unittest.main()
