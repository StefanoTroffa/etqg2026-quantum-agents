"""teacher.py (plan Task 6): the solved tables of one configuration, asked about any state of it."""

import json
import pickle
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tiqtaqtoe import compact as C
from tiqtaqtoe import engine as E
from tiqtaqtoe import solve as S
from tiqtaqtoe import teacher as T

Q0, HALF = C.Config(0, 0, True), Fraction(1, 2)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def opening(name):  # "split [0,4]" or "X on 4", as in results/q1-solution.json
    if name.startswith("split"):
        a, b = name[name.index("[") + 1:-1].split(",")
        return ("q", int(a), int(b))
    return ("c", int(name.split()[-1]))


class Teacher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = S.solve(Q0, S.reach(Q0))
        cls.dir = Path(tempfile.mkdtemp())
        S.write_tables(Q0, cls.table, cls.dir)
        cls.t = T.Teacher(Q0, cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_entry_and_move_values_at_the_root(self):
        root = E.initial()
        mv = self.t.move_values(root, HALF)
        self.assertEqual([m for m, _, _ in mv], E.legal_moves(root, 0))
        self.assertEqual(max(v for _, v, _ in mv), S.value(self.t.entry(root, HALF), HALF))

    def test_o_first_root_is_the_colour_flip(self):
        for w in S.W:
            self.assertEqual(self.t.entry(E.initial(first=2), w), S.flip(self.t.entry(E.initial(), 1 - w)))

    def test_rejects_states_outside_the_configuration(self):
        [(_, pair)] = E.do_move(E.initial(), ("q", 0, 1), 1)
        won = E.EState((("XXXOO0000", 1 + 0j),), (True,) * 5 + (False,) * 4, 2)
        for bad in (pair, won):
            with self.assertRaises(T.OutOfConfiguration):
                self.t.entry(bad, HALF)
        with self.assertRaises(ValueError):
            self.t.entry(E.initial(), Fraction(1, 3))

    def test_float_noise_and_stale_counters_are_harmless(self):
        noisy = E.EState((("X00000000", 1 + 1e-12j),), (True,) + (False,) * 8, 2, (7,) + (0,) * 8)
        clean = E.EState((("X00000000", 1 + 0j),), (True,) + (False,) * 8, 2)
        self.assertEqual(self.t.entry(noisy, HALF), self.t.entry(clean, HALF))

    def test_rejects_stale_tables(self):
        with tempfile.TemporaryDirectory() as d:
            for p in S.write_tables(Q0, self.table, Path(d)):
                blob = pickle.loads(p.read_bytes())
                p.write_bytes(pickle.dumps({**blob, "rules_hash": "stale"}))
            with self.assertRaises(RuntimeError):
                T.Teacher(Q0, Path(d))

    @unittest.skipUnless(all((S.DATA / f"q1-d0-m{m}").exists() for m in (0, 1)), "run the Q1 solves (Task 8)")
    def test_q1_numbers_reproduce_results(self):  # spec, Verification 3
        for rec in json.loads((RESULTS / "q1-solution.json").read_text()):
            t, root = T.Teacher(C.Config(1, 0, rec["manual_measurement"])), E.initial()
            self.assertEqual(2 * S.value(t.entry(root, HALF), HALF) - 1, Fraction(rec["value"]))
            self.assertEqual(S.value(t.entry(root, 0), 0), Fraction(rec["x_win_prob"]))
            self.assertEqual(1 - S.value(t.entry(root, 1), 1), Fraction(rec["o_win_prob_when_o_maximises_it"]))
            half = {m: v for m, v, _ in t.move_values(root, HALF)}
            zero = {m: v for m, v, _ in t.move_values(root, Fraction(0))}
            for op in rec["openings"]:
                self.assertEqual(2 * half[opening(op["move"])] - 1, Fraction(op["value"]))
                self.assertEqual(zero[opening(op["move"])], Fraction(op["x_win_prob"]))

    def test_rejects_missing_tables(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, r"w=0-1\.pkl"):  # nothing solved there: the first file is missing
                T.Teacher(Q0, Path(d))
            S.write_tables(Q0, self.table, Path(d))[3].unlink()
            with self.assertRaisesRegex(RuntimeError, r"w=3-4\.pkl"):  # one of the five missing, and named
                T.Teacher(Q0, Path(d))

    def test_rejects_corrupt_and_old_shape_files_naming_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            paths = S.write_tables(Q0, self.table, Path(d))
            paths[2].write_bytes(b"not a pickle at all")
            with self.assertRaisesRegex(RuntimeError, r"w=1-2\.pkl.*solve.*again"):
                T.Teacher(Q0, Path(d))
            paths[2].write_bytes(pickle.dumps({"config": Q0}))  # an old shape: fields missing
            with self.assertRaisesRegex(RuntimeError, r"w=1-2\.pkl.*solve.*again"):
                T.Teacher(Q0, Path(d))
            paths[2].write_bytes(pickle.dumps([1, 2]))  # not even a dict
            with self.assertRaisesRegex(RuntimeError, r"w=1-2\.pkl.*solve.*again"):
                T.Teacher(Q0, Path(d))

    def test_rejects_tables_of_another_configuration(self):
        other = C.Config(1, 0, False)
        with tempfile.TemporaryDirectory() as d:
            S.write_tables(Q0, self.table, Path(d))
            (Path(d) / Q0.name).rename(Path(d) / other.name)  # Q0's tables, filed under q1-d0-m0
            with self.assertRaises(RuntimeError):
                T.Teacher(other, Path(d))

    def test_rejects_a_table_filed_under_another_weight(self):
        with tempfile.TemporaryDirectory() as d:
            paths = S.write_tables(Q0, self.table, Path(d))
            paths[2].write_bytes(paths[1].read_bytes())  # the table of w = 1/4 where w = 1/2 belongs
            with self.assertRaises(RuntimeError):
                T.Teacher(Q0, Path(d))

    def test_tables_are_five_dicts_in_w_order_and_read_live(self):
        t, root = T.Teacher(Q0, self.dir), C.canonical(C.initial())[0]
        self.assertEqual(t.cfg, Q0)
        self.assertIsInstance(t.tables, tuple)
        self.assertEqual([type(table) for table in t.tables], [dict] * 5)
        self.assertEqual([table[root] for table in t.tables], list(self.table[root]))
        t.tables[2][root] = S.Entry(1, 0, 0, 0, 1)  # as a Bellman check does to corrupt a root: the next query sees it
        self.assertEqual(t.lookup(C.initial(), HALF), S.Entry(1, 0, 0, 0, 1))
        self.assertEqual(self.t.lookup(C.initial(), HALF), S.Entry(0, 0, 0, 1, 9))  # the other Teacher is untouched

    def test_compact_states_and_engine_states_get_the_same_answers(self):
        [(_, e_mid)] = E.do_move(E.initial(), ("c", 4), 0)  # X on the centre, O to move
        [(_, s_mid, _)] = C.apply(C.initial(), ("c", 4), Q0)
        for e, s in ((E.initial(), C.initial()), (e_mid, s_mid)):
            for w in S.W:
                self.assertEqual(self.t.entry(e, w), self.t.lookup(s, w))
                self.assertEqual(self.t.move_values(e, w), self.t.moves(s, w))

    def test_w_is_taken_by_value_and_answers_stay_exact(self):
        root = C.initial()
        for w, same in ((0, Fraction(0)), (0.5, HALF), (0.75, Fraction(3, 4)), (1, Fraction(1)), (1.0, Fraction(1))):
            self.assertEqual(self.t.lookup(root, w), self.t.lookup(root, same))
            values = [v for _, v, _ in self.t.moves(root, w)]
            self.assertEqual(values, [v for _, v, _ in self.t.moves(root, same)])
            self.assertTrue(all(type(v) is Fraction for v in values), (w, values))  # never a float

    def test_w_outside_the_grid_is_a_value_error_naming_it_and_not_out_of_configuration(self):
        for w in (Fraction(1, 3), 0.3, 2, -1, None, "1/2"):
            for ask in (self.t.lookup, self.t.moves):
                with self.assertRaises(ValueError) as caught:
                    ask(C.initial(), w)
                self.assertNotIsInstance(caught.exception, T.OutOfConfiguration)
                self.assertIn(repr(w), str(caught.exception))

    def test_the_best_move_is_worth_the_stored_entry_for_every_state_in_both_colours(self):
        bad = []
        for key in self.t.tables[0]:
            for s in (C.decode(key), C.recolour(C.decode(key))):  # the stored orientation and its colour swap
                for w in S.W:
                    best = (max if s.turn == 1 else min)(v for _, v, _ in self.t.moves(s, w))
                    if best != S.value(self.t.lookup(s, w), w):
                        bad.append((s, w))
        self.assertEqual(bad, [])

    def test_moves_refuse_a_state_the_tables_lack_even_when_all_its_children_are_stored(self):
        t = T.Teacher(Q0, self.dir)
        [(_, s, _)] = C.apply(C.initial(), ("c", 4), Q0)
        for table in t.tables:
            del table[C.canonical(s)[0]]
        for ask in (t.lookup, t.moves):
            with self.assertRaises(T.OutOfConfiguration):
                ask(s, HALF)

    def test_a_state_missing_from_one_of_the_tables_is_refused_at_every_w(self):
        t = T.Teacher(Q0, self.dir)
        [(_, s, _)] = C.apply(C.initial(), ("c", 4), Q0)
        del t.tables[4][C.canonical(s)[0]]  # only the table of w = 1 lacks it
        for w in S.W:
            for ask in (t.lookup, t.moves):
                with self.assertRaises(T.OutOfConfiguration):
                    ask(s, w)

    def test_a_finished_game_is_refused_even_if_its_key_is_stored(self):
        t = T.Teacher(Q0, self.dir)
        won = C.CState((1, 1, 1, 2, 2, 0, 0, 0, 0), (), (), 2, C.ZEROS)  # X has the top row, O to move
        self.assertEqual(C.winner(won), E.X_WINS)
        for table in t.tables:
            table[C.canonical(won)[0]] = S.Entry(1, 0, 0, 0, 0)
        for ask in (t.lookup, t.moves):
            with self.assertRaises(T.OutOfConfiguration):
                ask(won, HALF)

    def test_an_engine_state_outside_the_model_is_out_of_configuration(self):
        third = 3 ** -0.5  # a token on square 0, 1 or 2, a third each: three squares linked, beyond the model
        triple = E.EState((("00X000000", third + 0j), ("0X0000000", third + 0j), ("X00000000", third + 0j)),
                          (False,) * 9, 2)
        with self.assertRaises(ValueError):
            C.from_engine(triple, Q0)
        for ask in (self.t.entry, self.t.move_values):
            with self.assertRaises(T.OutOfConfiguration):
                ask(triple, HALF)


if __name__ == "__main__":
    unittest.main()
