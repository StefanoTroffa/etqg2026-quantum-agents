"""verify.py (plan Task 7): Bellman, floor and policy-disagreement checks on the solved tables."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from tiqtaqtoe import compact as C
from tiqtaqtoe import solve as S
from tiqtaqtoe import teacher as T
from tiqtaqtoe import verify as V

Q0 = C.Config(0, 0, True)


class Verify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        S.write_tables(Q0, S.solve(Q0, S.reach(Q0)), cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_bellman_check_passes_and_catches_a_corrupted_root(self):
        t = T.Teacher(Q0, self.dir)
        self.assertEqual(V.bellman_violations(t), 0)
        t.tables[2][C.canonical(C.initial())[0]] = S.Entry(1, 0, 0, 0, 1)
        self.assertGreater(V.bellman_violations(t), 0)

    def test_bellman_check_catches_probabilities_that_do_not_sum_to_one(self):
        t = T.Teacher(Q0, self.dir)
        key = C.canonical(C.initial())[0]
        t.tables[0][key] = t.tables[0][key]._replace(draw=Fraction(1, 7))
        self.assertGreater(V.bellman_violations(t), 0)

    def test_teacher_holds_its_value_against_random_and_greedy(self):
        t = T.Teacher(Q0, self.dir)
        for opponent in ("random", "greedy"):
            rows = V.floor_check(t, opponent, games=40, seed=3)
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertGreaterEqual(row["mean"], row["value"] - 3 * row["stderr"], row)

    def test_floor_rows_are_reproducible_and_complemented_for_o(self):
        t = T.Teacher(Q0, self.dir)
        rows = V.floor_check(t, "random", games=5, seed=1)
        self.assertEqual(rows, V.floor_check(t, "random", games=5, seed=1))
        self.assertEqual({(r["teacher"], r["first"]) for r in rows}, {("X", "X"), ("X", "O"), ("O", "X"), ("O", "O")})
        x_first = next(r for r in rows if r["w"] == Fraction(1, 2) and (r["teacher"], r["first"]) == ("X", "X"))
        o_first = next(r for r in rows if r["w"] == Fraction(1, 2) and (r["teacher"], r["first"]) == ("O", "O"))
        self.assertEqual(x_first["value"], o_first["value"])  # the colour-swapped game, complemented back

    def test_floor_rows_carry_ok_as_the_float_comparison(self):
        t = T.Teacher(Q0, self.dir)
        for row in V.floor_check(t, "greedy", games=20, seed=2):
            self.assertIs(row["ok"], row["mean"] >= float(row["value"]) - 3 * row["stderr"])

    def test_fewer_than_two_games_are_rejected(self):
        t = T.Teacher(Q0, self.dir)
        for games in (1, 0, -3):
            with self.assertRaises(ValueError):
                V.floor_check(t, "random", games=games, seed=0)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                V.main(["--quantumness", "0", "--decoherence", "0", "--games", str(games)])

    def test_unknown_opponent_is_rejected(self):
        with self.assertRaises(ValueError):
            V.floor_check(T.Teacher(Q0, self.dir), "clever", games=1, seed=0)

    def test_policy_disagreement_has_one_rate_per_neighbouring_pair(self):
        rates = V.policy_disagreement(T.Teacher(Q0, self.dir))
        self.assertEqual(list(rates), ["0-1/4", "1/4-1/2", "1/2-3/4", "3/4-1"])
        self.assertTrue(all(0 <= r <= 1 for r in rates.values()))

    def test_cli_prints_the_three_results_as_json(self):
        out = io.StringIO()
        with mock.patch.object(V, "DATA", self.dir), contextlib.redirect_stdout(out):
            V.main(["--quantumness", "0", "--decoherence", "0", "--games", "4"])
        result = json.loads(out.getvalue())
        self.assertEqual(set(result), {"bellman_violations", "policy_disagreement", "floor"})
        self.assertEqual(result["bellman_violations"], 0)
        self.assertEqual(len(result["floor"]), 16)


if __name__ == "__main__":
    unittest.main()
