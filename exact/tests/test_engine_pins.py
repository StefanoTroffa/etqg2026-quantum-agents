"""Literal pins on what compact.py imports from engine.py, which rules_hash() does not cover.

An engine edit that changes any of these changes the tables without changing the hash. This test fails first."""

import unittest

from tiqtaqtoe import engine as E


class EnginePins(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(E.LINES, ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8),
                                   (2, 4, 6)))

    def test_syms(self):
        self.assertEqual(E.SYMS, (
            (0, 1, 2, 3, 4, 5, 6, 7, 8), (2, 1, 0, 5, 4, 3, 8, 7, 6), (2, 5, 8, 1, 4, 7, 0, 3, 6),
            (0, 3, 6, 1, 4, 7, 2, 5, 8), (8, 7, 6, 5, 4, 3, 2, 1, 0), (6, 7, 8, 3, 4, 5, 0, 1, 2),
            (6, 3, 0, 7, 4, 1, 8, 5, 2), (8, 5, 2, 7, 4, 1, 6, 3, 0)))

    def test_char_and_winner_codes(self):
        self.assertEqual(E.CHAR, "0XO")
        self.assertEqual((E.CONTINUE, E.X_WINS, E.O_WINS, E.TIE, E.DRAW), (0, 1, 2, 3, 4))

    def test_components_of_a_pair_and_of_a_lone_token(self):
        def state(*branches):
            amps = tuple((b, 0.5 ** 0.5 + 0j) for b in branches)
            return E.EState(E._sorted(amps), (False,) * 9, 1)

        pair = state("XO0000000", "OX0000000")
        self.assertEqual(E.components(pair), [[0, 1]])
        self.assertEqual(E.components(state("X00000000", "0X0000000")), [[0, 1]])
        self.assertEqual(E.components(state("X00000000")), [])


PAYLOAD = {"quantumState": {"X00000000": {"real": 1, "imaginary": 0}}, "measuredSquares": [False] * 9,
           "currentTurn": "O"}


class Counters(unittest.TestCase):
    def test_payload_without_counters_and_no_argument_raises(self):
        with self.assertRaisesRegex(ValueError, "decoherenceCounters"):
            E.from_round_state(PAYLOAD)
        with self.assertRaises(ValueError):
            E.from_round_state({**PAYLOAD, "decoherenceCounters": None})

    def test_explicit_zeros_are_accepted(self):
        self.assertEqual(E.from_round_state(PAYLOAD, counters=(0,) * 9).counters, (0,) * 9)

    def test_payload_counters_are_read(self):
        d = {**PAYLOAD, "decoherenceCounters": [0, 2, 0, 0, 0, 0, 0, 0, 1]}
        self.assertEqual(E.from_round_state(d).counters, (0, 2, 0, 0, 0, 0, 0, 0, 1))

    def test_explicit_argument_wins_over_the_payload(self):
        d = {**PAYLOAD, "decoherenceCounters": [1] * 9}
        self.assertEqual(E.from_round_state(d, counters=[0, 0, 3, 0, 0, 0, 0, 0, 0]).counters,
                         (0, 0, 3, 0, 0, 0, 0, 0, 0))

    def test_wrong_length_counters_raise(self):
        with self.assertRaises(ValueError):
            E.from_round_state(PAYLOAD, counters=(0,) * 8)


if __name__ == "__main__":
    unittest.main()
