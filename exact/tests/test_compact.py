"""compact.py against engine.py (plan Task 3): the exact model must agree with the float port on random games."""

import random
import unittest
from collections import defaultdict
from fractions import Fraction

from tiqtaqtoe import compact as C
from tiqtaqtoe import engine as E

Z = (0,) * 9
Q2 = C.Config(2, 0, True)
CONFIGS = [C.Config(q, d, m) for q in (0, 1, 2) for d in (0, 1, 2, 3) for m in (True, False)]
SWAP_COLOURS = str.maketrans("XO", "OX")


def random_engine_states(cfg, games, seed):
    rng = random.Random(seed)
    for g in range(games):
        e = E.initial(first=1 + g % 2)
        while E.winner(e) == E.CONTINUE:
            yield e
            mv = rng.choice(E.legal_moves(e, cfg.quantumness, cfg.manual))
            outs = E.do_move(e, mv, cfg.quantumness, cfg.manual, cfg.decoherence)
            e = rng.choices([t for _, t in outs], [p for p, _ in outs])[0]


def random_compact_states(cfg, games, seed):
    """Non-terminal states of random games played inside the model (cheap, so many more than the engine's)."""
    rng = random.Random(seed)
    for g in range(games):
        s = C.initial(first=1 + g % 2)
        while C.winner(s) == E.CONTINUE:
            yield s
            _, s, _ = rng.choice(C.apply(s, rng.choice(C.moves(s, cfg)), cfg))


def transform(e, perm, swap):
    """The engine state with square i moved to perm[i] in every branch, in measured and in counters; with swap, the
    colours and the turn exchanged. Branches are re-sorted by basis index."""

    def moved_branch(b):
        out = ["0"] * 9
        for i, ch in enumerate(b):
            out[perm[i]] = ch
        text = "".join(out)
        return text.translate(SWAP_COLOURS) if swap else text

    def moved(values):
        out = [None] * 9
        for i, v in enumerate(values):
            out[perm[i]] = v
        return tuple(out)

    amps = sorted(((moved_branch(b), c) for b, c in e.amps), key=lambda bc: E.basis_index(bc[0]))
    return E.EState(tuple(amps), moved(e.measured), 3 - e.turn if swap else e.turn, moved(e.counters))


def relative(w, mover):
    """A winner code with X/O wins written as 1 (the mover wins) or 2 (the other player wins)."""
    return (1 if w == mover else 2) if w in (E.X_WINS, E.O_WINS) else w


def outcome_classes(e, cfg):
    """For each legal move of e, its sorted outcomes (probability, canonical key, winner relative to the mover)."""
    classes = []
    for mv in E.legal_moves(e, cfg.quantumness, cfg.manual):
        outs = E.do_move(e, mv, cfg.quantumness, cfg.manual, cfg.decoherence)
        classes.append(sorted((round(p, 9), C.canonical(C.from_engine(t, cfg))[0], relative(E.winner(t), e.turn))
                              for p, t in outs))
    return sorted(classes)


class CrossCheck(unittest.TestCase):
    def test_matches_engine_on_random_games(self):
        checked = 0
        for n, cfg in enumerate(CONFIGS):
            for e in random_engine_states(cfg, games=12, seed=n):
                c = C.from_engine(e, cfg)
                self.assertEqual(C.moves(c, cfg), E.legal_moves(e, cfg.quantumness, cfg.manual))
                for mv in C.moves(c, cfg):
                    want, got = defaultdict(float), defaultdict(float)
                    for p, t in E.do_move(e, mv, cfg.quantumness, cfg.manual, cfg.decoherence):
                        want[(C.from_engine(t, cfg), E.winner(t))] += p
                    for p, t, w in C.apply(c, mv, cfg):
                        got[(t, w)] += float(p)
                    self.assertEqual(want.keys(), got.keys(), (cfg, c, mv))
                    for k in want:
                        self.assertAlmostEqual(want[k], got[k], places=9)
                    checked += 1
        self.assertGreater(checked, 3000)

    def test_rules_commute_with_board_symmetries_and_colour_swap(self):
        for cfg in (Q2, C.Config(2, 2, True)):
            for e in random_engine_states(cfg, games=4, seed=11):
                base = outcome_classes(e, cfg)
                for perm in E.SYMS:
                    for swap in (False, True):
                        self.assertEqual(outcome_classes(transform(e, perm, swap), cfg), base)


class Rules(unittest.TestCase):
    def test_classical_token_stays_unmeasured_at_q2(self):
        [(p, t, w)] = C.apply(C.CState((1, 1, 0, 2, 2, 0, 0, 0, 0), (), (), 1, Z), ("c", 2), Q2)
        self.assertEqual((p, t.board[2], w), (1, 3, E.CONTINUE))
        [(p, _, w)] = C.apply(C.CState((1, 1, 3, 2, 2, 4, 0, 0, 0), (), (), 1, Z), ("m", 2), Q2)
        self.assertEqual((p, w), (1, E.X_WINS))

    def test_entangling_needs_an_unmeasured_certain_opponent_token(self):
        s = C.CState((3, 0, 0, 0, 1, 0, 0, 0, 0), (), (), 2, Z)
        self.assertIn(("q", 0, 1), C.moves(s, Q2))
        self.assertNotIn(("q", 1, 4), C.moves(s, Q2))
        self.assertNotIn(("q", 0, 1), C.moves(s._replace(turn=1), Q2))
        [(_, t, _)] = C.apply(s, ("q", 0, 1), Q2)
        self.assertEqual((t.board[0], t.swaps), (0, ((0, 1),)))

    def test_full_board_resolves_every_link(self):
        s = C.CState((0, 0, 0, 0, 3, 4, 1, 2, 0), ((1, 0, 1),), ((2, 3),), 1, Z)
        outs = C.apply(s, ("c", 8), Q2)
        self.assertEqual((len(outs), sum(p for p, _, _ in outs)), (4, 1))
        self.assertTrue(all(t.pairs == t.swaps == () and set(t.board) <= {0, 1, 2} for _, t, _ in outs))

    def test_decoherence_measures_untouched_tokens(self):
        for d, measured in ((1, True), (2, False)):
            cfg = C.Config(2, d, True)
            [(_, s, _)] = C.apply(C.initial(), ("c", 0), cfg)
            [(_, s, _)] = C.apply(s, ("c", 8), cfg)
            self.assertEqual((s.board[0], s.board[8], s.clocks[0]), (1, 4, 0) if measured else (3, 4, 1))

    def test_canonical_is_invariant_and_encoding_round_trips(self):
        cfg = C.Config(2, 2, True)
        for e in random_engine_states(cfg, games=6, seed=7):
            s = C.from_engine(e, cfg)
            self.assertEqual(C.decode(C.encode(s)), s)
            for img, _ in C.images(s):
                self.assertEqual(C.canonical(img)[0], C.canonical(s)[0])

    def test_from_engine_rejects_q3_states_and_masks_stale_clocks(self):
        [(_, e)] = E.do_move(E.initial(), ("q", 0, 1), 3)
        [(_, e)] = E.do_move(e, ("q", 0, 2), 3)  # O entangles a pair square: legal only from Q3
        with self.assertRaises(ValueError):
            C.from_engine(e, Q2)
        stale = E.EState((("X00000000", 1 + 0j),), (True,) + (False,) * 8, 2, (5,) + (0,) * 8)
        self.assertEqual(C.from_engine(stale, C.Config(2, 2, True)).clocks, Z)


class Interface(unittest.TestCase):
    """Parts of the interface the brief's tests only touch indirectly."""

    def test_config_validates_and_names_itself(self):
        self.assertEqual(Q2.name, "q2-d0-m1")
        self.assertEqual(C.Config(1, 3, False).name, "q1-d3-m0")
        for bad in ((3, 0, True), (-1, 0, True), (2, 4, True), (2, -1, True)):
            with self.assertRaises(ValueError):
                C.Config(*bad)

    def test_potential_counts_two_per_measured_square_and_one_per_non_empty_square(self):
        self.assertEqual(C.potential(C.initial()), 0)
        s = C.CState((1, 3, 0, 0, 0, 0, 0, 0, 0), ((1, 2, 3),), ((4, 5),), 1, Z)
        self.assertEqual(C.potential(s), 2 * 1 + (2 + 2 + 2))

    def test_every_outcome_of_every_move_raises_the_potential_and_probabilities_sum_to_one(self):
        for n, cfg in enumerate(CONFIGS):
            for s in random_compact_states(cfg, games=30, seed=100 + n):
                for mv in C.moves(s, cfg):
                    outs = C.apply(s, mv, cfg)
                    self.assertEqual(sum(p for p, _, _ in outs), 1, (cfg, s, mv))
                    for _, t, _ in outs:
                        self.assertGreater(C.potential(t), C.potential(s), (cfg, s, mv))

    def test_canonical_is_the_least_encoded_image(self):
        for n, cfg in enumerate(CONFIGS):
            for s in random_compact_states(cfg, games=10, seed=200 + n):
                imgs = C.images(s)
                self.assertEqual(len(imgs), 16)
                self.assertEqual(C.canonical(s), min((C.encode(img), flag) for img, flag in imgs))
                self.assertEqual(C.recolour(C.recolour(s)), s)
                self.assertTrue(all(C.decode(C.encode(img)) == img for img, _ in imgs))

    def test_images_are_the_engine_states_under_symmetry_and_colour_swap(self):
        cfg = C.Config(2, 2, True)
        for e in random_engine_states(cfg, games=3, seed=5):
            want = {(C.from_engine(transform(e, perm, swap), cfg), swap) for perm in E.SYMS for swap in (False, True)}
            self.assertEqual(set(C.images(C.from_engine(e, cfg))), want)

    def test_branches_of_independent_links(self):
        # X certain on 0; O's pair on {1, 2}; a swap on {3, 4}: two equally likely contents each
        s = C.CState((3, 0, 0, 0, 0, 0, 0, 0, 0), ((2, 1, 2),), ((3, 4),), 1, Z)
        want = dict.fromkeys(("XO0XO0000", "XO0OX0000", "X0OXO0000", "X0OOX0000"), Fraction(1, 4))
        self.assertEqual(C.branches(s), want)

    def test_entangling_needs_quantumness_two(self):
        s = C.CState((3, 0, 0, 0, 0, 0, 0, 0, 0), (), (), 2, Z)  # X's certain unmeasured token, O to move
        q1 = C.Config(1, 0, True)
        self.assertIn(("q", 0, 1), C.moves(s, Q2))
        self.assertNotIn(("q", 0, 1), C.moves(s, q1))
        self.assertIn(("q", 1, 2), C.moves(s, q1))

    def test_an_entangling_move_that_fills_the_board_collapses_like_the_engine(self):
        # The harness marks the old certain token measured, then collapses the new swap; random play seldom gets here.
        for first in (1, 2):
            e = E.initial(first)
            for sq in (0, 1, 2, 3, 5, 4, 6, 7):  # eight unmeasured tokens at Q2, square 8 left empty
                [(_, e)] = E.do_move(e, ("c", sq), 2)
            c = C.from_engine(e, Q2)
            fills = [mv for mv in C.moves(c, Q2) if mv[0] == "q"]
            self.assertEqual(fills, [("q", 1, 8), ("q", 3, 8), ("q", 4, 8), ("q", 7, 8)])
            for mv in fills:
                want, got = defaultdict(float), defaultdict(float)
                for p, t in E.do_move(e, mv, 2):
                    want[(C.from_engine(t, Q2), E.winner(t))] += p
                for p, t, w in C.apply(c, mv, Q2):
                    got[(t, w)] += float(p)
                self.assertEqual((len(want), want.keys()), (2, got.keys()))
                for k in want:
                    self.assertAlmostEqual(want[k], got[k], places=9)

    def test_winner_counts_measured_lines_only(self):
        full = (1, 2, 1, 2, 1, 2, 2, 1, 2)  # a full board with no line
        states = {
            "X line": (C.CState((1, 1, 1, 0, 2, 2, 0, 0, 0), (), (), 2, Z), E.X_WINS),
            "O line": (C.CState((1, 0, 0, 2, 2, 2, 1, 0, 0), (), (), 1, Z), E.O_WINS),
            "one line each": (C.CState((1, 1, 1, 2, 2, 2, 0, 0, 0), (), (), 1, Z), E.TIE),
            "unmeasured line": (C.CState((3, 3, 3, 4, 4, 0, 0, 0, 0), (), (), 1, Z), E.CONTINUE),
            "full board": (C.CState(full, (), (), 1, Z), E.DRAW),
            "full board with a pair pending": (C.CState((0, 0) + full[2:], ((1, 0, 1),), (), 1, Z), E.CONTINUE),
        }
        for name, (s, want) in states.items():
            with self.subTest(name):
                self.assertEqual(C.winner(s), want)

    def test_from_engine_rejects_a_measured_flag_on_an_empty_square(self):
        e = E.EState((("0" * 9, 1 + 0j),), (True,) + (False,) * 8, 1)
        with self.assertRaises(ValueError):
            C.from_engine(e, Q2)

    def test_from_engine_rejects_states_outside_the_model(self):
        flat = (False,) * 9

        def state(*branches):  # (branch, probability) pairs, in basis order, as an engine state with X to move
            return E.EState(tuple((b, p ** 0.5 + 0j) for b, p in branches), flat, 1)

        cases = {
            "a square in superposition outside a link": state(("X00000000", 0.5), ("O00000000", 0.5)),
            "two squares linked as neither a pair nor a swap": state(("XX0000000", 0.5), ("OO0000000", 0.5)),
            "a pair whose branches are not equally likely": state(("0X0000000", 0.1), ("X00000000", 0.9)),
        }
        for name, e in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                C.from_engine(e, Q2)
        counted = E.EState((("X00000000", 1 + 0j),), flat, 2, (7,) + (0,) * 8)
        with self.assertRaises(ValueError):  # a counter the encoding cannot hold
            C.from_engine(counted, C.Config(2, 2, True))
        self.assertEqual(C.from_engine(counted, Q2).clocks, Z)  # with decoherence off, counters are ignored


if __name__ == "__main__":
    unittest.main()
