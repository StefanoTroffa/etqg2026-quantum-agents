import random
import unittest
from collections import defaultdict
from fractions import Fraction

from tiqtaqtoe import engine as E
from tiqtaqtoe import q1


def to_compact(s):
    """Map an engine state (quantumness <= 1) onto the compact (board, pairs, turn) form."""
    first = s.amps[0][0]
    board = tuple(E.CHAR.index(first[i]) if s.measured[i] else 0 for i in range(9))
    adj, pairs = E.partners(s), set()
    for a in range(9):
        if s.measured[a] or E.is_empty(s, a):
            continue
        (b,) = adj[a]  # exactly one partner at quantumness 1
        owner = next(E.CHAR.index(br[a]) for br, _ in s.amps if br[a] != "0")
        pairs.add((owner, min(a, b), max(a, b)))
    return board, tuple(sorted(pairs)), s.turn


def engine_moves(s, q, manual):
    cands = [("c", i) for i in range(9)] + [("m", i) for i in range(9)]
    cands += [("q", a, b) for a in range(9) for b in range(a + 1, 9)]
    return [m for m in cands if E.legal(s, m, q, manual)]


def engine_dist(s, move, q, manual):
    out = defaultdict(float)
    for p, t in E.do_move(s, move, q, manual):
        out[(to_compact(t), E.winner(t))] += p
    return out


def compact_dist(state, move):
    out = defaultdict(float)
    for p, nxt, w in q1.apply(state, move):
        out[(nxt, w)] += float(p)
    return out


class CrossCheck(unittest.TestCase):
    def test_random_games_match_engine(self):
        rng = random.Random(2026)
        checked = 0
        for game in range(150):
            q, manual = (1, game % 2 == 0) if game % 5 else (0, True)
            s = E.initial()
            while True:
                c = to_compact(s)
                em = engine_moves(s, q, manual)
                cm = q1.moves(c, q, manual)
                self.assertEqual({m for m in em if m[0] != "m"}, {m for m in cm if m[0] != "m"})
                self.assertEqual({m[1] for m in em if m[0] == "m"}, {x for _, a, b in c[1] for x in (a, b)} if manual else set())
                for m in cm:
                    ed, cd = engine_dist(s, m, q, manual), compact_dist(c, m)
                    self.assertEqual(ed.keys(), cd.keys(), (c, m))
                    for k in ed:
                        self.assertAlmostEqual(ed[k], cd[k], places=9)
                    checked += 1
                mv = rng.choice(em)
                outs = E.do_move(s, mv, q, manual)
                s = rng.choices([t for _, t in outs], [p for p, _ in outs])[0]
                if E.winner(s) != E.CONTINUE:
                    break
        self.assertGreater(checked, 1000)


class Rules(unittest.TestCase):
    def test_classical_line_wins_with_pairs_pending(self):
        state = ((1, 1, 0, 2, 2, 0, 0, 0, 0), ((2, 7, 8),), 1)
        [(p, _, w)] = q1.apply(state, ("c", 2))
        self.assertEqual((p, w), (1, E.X_WINS))

    def test_full_board_measures_every_pair(self):
        # X pairs {0,1} and {2,5}, O pairs {3,4} and {6,7}; X fills square 8 classically.
        state = ((0,) * 9, ((1, 0, 1), (1, 2, 5), (2, 3, 4), (2, 6, 7)), 1)
        outs = q1.apply(state, ("c", 8))
        self.assertEqual(len(outs), 16)
        self.assertEqual(sum(p for p, _, _ in outs), 1)
        self.assertTrue(all(nxt[1] == () and sum(v == 0 for v in nxt[0]) == 4 for _, nxt, _ in outs))

    def test_equal_lines_after_measurement_is_tie(self):
        state = ((1, 1, 0, 2, 2, 0, 0, 0, 0), ((1, 2, 6), (2, 5, 7)), 1)
        outs = {nxt[0]: w for _, nxt, w in q1.apply(state, ("c", 8))}
        self.assertEqual(outs[(1, 1, 1, 2, 2, 2, 0, 0, 1)], E.TIE)

    def test_q0_is_a_draw(self):
        self.assertEqual(q1.Solver(q1.SCORE, quantumness=0).value(q1.initial()), 0)

    def test_symmetries_form_the_dihedral_group(self):
        self.assertEqual(len(set(q1.SYMS)), 8)
        self.assertEqual({m[4] for m in q1.SYMS}, {4})


if __name__ == "__main__":
    unittest.main()
