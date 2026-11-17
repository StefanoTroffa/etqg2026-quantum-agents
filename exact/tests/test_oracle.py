"""engine.py against the referee: every (state, move) pair of the JS oracle traces (plan Task 1, oracle/trace.mjs)."""

import json
import unittest
from pathlib import Path

from tiqtaqtoe import engine as E

TRACES = Path(__file__).resolve().parents[1] / "results" / "data" / "oracle"
TOL = 1e-9
EXPECTED = ["q0-d0", "q1-d0", "q1-d1", "q1-d2", "q1-d3", "q2-d0", "q2-d1", "q2-d2", "q2-d3", "q3-d0", "q4-d0"]


def outcome_groups(dist):
    """{(branch strings, measured, counters, turn, winner): [summed probability, amplitude vectors]}."""
    groups = {}
    for p, t in dist:
        key = (tuple(b for b, _ in t.amps), t.measured, t.counters, t.turn, E.winner(t))
        group = groups.setdefault(key, [0.0, []])
        group[0] += p
        group[1].append([c for _, c in t.amps])
    return groups


@unittest.skipUnless(list(TRACES.glob("q*-d*.jsonl")), "no oracle traces: run oracle/trace.mjs (plan Task 1)")
class Oracle(unittest.TestCase):
    def test_the_full_set_of_traces_is_present(self):
        self.assertEqual(sorted(p.stem for p in TRACES.glob("q*-d*.jsonl")), EXPECTED)

    def test_engine_matches_harness(self):
        for path in sorted(TRACES.glob("q*-d*.jsonl")):
            pairs = 0
            with path.open() as lines:
                for line in lines:
                    rec = json.loads(line)
                    q, d = rec["settings"]["quantumness"], rec["settings"]["decoherence"]
                    s = E.from_round_state(rec["state"])
                    self.assertEqual(E.legal_moves(s, q), [E.move_from_harness(m) for m in rec["legal"]], path.name)
                    for child in rec["children"]:
                        mv = E.move_from_harness(child["move"])
                        want = [(o["p"], E.from_round_state(o["state"])) for o in child["outcomes"]]
                        for o, (_, t) in zip(child["outcomes"], want):
                            self.assertEqual(o["state"]["winner"], E.winner(t))
                            self.assertEqual(o["state"]["emptySquares"], [E.is_empty(t, i) for i in range(9)])
                        self.assert_same_outcomes(E.do_move(s, mv, q, decoherence=d), want, (path.name, line, mv))
                        pairs += 1
            self.assertGreater(pairs, 200, path.name)

    def assert_same_outcomes(self, got, want, msg):
        """Same outcome groups; probabilities and every amplitude (each vector against the harness's) within TOL."""
        g, w = outcome_groups(got), outcome_groups(want)
        self.assertEqual(g.keys(), w.keys(), msg)
        for key, (pw, vw) in w.items():
            pg, vg = g[key]
            self.assertLess(abs(pg - pw), TOL, (msg, key, pg, pw))
            for amps in vg + vw:
                self.assertLess(max(abs(x - y) for x, y in zip(amps, vw[0])), TOL, (msg, key, amps, vw[0]))


if __name__ == "__main__":
    unittest.main()
