"""server.py: the tournament's POST /move endpoint, answered from the solved tables."""

import json
import shutil
import tempfile
import threading
import unittest
import urllib.request
from fractions import Fraction
from pathlib import Path

from tiqtaqtoe import compact as C
from tiqtaqtoe import engine as E
from tiqtaqtoe import server as V
from tiqtaqtoe import solve as S
from tiqtaqtoe import teacher as T

Q0, HALF = C.Config(0, 0, True), Fraction(1, 2)


def payload(e, quantumness=0, decoherence=0):
    """The tournament payload of the engine state e, as fetchRemoteAiMove sends it (the fields server.py reads)."""
    turn = "X" if e.turn == 1 else "O"
    return {"player": turn, "currentTurn": turn, "measuredSquares": list(e.measured),
            "quantumState": {b: {"real": a.real, "imaginary": a.imag} for b, a in e.amps},
            "settings": {"quantumness": quantumness, "decoherence": decoherence},
            "legalMoves": [E.move_to_harness(m) for m in E.legal_moves(e, quantumness)]}


class ChooseMove(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        S.write_tables(Q0, S.solve(Q0, S.reach(Q0)), cls.dir)
        cls.t = T.Teacher(Q0, cls.dir)
        cls.server = V.MoveServer(cls.dir, HALF)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def best(self, e):
        """The harness moves of e with the mover's best value at w = 1/2."""
        mv = self.t.move_values(e, HALF)
        pick = max if e.turn == 1 else min
        top = pick(v for _, v, _ in mv)
        return [E.move_to_harness(m) for m, v, _ in mv if v == top]

    def test_root_move_is_legal_and_optimal_for_both_colours(self):
        for first in (1, 2):
            p = payload(E.initial(first))
            move, how = self.server.choose(p)
            self.assertEqual(how, "table")
            self.assertIn(move, p["legalMoves"])
            self.assertIn(move, self.best(E.initial(first)))

    def test_mid_game_move_is_optimal(self):
        e = E.initial()
        for m in (("c", 4), ("c", 0)):
            e = E.do_move(e, m, 0)[0][1]
        move, how = self.server.choose(payload(e))
        self.assertEqual(how, "table")
        self.assertIn(move, self.best(e))

    def test_unsolved_settings_fall_back_to_a_legal_move(self):
        p = payload(E.initial(), quantumness=3)
        move, how = self.server.choose(p)
        self.assertIn(move, p["legalMoves"])
        self.assertTrue(how.startswith("fallback"), how)

    def test_decoherence_without_counters_falls_back(self):
        p = payload(E.initial(), decoherence=2)
        move, how = self.server.choose(p)
        self.assertIn(move, p["legalMoves"])
        self.assertTrue(how.startswith("fallback"), how)

    def test_legal_list_disagreeing_with_ours_falls_back(self):
        p = payload(E.initial())
        p["legalMoves"] = [[8]]
        self.assertEqual(self.server.choose(p), ([8], "fallback: legalMoves differ from ours"))

    def test_garbage_state_still_answers_a_legal_move(self):
        move, how = self.server.choose({"player": "X", "legalMoves": [[0], [1, 5], [7]]})
        self.assertIn(move, [[0], [1, 5], [7]])
        self.assertTrue(how.startswith("fallback"), how)

    def test_no_legal_moves_is_an_error(self):
        with self.assertRaises(ValueError):
            self.server.choose({"player": "X", "legalMoves": []})


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        S.write_tables(Q0, S.solve(Q0, S.reach(Q0)), cls.dir)
        cls.httpd = V.make_http_server("127.0.0.1", 0, V.MoveServer(cls.dir, HALF))
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir)

    def request(self, path, body=None, method=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def test_post_move(self):
        p = payload(E.initial())
        status, headers, body = self.request("/move", p)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn(json.loads(body)["move"], p["legalMoves"])

    def test_health(self):
        status, _, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_cors_preflight(self):
        status, headers, _ = self.request("/move", method="OPTIONS")
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")

    def test_bad_json_is_400_not_a_crash(self):
        req = urllib.request.Request(self.url + "/move", data=b"{not json", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 400)

    def test_unknown_path_is_404(self):
        self.assertEqual(self.request("/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
