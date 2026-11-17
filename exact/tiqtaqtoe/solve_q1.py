"""Exact quantumness-1 solve for both manual-measurement settings: python3 -m tiqtaqtoe.solve_q1 [out.json]"""

import json
import sys
import time

from tiqtaqtoe import q1

O_WIN_PROB = {q1.X_WINS: 0, q1.O_WINS: 1, q1.TIE: 0, q1.DRAW: 0}


def name(mv):
    return f"split [{mv[1]},{mv[2]}]" if mv[0] == "q" else f"X on {mv[1]}"


def solve(manual):
    t0 = time.time()
    score = q1.Solver(q1.SCORE, manual=manual)
    xwin = q1.Solver(q1.X_WIN_PROB, manual=manual)
    # O maximises its own win probability: negate so the solver's X-max / O-min convention still applies.
    owin = q1.Solver({k: -v for k, v in O_WIN_PROB.items()}, manual=manual)
    root = q1.initial()
    out = {
        "manual_measurement": manual,
        "value": str(score.value(root)),
        "x_win_prob": str(xwin.value(root)),
        "o_win_prob_when_o_maximises_it": str(-owin.value(root)),
        "openings": [
            {"move": name(mv), "value": str(v), "x_win_prob": str(xwin.move_value(root, mv))}
            for mv, v in score.openings()
        ],
        "canonical_states": len(score.memo),
        "seconds": round(time.time() - t0, 1),
    }
    out["split_0_4_x_win_prob"] = str(xwin.move_value(root, ("q", 0, 4)))
    return out


def main():
    results = [solve(manual) for manual in (True, False)]
    text = json.dumps(results, indent=1)
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
