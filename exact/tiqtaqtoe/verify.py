"""Checks on a solved configuration's tables (spec, Verification 4 and 5), run through the Teacher.

bellman_violations: every stored entry is the best of its state's moves, with probabilities summing to 1.
policy_disagreement: how often the stored policy at one draw weight is not optimal at the next one up.
floor_check: the teacher plays the compact model against a random or a greedy opponent and must not do worse than the
state's value. Values stay Fractions; floats appear only in the floor rows' mean and stderr and in the rates.
"""

import argparse
import json
import math
import random
import sys
from fractions import Fraction

from tiqtaqtoe import compact, solve
from tiqtaqtoe.engine import CONTINUE, O_WINS, X_WINS
from tiqtaqtoe.solve import DATA, W
from tiqtaqtoe.teacher import Teacher

OPPONENTS = ("random", "greedy")
SIDE = {"X": 1, "O": 2}


def _states_with_moves(t):
    """[(key, state)] of the stored keys that have at least one legal move."""
    states = [(key, compact.decode(key)) for key in t.tables[0]]
    return [(key, s) for key, s in states if compact.moves(s, t.cfg)]


def bellman_violations(t):
    """The number of (stored key, w) whose entry's value differs from the best value over the state's moves (X the
    highest, O the lowest) or whose four probabilities do not sum to 1."""
    bad = 0
    for key, s in _states_with_moves(t):
        pick = max if s.turn == 1 else min
        for wi, w in enumerate(W):
            e = t.tables[wi][key]
            best = pick(v for _, v, _ in t.moves(s, w))
            if solve.value(e, w) != best or e.x + e.o + e.tie + e.draw != 1:
                bad += 1
    return bad


def policy_disagreement(t):
    """{"0-1/4": rate, ...}: over the stored keys with a legal move, the fraction where the stored-policy move at the
    lower draw weight of a neighbouring pair is not optimal (not of best value) at the higher one."""
    states, counts = _states_with_moves(t), [0] * (len(W) - 1)
    for _, s in states:
        sign = 1 if s.turn == 1 else -1
        listed = [t.moves(s, w) for w in W]
        for i in range(len(W) - 1):
            move, _ = solve.choose(s.turn, [(m, e) for m, _, e in listed[i]], W[i])
            values = {m: v for m, v, _ in listed[i + 1]}
            counts[i] += sign * values[move] != max(sign * v for v in values.values())
    return {f"{W[i]}-{W[i + 1]}": counts[i] / len(states) for i in range(len(W) - 1)}


def _payoff(win, w):
    """X's payoff of a finished game: X win 1, O win 0, tie or draw w."""
    return Fraction(1) if win == X_WINS else Fraction(0) if win == O_WINS else w


def _opponent_move(s, cfg, kind, rng):
    """The opponent's move in s: uniform for "random"; for "greedy" the highest immediate probability of its own
    outright win, random among ties."""
    moves = compact.moves(s, cfg)
    if kind == "random":
        return rng.choice(moves)
    mine = X_WINS if s.turn == 1 else O_WINS
    gain = [sum(p for p, _, win in compact.apply(s, m, cfg) if win == mine) for m in moves]
    return rng.choice([m for m, g in zip(moves, gain) if g == max(gain)])


def _play(t, kind, w, teacher_side, first, rng):
    """One game on the compact model; X's payoff."""
    s = compact.initial(SIDE[first])
    while True:
        if s.turn == SIDE[teacher_side]:
            move, _ = solve.choose(s.turn, [(m, e) for m, _, e in t.moves(s, w)], w)
        else:
            move = _opponent_move(s, t.cfg, kind, rng)
        outcomes = compact.apply(s, move, t.cfg)
        _, s, win = rng.choices(outcomes, weights=[float(p) for p, _, _ in outcomes])[0]
        if win != CONTINUE:
            return _payoff(win, w)


def floor_check(t, opponent, games, seed, w_values=(Fraction(0), Fraction(1, 2))):
    """One row {w, teacher, first, opponent, games, mean, stderr, value} per w, teacher side and first player: the
    teacher's mean score over `games` games (X's payoff, complemented when the teacher is O) and the start state's value
    likewise. Each row's games use random.Random(seed)."""
    if opponent not in OPPONENTS:
        raise ValueError(f"opponent {opponent!r} is not one of {', '.join(OPPONENTS)}")
    if games < 2:
        raise ValueError(f"games is {games}: the standard error needs at least 2 games")
    rows = []
    for w in w_values:
        for teacher_side in SIDE:
            for first in SIDE:
                rng = random.Random(seed)
                scores = [float(_payoff_for(_play(t, opponent, w, teacher_side, first, rng), teacher_side))
                          for _ in range(games)]
                mean = sum(scores) / games
                var = sum((x - mean) ** 2 for x in scores) / (games - 1)
                start = solve.value(t.lookup(compact.initial(SIDE[first]), w), w)
                value, stderr = _payoff_for(start, teacher_side), math.sqrt(var / games)
                rows.append({"w": w, "teacher": teacher_side, "first": first, "opponent": opponent, "games": games,
                             "mean": mean, "stderr": stderr, "value": value,
                             "ok": mean >= float(value) - 3 * stderr})
    return rows


def _payoff_for(x_payoff, side):
    """The payoff of `side` ("X" or "O") given X's payoff."""
    return x_payoff if side == "X" else 1 - x_payoff


def main(argv=None):
    """Run the three checks on a solved configuration's tables and print them as JSON."""
    parser = argparse.ArgumentParser(prog="python3 -m tiqtaqtoe.verify", description=main.__doc__)
    parser.add_argument("--quantumness", type=int, required=True)
    parser.add_argument("--decoherence", type=int, required=True)
    parser.add_argument("--manual-off", action="store_true", help="manual measurement off (default: on)")
    parser.add_argument("--games", type=int, default=200, help="games per floor row (default 200)")
    args = parser.parse_args(argv)
    if args.games < 2:
        parser.error("--games must be at least 2 (the standard error needs two games)")
    try:
        cfg = compact.Config(args.quantumness, args.decoherence, not args.manual_off)
    except ValueError as err:
        parser.error(str(err))
    t = Teacher(cfg, DATA)
    floor = [row for opponent in OPPONENTS for row in floor_check(t, opponent, args.games, seed=0)]
    result = {"bellman_violations": bellman_violations(t), "policy_disagreement": policy_disagreement(t),
              "floor": floor}
    json.dump(result, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
