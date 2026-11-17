"""Compact quantumness-0/1 model and exact solver.

At quantumness <= 1 every superposition is an independent 50/50 pair on two squares that were empty, so a state is
(board, pairs, turn): board holds measured tokens (0 empty, 1 X, 2 O), pairs is a sorted tuple of (owner, a, b) with
a < b. Every move raises 2*measured + covered squares, so the game graph is a DAG and memoised backward induction is
exact (no value iteration needed). Squares are zero-based, as on the site.
"""

from fractions import Fraction

from tiqtaqtoe.engine import CONTINUE, DRAW, LINES, O_WINS, SYMS, TIE, X_WINS

HALF = Fraction(1, 2)


def initial():
    return ((0,) * 9, (), 1)


def covered(board, pairs):
    cov = [v != 0 for v in board]
    for _, a, b in pairs:
        cov[a] = cov[b] = True
    return cov


def winner(board):
    nx = sum(all(board[i] == 1 for i in line) for line in LINES)
    no = sum(all(board[i] == 2 for i in line) for line in LINES)
    if nx > no:
        return X_WINS
    if no > nx:
        return O_WINS
    if nx > 0:
        return TIE
    return DRAW if all(board) else CONTINUE


def moves(state, quantumness, manual=True):
    board, pairs, _ = state
    free = [i for i, c in enumerate(covered(board, pairs)) if not c]
    out = [("c", i) for i in free]
    if quantumness >= 1:
        out += [("q", a, b) for k, a in enumerate(free) for b in free[k + 1:]]
        if manual:
            out += [("m", a) for _, a, _b in pairs]  # measuring either square of a pair has the same effect
    return out


def _collapse(board, pairs, chosen):
    """Measure the given pairs: every combination of outcomes, each with probability 2^-len(chosen)."""
    dist = [(Fraction(1), board)]
    for owner, a, b in chosen:
        dist = [(p * HALF, bd[:s] + (owner,) + bd[s + 1:]) for p, bd in dist for s in (a, b)]
    rest = tuple(q for q in pairs if q not in chosen)
    return [(p, bd, rest) for p, bd in dist]


def apply(state, move):
    """Returns [(probability, next state, winner code)] with the turn passed to the opponent."""
    board, pairs, turn = state
    if move[0] == "m":
        dist = _collapse(board, pairs, [q for q in pairs if q[1] == move[1]])
    else:
        if move[0] == "c":
            i = move[1]
            board = board[:i] + (turn,) + board[i + 1:]
        else:
            pairs = tuple(sorted(pairs + ((turn, move[1], move[2]),)))
        full = all(covered(board, pairs))
        dist = _collapse(board, pairs, pairs) if full and pairs else [(Fraction(1), board, pairs)]
    return [(p, (bd, pr, 3 - turn), winner(bd)) for p, bd, pr in dist]


def canonical(state):
    board, pairs, turn = state
    keys = []
    for m in SYMS:
        bd = [0] * 9
        for i, v in enumerate(board):
            bd[m[i]] = v
        pr = tuple(sorted((o, *sorted((m[a], m[b]))) for o, a, b in pairs))
        keys.append((tuple(bd), pr, turn))
    return min(keys)


SCORE = {X_WINS: Fraction(1), O_WINS: Fraction(-1), TIE: Fraction(0), DRAW: Fraction(0)}
X_WIN_PROB = {X_WINS: Fraction(1), O_WINS: Fraction(0), TIE: Fraction(0), DRAW: Fraction(0)}


class Solver:
    """X maximises, O minimises the expected terminal reward."""

    def __init__(self, reward, quantumness=1, manual=True):
        self.reward, self.q, self.manual, self.memo = reward, quantumness, manual, {}

    def move_value(self, state, move):
        return sum(p * (self.reward[w] if w != CONTINUE else self.value(nxt)) for p, nxt, w in apply(state, move))

    def value(self, state):
        key = canonical(state)
        if key not in self.memo:
            vals = [self.move_value(key, mv) for mv in moves(key, self.q, self.manual)]
            self.memo[key] = max(vals) if key[2] == 1 else min(vals)
        return self.memo[key]

    def openings(self):
        """X's first moves, one per symmetry class: [(move, value)] sorted best first."""
        root, seen, out = initial(), set(), []
        for mv in moves(root, self.q, self.manual):
            nxt = canonical(apply(root, mv)[0][1])
            if nxt not in seen:
                seen.add(nxt)
                out.append((mv, self.move_value(root, mv)))
        return sorted(out, key=lambda t: -t[1])
