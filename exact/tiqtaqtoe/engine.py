"""Literal port of the referee's rules on the full amplitude state.

The referee is the organisers' harness, ../workshop_resources/local-tiqtaqtoe-server.js (line numbers below), on
@quantumplayed/quantum-game-engine 1.0.3 (Operations.Shift, Operations.SqrtSwap, JointQuantumSystem.measureTarget).
Floats, as in the JS. A measurement returns every outcome the engine's sampler can draw, with its probability, instead
of sampling. tests/test_oracle.py checks this record by record against JS traces. Squares are zero-based, as on the
site.
"""

from dataclasses import dataclass, replace
from math import sqrt

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
CONTINUE, X_WINS, O_WINS, TIE, DRAW = 0, 1, 2, 3, 4  # TicTacResult: UNFINISHED, X, O, BOTH, DRAW
CHAR = "0XO"  # qutrit values: 0 empty, 1 X, 2 O
KEEP = 1e-4  # serializeEngineState keeps branches with |a|^2 >= KEEP
NEAR = 0.01  # near(a, b) is |a - b| < NEAR
DEPENDENT = 0.01  # deriveEntanglements links squares whose joint probability is off the product by more
DEGENERATE = 1e-12  # measureTarget collapses normally only onto an outcome of mass above this


def _sym_maps():
    rot = tuple(3 * c + (2 - r) for r in range(3) for c in range(3))  # square (r, c) -> (c, 2 - r)
    ref = tuple(3 * r + (2 - c) for r in range(3) for c in range(3))
    maps, m = [], tuple(range(9))
    for _ in range(4):
        maps += [m, tuple(ref[m[i]] for i in range(9))]
        m = tuple(rot[m[i]] for i in range(9))
    return tuple(maps)


SYMS = _sym_maps()  # the 8 board symmetries, as square permutations


@dataclass(frozen=True)
class EState:
    amps: tuple  # ((branch string over "0XO", complex amplitude), ...), sorted by basis_index
    measured: tuple  # 9 bools: measuredSquares
    turn: int  # 1 = X, 2 = O
    counters: tuple = (0,) * 9  # decoherenceCounters


def basis_index(branch):
    """boardStringToBasisIndex (41-48): base 3, square 0 most significant, '0' < 'X' < 'O'."""
    k = 0
    for ch in branch:
        k = 3 * k + CHAR.index(ch)
    return k


def _norm2(c):
    return c.real * c.real + c.imag * c.imag


def _sorted(pairs):
    return tuple(sorted(pairs, key=lambda bc: basis_index(bc[0])))


def _serialize(pairs):
    """serializeEngineState (71-81): drop branches with |a|^2 < KEEP, without renormalising; order by basis index.

    The engine's cleanAmplitudes (drop an amplitude with |re| and |im| < 1e-10) needs no port: every operation starts
    from this output, so only a SqrtSwap's own cancellations fall that low, and this drops them next."""
    return _sorted((b, c) for b, c in pairs if _norm2(c) >= KEEP)


def initial(first=1):
    """createFreshRound (645-666) with `first` to move: 1 X, 2 O."""
    return EState((("0" * 9, 1 + 0j),), (False,) * 9, first)


def from_round_state(d, counters=None):
    """EState from a harness roundState, a tournament payload or a trace state.

    The decoherence counters come from the `counters` argument if given, else from the payload's
    `decoherenceCounters`. Tournament payloads carry none, so then the caller must pass the ones it tracked (zeros
    when decoherence is off); ValueError otherwise, never a silent zero clock."""
    amps = [(b, complex(c["real"], c["imaginary"])) for b, c in d["quantumState"].items()]
    measured = tuple(bool(m) for m in d["measuredSquares"])
    if counters is None:
        counters = d.get("decoherenceCounters")
    if counters is None:
        raise ValueError("the state has no decoherenceCounters: pass counters=, (0,) * 9 when decoherence is off")
    counters = tuple(int(n) for n in counters)
    if not amps or any(len(b) != 9 or set(b) - set(CHAR) for b, _ in amps):
        raise ValueError(f"bad quantumState {d['quantumState']!r}")
    if len(measured) != 9 or len(counters) != 9 or d["currentTurn"] not in ("X", "O"):
        raise ValueError("measuredSquares and decoherenceCounters need 9 entries, currentTurn 'X' or 'O'")
    return EState(_sorted(amps), measured, 1 if d["currentTurn"] == "X" else 2, counters)


def move_from_harness(m):
    """[i] -> ("c", i), [a, b] -> ("q", a, b), [i, "M"] -> ("m", i)."""
    if len(m) == 2 and m[1] in ("M", "measure"):
        move = ("m", m[0])
    elif len(m) in (1, 2):
        move = ("c", *m) if len(m) == 1 else ("q", *m)
    else:
        raise ValueError(f"bad move {m!r}")
    if not all(type(i) is int and 0 <= i <= 8 for i in move[1:]):
        raise ValueError(f"bad move {m!r}")
    return move


def move_to_harness(m):
    """("c", i) -> [i], ("q", a, b) -> [a, b], ("m", i) -> [i, "M"]."""
    return [m[1], "M"] if m[0] == "m" else list(m[1:])


def is_empty(s, i):
    """computeIsEmpty (233): empty in every branch."""
    return all(b[i] == "0" for b, _ in s.amps)


def winner(s):
    """result (349-351): winner (237-252) on the first branch by basis index, counting measured lines only."""
    branch = s.amps[0][0]
    nx = sum(all(branch[i] == "X" and s.measured[i] for i in line) for line in LINES)
    no = sum(all(branch[i] == "O" and s.measured[i] for i in line) for line in LINES)
    if nx > no:
        return X_WINS
    if no > nx:
        return O_WINS
    if nx > 0:
        return TIE
    return CONTINUE if "0" in branch else DRAW


def _probabilities(s):
    """computeSquareProbabilities (254-269): unnormalised [empty, X, O] masses per square."""
    probs = [[0.0, 0.0, 0.0] for _ in range(9)]
    for b, c in s.amps:
        for i, ch in enumerate(b):
            probs[i][CHAR.index(ch)] += _norm2(c)
    return probs


def _near(a, b):
    return abs(a - b) < NEAR


def partners(s):
    """deriveEntanglements (96-150): {square: linked squares}, linked when some joint probability (normalised) is
    off the product of the marginals by more than DEPENDENT."""
    total = sum(_norm2(c) for _, c in s.amps) or 1
    probs = [(b, _norm2(c) / total) for b, c in s.amps]
    marginal = [dict.fromkeys(CHAR, 0.0) for _ in range(9)]
    for b, p in probs:
        for i, ch in enumerate(b):
            marginal[i][ch] += p
    adj = {i: set() for i in range(9)}
    for i in range(9):
        for j in range(i + 1, 9):
            joint = {}
            for b, p in probs:
                joint[b[i] + b[j]] = joint.get(b[i] + b[j], 0.0) + p
            if any(abs(joint.get(x + y, 0.0) - marginal[i][x] * marginal[j][y]) > DEPENDENT
                   for x in CHAR for y in CHAR):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def components(s):
    """deriveEntanglements (152-177): the connected groups of two or more linked squares, each sorted."""
    adj, seen, groups = partners(s), set(), []
    for i in range(9):
        if i in seen:
            continue
        group, todo = [], [i]
        seen.add(i)
        while todo:
            j = todo.pop()
            group.append(j)
            todo += adj[j] - seen
            seen |= adj[j]
        if len(group) >= 2:
            groups.append(sorted(group))
    return groups


def _group(s, i):
    """pairs[squareToPair[i]]: the group holding square i, or None where squareToPair[i] is undefined."""
    return next((g for g in components(s) if i in g), None)


def legal(s, move, quantumness, manual=True):
    """isLegalMove (353-435). Moves: ("c", i), ("q", a, b) with a < b (the only pairs getAvailableMoves lists),
    ("m", i). manual=False forbids measure moves: the site's switch, which the harness lacks."""
    if move[0] == "m":
        return manual and not is_empty(s, move[1]) and not s.measured[move[1]]
    if move[0] == "c":
        return is_empty(s, move[1])
    a, b = move[1], move[2]
    if not a < b or quantumness == 0 or s.measured[a] or s.measured[b]:
        return False
    if is_empty(s, a) and is_empty(s, b):
        return True
    p, me = _probabilities(s), s.turn
    if not is_empty(s, a) and not is_empty(s, b):  # phase move; two squares outside any group pass too (undefined)
        return quantumness >= 4 and not _near(p[a][me], 0) and _group(s, a) == _group(s, b)
    full = b if is_empty(s, a) else a  # entangling move: never onto your own token
    if quantumness < 2 or not _near(p[full][me], 0):
        return False
    return quantumness >= 3 or _near(p[full][3 - me], 1)


def legal_moves(s, quantumness, manual=True):
    """getAvailableMoves (437-456): classical moves by square, then pairs by (a, b), then measures by square."""
    moves = [("c", i) for i in range(9)] + [("q", a, b) for a in range(9) for b in range(a + 1, 9)]
    return [m for m in moves + [("m", i) for i in range(9)] if legal(s, m, quantumness, manual)]


def _shift(amps, i, v):
    """Operations.Shift(v) on square i: value k -> (k + v) mod 3."""
    return tuple((b[:i] + CHAR[(CHAR.index(b[i]) + v) % 3] + b[i + 1:], c) for b, c in amps)


def _sqrt_swap(amps, a, b):
    """Operations.SqrtSwap on squares (a, b): |xy> -> (1+i)/2 |xy> + (1-i)/2 |yx> for x != y, |xx> unchanged."""
    out = {}
    for br, c in amps:
        if br[a] == br[b]:
            out[br] = out.get(br, 0) + c
            continue
        sw = list(br)
        sw[a], sw[b] = br[b], br[a]
        sw = "".join(sw)
        out[br] = out.get(br, 0) + c * (0.5 + 0.5j)
        out[sw] = out.get(sw, 0) + c * (0.5 - 0.5j)
    return out.items()


def _classical(s, i, measure_now):
    """doClassicalMove (458-471): Shift the mover's token onto square i, measured at once iff measure_now."""
    measured = s.measured[:i] + (s.measured[i] or measure_now,) + s.measured[i + 1:]
    return replace(s, amps=_serialize(_shift(s.amps, i, s.turn)), measured=measured)


def _quantum(s, a, b):
    """doQuantumMove (473-498): Shift onto a if both squares are empty, else onto the empty one if there is one
    (entangling); then SqrtSwap on the two squares (alone for a phase move)."""
    amps = s.amps
    if is_empty(s, a) or is_empty(s, b):
        amps = _shift(amps, a if is_empty(s, a) else b, s.turn)
    return replace(s, amps=_serialize(_sqrt_swap(amps, a, b)))


def _measure_target(amps, i):
    """measureTarget on square i, forced to each outcome its sampler can draw: [(probability, collapsed amps)].

    The sampler draws r in [0, 1) and takes the first outcome k, in ascending order, with r <= C_k (the cumulative
    unnormalised mass), else the last: outcome k has probability min(C_k, 1) - min(C_(k-1), 1), and the last absorbs
    any mass lost to pruning. The chosen outcome's branches are kept, scaled by 1/sqrt(its mass)."""
    masses = {}
    for b, c in amps:
        masses[CHAR.index(b[i])] = masses.get(CHAR.index(b[i]), 0.0) + _norm2(c)
    outcomes, cumulative, out = sorted(masses), 0.0, []
    for k, v in enumerate(outcomes):
        before = min(cumulative, 1.0)
        cumulative += masses[v]
        p = 1.0 - before if k == len(outcomes) - 1 else min(cumulative, 1.0) - before
        if p <= DEGENERATE:
            continue
        if masses[v] <= DEGENERATE:
            raise NotImplementedError(f"measureTarget's degenerate branch: square {i}, outcome {v}")
        scale = 1.0 / sqrt(masses[v])
        out.append((p, tuple((b, c * scale) for b, c in amps if b[i] == CHAR[v])))
    return out


def _measure_squares(s, squares):
    """measureSquares (500-526): measure `squares` in order and prune once; mark them measured, then unmark every
    square left empty in every branch."""
    dist = [(1.0, s.amps)]
    for i in squares:
        dist = [(p * q, kept) for p, amps in dist for q, kept in _measure_target(amps, i)]
    out = []
    for p, amps in dist:
        t = replace(s, amps=_serialize(amps))
        measured = tuple((t.measured[i] or i in squares) and not is_empty(t, i) for i in range(9))
        out.append((p, replace(t, measured=measured)))
    return out


def _with_group(s, i):
    """The squares a measurement of i takes (doMove 580-587, 603-609): i, then the unmeasured rest of its group."""
    return [i] + [j for j in _group(s, i) or () if j != i and not s.measured[j]]


def _occupied_unmeasured(s, i):
    return not s.measured[i] and not is_empty(s, i)


def _collapse(t, probs):
    """doMove's full-board branch (599-612): resolveDeterministicSquares (271-280) with `probs`, then each unmeasured
    occupied square in square order, measured with its group."""
    certain = [not is_empty(t, i) and (_near(probs[i][1], 1) or _near(probs[i][2], 1)) for i in range(9)]
    dist = [(1.0, replace(t, measured=tuple(m or c for m, c in zip(t.measured, certain))))]
    for i in range(9):
        nxt = []
        for p, u in dist:
            outs = _measure_squares(u, _with_group(u, i)) if _occupied_unmeasured(u, i) else [(1.0, u)]
            nxt += [(p * q, w) for q, w in outs]
        dist = nxt
    return dist


def _decohere(t, limit, touched):
    """computeDecoherence (528-568): reset the touched squares' counters, count up every other unmeasured occupied
    square, and measure the squares at the limit with their groups, resetting those counters."""
    counters = [0 if i in touched else n for i, n in enumerate(t.counters)]
    due = []
    for i in range(9):
        if _occupied_unmeasured(t, i):
            if i not in touched:
                counters[i] += 1
            if counters[i] >= limit:
                due.append(i)
    t = replace(t, counters=tuple(counters))
    if not due:
        return [(1.0, t)]
    squares = list(due)
    for i in due:
        squares += [j for j in _group(t, i) or () if j not in squares and not t.measured[j]]
    return [(p, replace(u, counters=tuple(0 if i in squares else n for i, n in enumerate(u.counters))))
            for p, u in _measure_squares(t, squares)]


def do_move(s, move, quantumness, manual=True, decoherence=0):
    """doMove (570-643) with project true: [(probability, next state)], the turn passed to the opponent."""
    if not legal(s, move, quantumness, manual):
        raise ValueError(f"illegal move {move}")
    if move[0] == "m":
        dist = _measure_squares(s, _with_group(s, move[1]))
    elif move[0] == "c":
        dist = [(1.0, _classical(s, move[1], quantumness <= 1))]
    else:
        dist = [(1.0, _quantum(s, move[1], move[2]))]
    out = []
    for p, t in dist:
        if all(not is_empty(t, i) for i in range(9)) and any(_occupied_unmeasured(t, i) for i in range(9)):
            # resolveDeterministicSquares reads roundState.probabilities as the move left them: recomputed by
            # measureSquares, otherwise still those of the state before the move
            branch = _collapse(t, _probabilities(t if move[0] == "m" else s))
        elif decoherence > 0:
            branch = _decohere(t, decoherence, move[1:])
        else:
            branch = [(1.0, t)]
        out += [(p * q, replace(u, turn=3 - s.turn)) for q, u in branch]
    return out
