"""Exact compact model of TiqTaqToe for quantumness <= 2.

Below Q3 links never interact. Entangling needs an empty square and an unmeasured opponent token held with certainty;
no other quantum move may touch a non-empty square (a linked square is never certain, and moves on two non-empty
squares need Q4). So a state is single tokens plus independent two-square links, every marginal is 0, 1/2 or 1, no
harness threshold (0.01 correlation, 0.01 certainty, 1e-4 pruning) comes near a value, and phases never reach a
probability. Probabilities are exact powers of 1/2. Squares are zero-based, as on the site.

engine.py is the reference; tests/test_compact.py checks this model against it on random games. Every move raises
potential() strictly, so the game graph has no repeats.
"""

from fractions import Fraction
from typing import NamedTuple

from tiqtaqtoe import engine as E
from tiqtaqtoe.engine import CONTINUE, DRAW, LINES, O_WINS, SYMS, TIE, X_WINS

ONE = Fraction(1)
SHARE = {1 << k: Fraction(1, 1 << k) for k in range(5)}  # the chance of each outcome of k independent 50/50 links
ZEROS = (0,) * 9
TOL = 1e-9  # from_engine: how far an engine branch probability may be from the model's


class Config(NamedTuple("_Settings", [("quantumness", int), ("decoherence", int), ("manual", bool)])):
    """Settings the model covers: quantumness 0-2, decoherence off (0) or limit 1-3, manual measurement on or off."""

    __slots__ = ()

    def __new__(cls, quantumness, decoherence, manual):
        if not (0 <= quantumness <= 2 and 0 <= decoherence <= 3):
            raise ValueError(f"quantumness {quantumness} (0-2) or decoherence {decoherence} (0-3) is outside the model")
        return super().__new__(cls, quantumness, decoherence, manual)

    @property
    def name(self):
        return f"q{self.quantumness}-d{self.decoherence}-m{int(self.manual)}"


class CState(NamedTuple):
    """board[i]: 0 empty or linked, 1/2 X/O measured, 3/4 X/O unmeasured but certain. pairs: sorted (owner, a, b) with
    a < b, the owner's token on a or b (1/2 each) and the other square empty. swaps: sorted (a, b) with a < b, X on one
    and O on the other (1/2 each). turn: 1 X, 2 O. clocks: decoherence counters, 0 on empty or measured squares and
    everywhere when decoherence is off."""

    board: tuple
    pairs: tuple
    swaps: tuple
    turn: int
    clocks: tuple


def initial(first=1):
    """The empty board with `first` to move: 1 X, 2 O."""
    return CState(ZEROS, (), (), first, ZEROS)


def _put(t, i, v):
    return t[:i] + (v,) + t[i + 1:]


def _linked(s):
    """The squares held by a pair or a swap."""
    return {x for _, a, b in s.pairs for x in (a, b)} | {x for a, b in s.swaps for x in (a, b)}


def moves(s, cfg):
    """Legal moves in engine.legal_moves order: classical by square, pairs by (a, b), measures by square. Both squares
    of a link are listed for measuring."""
    board, linked = s.board, _linked(s)
    free = {i for i in range(9) if board[i] == 0 and i not in linked}
    prey = {i for i in range(9) if board[i] == 5 - s.turn} if cfg.quantumness >= 2 else set()  # certain opponent tokens
    out = [("c", i) for i in sorted(free)]
    if cfg.quantumness >= 1:
        out += [("q", a, b) for a in range(9) for b in range(a + 1, 9)
                if (a in free and (b in free or b in prey)) or (a in prey and b in free)]
    if cfg.manual:
        out += [("m", i) for i in range(9) if board[i] > 2 or i in linked]
    return out


def winner(s):
    """Winner code: measured lines decide (more wins, equal and > 0 ties); with none, DRAW if no square is empty and no
    pair is pending, that is, if every zero on the board is a swap square."""
    b = s.board
    nx = sum(b[x] == b[y] == b[z] == 1 for x, y, z in LINES)
    no = sum(b[x] == b[y] == b[z] == 2 for x, y, z in LINES)
    if nx != no:
        return X_WINS if nx > no else O_WINS
    if nx:
        return TIE
    return DRAW if b.count(0) == 2 * len(s.swaps) else CONTINUE


def potential(s):
    """2 x (measured squares) + (non-empty squares)."""
    b = s.board
    return 2 * (b.count(1) + b.count(2)) + 9 - b.count(0) + 2 * (len(s.pairs) + len(s.swaps))


def _link(s, a, b):
    """A quantum move: a pair of the mover's token on two empty squares, else a swap of the mover's new token with the
    opponent's certain one."""
    if s.board[a] == s.board[b] == 0:
        return s._replace(pairs=tuple(sorted(s.pairs + ((s.turn, a, b),))))
    held = a if s.board[a] else b
    return s._replace(board=_put(s.board, held, 0), swaps=tuple(sorted(s.swaps + ((a, b),))))


def _collapse(t, pairs, swaps, certain):
    """Measure the given pairs, swaps and certain tokens of t: [(probability, state)], every combination of outcomes at
    2^-links each. The measured squares' clocks reset."""
    board, clocks = t.board, t.clocks
    for i in certain:
        board, clocks = _put(board, i, board[i] - 2), _put(clocks, i, 0)
    dist = [board]
    for owner, a, b in pairs:
        dist = [_put(bd, x, owner) for bd in dist for x in (a, b)]
        clocks = _put(_put(clocks, a, 0), b, 0)
    for a, b in swaps:
        dist = [_put(_put(bd, x, 1), y, 2) for bd in dist for x, y in ((a, b), (b, a))]
        clocks = _put(_put(clocks, a, 0), b, 0)
    rest = (tuple(q for q in t.pairs if q not in pairs), tuple(q for q in t.swaps if q not in swaps))
    return [(SHARE[len(dist)], CState(bd, *rest, t.turn, clocks)) for bd in dist]


def _decohere(t, touched, limit):
    """computeDecoherence: the touched squares' clocks reset, every other unmeasured non-empty square's goes up by 1,
    and each square at the limit is measured together with its link partner."""
    live = [i for i, v in enumerate(t.board) if v > 2] + sorted(_linked(t))
    clocks = [0 if i in touched else c for i, c in enumerate(t.clocks)]
    for i in live:
        if i not in touched:
            clocks[i] += 1
    t = t._replace(clocks=tuple(clocks))
    due = {i for i in live if clocks[i] >= limit}
    if not due:
        return [(ONE, t)]
    return _collapse(t, [q for q in t.pairs if q[1] in due or q[2] in due],
                     [q for q in t.swaps if q[0] in due or q[1] in due], [i for i in due if t.board[i] > 2])


def _settle(t, touched, limit):
    """What a move leaves behind: with no square empty and something unmeasured, everything is measured; otherwise
    decoherence (off when limit is 0) ticks. `touched` are the squares the move named."""
    links = len(t.pairs) + len(t.swaps)
    if (links or 3 in t.board or 4 in t.board) and t.board.count(0) == 2 * links:
        return _collapse(t, t.pairs, t.swaps, [i for i, v in enumerate(t.board) if v > 2])
    return _decohere(t, touched, limit) if limit else [(ONE, t)]


def apply(s, move, cfg):
    """[(probability, next state with the turn passed, winner code)] for `move`, which must be one of moves(s, cfg):
    the solver and the teacher only pass listed moves, so legality is not re-checked here."""
    kind, i = move[0], move[1]
    if kind == "c":
        token = s.turn if cfg.quantumness <= 1 else s.turn + 2  # measured at once only up to Q1
        dist = [(ONE, s._replace(board=_put(s.board, i, token)))]
    elif kind == "q":
        dist = [(ONE, _link(s, i, move[2]))]
    else:
        dist = _collapse(s, [q for q in s.pairs if i in q[1:]], [q for q in s.swaps if i in q],
                         [i] if s.board[i] > 2 else [])
    out = []
    for p, t in dist:
        for q, u in _settle(t, move[1:], cfg.decoherence):
            u = u._replace(turn=3 - s.turn)
            out.append((p * q, u, winner(u)))
    return out


# --- symmetry, encoding and the canonical key -------------------------------------------------------------------

_COLOUR = (0, 2, 1, 4, 3)  # a board value with X and O exchanged


def recolour(s):
    """s with X and O exchanged: tokens, pair owners and the turn."""
    return CState(tuple(map(_COLOUR.__getitem__, s.board)), tuple(sorted((3 - o, a, b) for o, a, b in s.pairs)),
                  s.swaps, 3 - s.turn, s.clocks)


def _permute(s, perm):
    """s with square i moved to perm[i]."""
    board, clocks = [0] * 9, [0] * 9
    for i in range(9):
        board[perm[i]], clocks[perm[i]] = s.board[i], s.clocks[i]
    return s._replace(board=tuple(board), clocks=tuple(clocks),
                      pairs=tuple(sorted((o, *sorted((perm[a], perm[b]))) for o, a, b in s.pairs)),
                      swaps=tuple(sorted(tuple(sorted((perm[a], perm[b]))) for a, b in s.swaps)))


def images(s):
    """The 16 images of s under the board symmetries and the colour swap: [(state, swapped)], the unswapped first."""
    moved = [_permute(s, perm) for perm in SYMS]
    return [(m, False) for m in moved] + [(recolour(m), True) for m in moved]


# A key is the turn (lowest bit), then base-29 digits per square, then two bits of clock per square. A square's digit
# is 0 (empty, or the upper end of a link), its board value 1-4, or 5 + 8 * kind + (gap to the partner - 1) on the
# lower end of a link, kind 0 a pair of X, 1 a pair of O, 2 a swap.
RADIX = 29
SPAN = RADIX ** 9


def _link_digit(kind, a, b):
    return 5 + 8 * kind + b - a - 1


def encode(s):
    """The key of s, an integer below 2^63; decode() inverts it."""
    key = sum(v * RADIX ** i for i, v in enumerate(s.board))
    key += sum(_link_digit(o - 1, a, b) * RADIX ** a for o, a, b in s.pairs)
    key += sum(_link_digit(2, a, b) * RADIX ** a for a, b in s.swaps)
    clock = sum(c << 2 * i for i, c in enumerate(s.clocks))
    return (s.turn - 1) | (key + SPAN * clock) << 1


def decode(key):
    """The state whose encode() is key."""
    clock, rest = divmod(key >> 1, SPAN)
    board, pairs, swaps = [0] * 9, [], []
    for i in range(9):
        rest, digit = divmod(rest, RADIX)
        if digit <= 4:
            board[i] = digit
            continue
        kind, gap = divmod(digit - 5, 8)
        if kind == 2:
            swaps.append((i, i + gap + 1))
        else:
            pairs.append((kind + 1, i, i + gap + 1))
    return CState(tuple(board), tuple(sorted(pairs)), tuple(sorted(swaps)), 1 + (key & 1),
                  tuple(clock >> 2 * i & 3 for i in range(9)))


# What each part of a state adds to encode() of each of the 16 images (the order of images()): one row of 16 numbers
# per token, clock, link and turn, so canonical() sums rows instead of building the images.
_IMAGES = [(perm, False) for perm in SYMS] + [(perm, True) for perm in SYMS]


def _link_row(kind, a, b):
    row = []
    for perm, swapped in _IMAGES:
        lo, hi = sorted((perm[a], perm[b]))
        row.append(2 * _link_digit(1 - kind if swapped and kind < 2 else kind, lo, hi) * RADIX ** lo)
    return tuple(row)


_TURN = {t: tuple((3 - t if swapped else t) - 1 for _, swapped in _IMAGES) for t in (1, 2)}
_TOKEN = [[tuple(2 * (_COLOUR[v] if swapped else v) * RADIX ** perm[i] for perm, swapped in _IMAGES)
           for v in range(5)] for i in range(9)]
_CLOCK = [[tuple(2 * SPAN * (c << 2 * perm[i]) for perm, _ in _IMAGES) for c in range(4)] for i in range(9)]
_PAIR = {(o, a, b): _link_row(o - 1, a, b) for o in (1, 2) for a in range(9) for b in range(a + 1, 9)}
_SWAP = {(a, b): _link_row(2, a, b) for a in range(9) for b in range(a + 1, 9)}


def canonical(s):
    """(key, swapped): the least (encode(image), flag) over images(s), summed from the tables above. The turn is the
    lowest bit of a key and the colour swap flips it, so no two images of different flags share a key."""
    rows = [_TURN[s.turn]]
    rows += [_TOKEN[i][v] for i, v in enumerate(s.board) if v]
    rows += [_PAIR[q] for q in s.pairs]
    rows += [_SWAP[q] for q in s.swaps]
    if s.clocks != ZEROS:
        rows += [_CLOCK[i][c] for i, c in enumerate(s.clocks) if c]
    keys = [sum(column) for column in zip(*rows)]
    low = min(keys)
    return low, keys.index(low) >= len(SYMS)


# --- the bridge to the engine -----------------------------------------------------------------------------------

_TOKEN_CHAR = "0XOXO"  # indexed by board value


def _splits(s):
    """For each link, its two equally likely contents, each as ((square, char), (square, char))."""
    for o, a, b in s.pairs:
        yield ((a, _TOKEN_CHAR[o]), (b, "0")), ((a, "0"), (b, _TOKEN_CHAR[o]))
    for a, b in s.swaps:
        yield ((a, "X"), (b, "O")), ((a, "O"), (b, "X"))


def branches(s):
    """{branch string over "0XO": probability}: the engine's branches of s, 2^links of them, equally likely."""
    out = {"".join(_TOKEN_CHAR[v] for v in s.board): ONE}
    for options in _splits(s):
        out = {_fill(br, fills): p / 2 for br, p in out.items() for fills in options}
    return out


def _fill(branch, fills):
    for i, ch in fills:
        branch = branch[:i] + ch + branch[i + 1:]
    return branch


def _square_value(e, i, linked):
    """The board value of square i of engine state e."""
    seen = {br[i] for br, _ in e.amps}
    if linked or seen == {"0"}:
        if e.measured[i]:
            raise ValueError(f"square {i} is marked measured but holds no token")
        return 0
    if len(seen) != 1:
        raise ValueError(f"square {i} is in superposition outside a link")
    return E.CHAR.index(seen.pop()) + (0 if e.measured[i] else 2)


def from_engine(e, cfg):
    """The compact state of engine state e. Clocks are kept only with decoherence on and only on unmeasured non-empty
    squares (the harness leaves stale counters elsewhere, and payloads carry none). ValueError unless the result's
    branches reproduce e's within TOL: that rejects Q3 states, chains of links and anything else outside the model."""
    pairs, swaps, linked = [], [], set()
    for group in E.components(e):
        if len(group) != 2:
            raise ValueError(f"squares {group} are linked beyond a pair: outside the model")
        a, b = group
        held = {(br[a], br[b]) for br, _ in e.amps}
        if held == {("X", "O"), ("O", "X")}:
            swaps.append((a, b))
        elif held in ({("X", "0"), ("0", "X")}, {("O", "0"), ("0", "O")}):
            pairs.append((1 if ("X", "0") in held else 2, a, b))
        else:
            raise ValueError(f"squares {group} hold {sorted(held)}: neither a pair nor a swap")
        linked |= set(group)
    board = tuple(_square_value(e, i, i in linked) for i in range(9))
    clocks = tuple(e.counters[i] if cfg.decoherence and (board[i] > 2 or i in linked) else 0 for i in range(9))
    if not all(0 <= c <= 3 for c in clocks):
        raise ValueError(f"decoherence counters {clocks} are outside 0-3")
    c = CState(board, tuple(sorted(pairs)), tuple(sorted(swaps)), e.turn, clocks)
    got = {}
    for br, amp in e.amps:
        got[br] = got.get(br, 0.0) + abs(amp) ** 2
    want = branches(c)
    if got.keys() != want.keys() or any(abs(got[br] - float(p)) > TOL for br, p in want.items()):
        raise ValueError("the branches of the state are not those of a model state")
    return c
