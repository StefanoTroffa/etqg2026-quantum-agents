"""The teacher: the tables solve.py wrote for one configuration, asked about any state of it.

Teacher(cfg) loads the five tables of cfg, one per draw weight in solve.W. For a state and a w it gives the stored entry
(lookup for a compact.CState, entry for an engine.EState) and every legal move's value and entry (moves, move_values).
A move's entry is not stored: it is solve.move_entry's one-ply sum over the stored entries of the move's outcomes, so it
exists for every state of the tables, on the line of optimal play or off it. Each move is applied to the caller's own
state, so the moves come out in the caller's orientation; only the lookup of an outcome goes through the colour swap.

What the tables cannot answer raises, never a guess: ValueError for a w outside W; OutOfConfiguration (also a
ValueError) for a finished game, a state the compact model cannot hold, or one that no game of the configuration
reaches; RuntimeError at load for a table that is missing, corrupt or of an old shape, belongs to another configuration
or weight, or was made by other rules code than the one on disk now.

Values are exact at every w. At w = 0 and w = 1 the outcomes worth w are value-equal, so how an entry splits its mass
among them, and its plies, follow the stored policy's tie-break ("shorter if equal"), not the game: the Q0 root stores
(0, 1, 0, 0, 6) at w = 0 although Q0 is a draw. Read distributions only at w = 1/4, 1/2, 3/4.

Tables solved with manual measurement off (cfg.manual False) list no measure moves, so their moves differ from the
tournament harness's legal moves; the tournament configuration has manual measurement on.
"""

import pickle
from pathlib import Path

from tiqtaqtoe import compact, solve
from tiqtaqtoe.engine import CONTINUE
from tiqtaqtoe.solve import DATA, W


class OutOfConfiguration(ValueError):
    """A state the tables cannot answer: the game is over, the compact model cannot hold the state, or no game of the
    configuration reaches it, so nothing is stored for it."""


class _Rows:
    """The five tables as the one table solve.entry_at() and solve.move_entry() read: [key] is the five entries of the
    key in W order, taken from the dicts as they are at that moment, so an edit of a Teacher's tables shows at the next
    query. KeyError if a dict lacks the key; `key in rows` is true when none does. Read-only."""

    def __init__(self, tables):
        self._tables = tables

    def __getitem__(self, key):
        return tuple(table[key] for table in self._tables)

    def __contains__(self, key):
        return all(key in table for table in self._tables)


def _weight_index(w):
    """The index in W of the draw weight equal to w, by value: 0, 1, 0.5 and Fraction(1, 2) all match. ValueError if w
    is none of them."""
    try:
        return W.index(w)
    except ValueError:
        raise ValueError(f"draw weight {w!r} is not one of {', '.join(map(str, W))}") from None


def _load(cfg, table_dir, w, rules):
    """The entries {key: Entry} of cfg at draw weight w, from the file solve.write_tables() put in table_dir.
    RuntimeError, naming the file, if it is missing, corrupt or of an old shape, holds another configuration or weight,
    or was made by other rules code than `rules` (a rules_hash())."""
    path = table_dir / cfg.name / f"w={w.numerator}-{w.denominator}.pkl"
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"{path} is missing: solve {cfg.name} first (python3 -m tiqtaqtoe.solve)") from None
    except Exception as err:  # a truncated or garbage file raises any of several errors
        raise RuntimeError(f"{path} cannot be unpickled ({err!r}): solve {cfg.name} again "
                           f"(python3 -m tiqtaqtoe.solve)") from err
    try:
        found_cfg, found_w, found_rules, entries = blob["config"], blob["w"], blob["rules_hash"], blob["entries"]
    except (TypeError, KeyError, IndexError) as err:
        raise RuntimeError(f"{path} has an old or damaged shape ({err!r}): solve {cfg.name} again "
                           f"(python3 -m tiqtaqtoe.solve)") from err
    if found_cfg != cfg or found_w != w:
        raise RuntimeError(f"{path} holds the table of {found_cfg} at w = {found_w}, not of {cfg} at w = {w}")
    if found_rules != rules:
        raise RuntimeError(f"{path} was made by other rules code (its hash {found_rules}, now {rules}): "
                           f"solve {cfg.name} again (python3 -m tiqtaqtoe.solve)")
    return entries


class Teacher:
    """The tables of one configuration, loaded. cfg is the configuration; tables is a tuple of five dicts
    {canonical key: Entry}, one per draw weight in W order, read as they stand at each query.

    Engine states carry the decoherence counters, which tournament payloads do not. With decoherence on (d1-d3) the
    caller supplies the counters it tracked: E.from_round_state(payload, counters=...), which raises without them.
    With decoherence off pass (0,) * 9."""

    def __init__(self, cfg, table_dir=DATA):
        """Load the five tables of cfg from table_dir. RuntimeError if one is missing, belongs to another configuration
        or weight, or carries a rules_hash other than solve.rules_hash(): made by other rules code, so solve again."""
        rules = solve.rules_hash()
        self.cfg = cfg
        self.tables = tuple(_load(cfg, Path(table_dir), w, rules) for w in W)
        self._rows = _Rows(self.tables)

    def _check(self, s, w):
        """The index of w in W, once the state s is known to be an unfinished one of the tables. ValueError if w is not
        in W; OutOfConfiguration if the game in s is over (no finished game is ever answered, whatever the tables hold)
        or the tables lack its class."""
        wi = _weight_index(w)
        if compact.winner(s) != CONTINUE:
            raise OutOfConfiguration(f"the game in {s!r} is over: the tables answer for unfinished games only")
        if compact.canonical(s)[0] not in self._rows:
            raise OutOfConfiguration(f"{s!r} is not a state of {self.cfg.name}: the tables hold nothing for it")
        return wi

    def lookup(self, s, w):
        """The stored Entry of the compact state s at draw weight w, in s's own colours: the probabilities of an X win,
        an O win, a tie and a draw and the expected plies left, under the stored policy. ValueError if w is not in W;
        OutOfConfiguration if the game in s is over or the tables lack the state."""
        return solve.entry_at(self._rows, s, self._check(s, w))

    def moves(self, s, w):
        """[(move, value, entry)] for every legal move of the compact state s, in harness order (compact.moves): the
        entry the game has after the move (solve.move_entry: its outcomes by chance, one ply longer, in s's colours)
        and its value at w, X's payoff as an exact Fraction. The mover (s.turn, 1 X and 2 O) wants the highest value as
        X and the lowest as O. Errors as in lookup()."""
        wi = self._check(s, w)
        moves = compact.moves(s, self.cfg)
        entries = [solve.move_entry(self._rows, s, move, self.cfg, wi) for move in moves]
        return [(move, solve.value(e, W[wi]), e) for move, e in zip(moves, entries)]

    def _compact(self, e):
        """The compact state of the engine state e (compact.from_engine, which snaps float noise within 1e-9 and ignores
        stale decoherence counters). OutOfConfiguration, with the model's reason, if the model cannot hold e."""
        try:
            return compact.from_engine(e, self.cfg)
        except ValueError as err:
            raise OutOfConfiguration(f"the engine state is outside the model of {self.cfg.name}: {err}") from err

    def entry(self, e, w):
        """lookup() for the engine state e."""
        return self.lookup(self._compact(e), w)

    def move_values(self, e, w):
        """moves() for the engine state e: e's own moves, in e's orientation."""
        return self.moves(self._compact(e), w)
