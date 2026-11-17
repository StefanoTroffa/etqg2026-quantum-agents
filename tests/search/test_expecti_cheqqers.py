"""ExpectiCheqqersPlayer: regressions found by the pre-flight against the organisers' harness.

Position of 10 November 2026 (pre-flight game 6, lost to the random player): our last piece on square 7, an enemy
piece on square 6, us to move. Splitting to 10 and 11 lets the enemy capture one half, which loses the game with
probability 1/2; the plain move to 11 is safe.
"""

import json
from pathlib import Path

import pytest

from env.cheqqers_official import OfficialCheqqersEnv
from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move

SAFE_MOVE = (7, 11)
SPLIT_NEXT_TO_ENEMY = (7, 10, 11)


def last_piece_next_to_enemy() -> OfficialCheqqersEnv:
    board = [{"piece": None, "probability": 0} for _ in range(18)]
    board[6] = {"piece": "white", "probability": 1.0}
    board[7] = {"piece": "black", "probability": 1.0}
    payload = {"player": "black", "currentTurn": "black", "board": board, "settings": {"boardSize": 6, "quantumness": 2}}
    return OfficialCheqqersEnv.from_tournament_json(payload)


def chosen_move(**player_kwargs) -> tuple:
    player = ExpectiCheqqersPlayer(time_limit_ms=60000.0, **player_kwargs)
    return canonicalize_cheqqers_move(player.get_move(last_piece_next_to_enemy()))


# ── Transposition table and forced-capture extension ───────────────────────────────────────────────────────────


class TestTranspositionTableDepth:
    def test_get_move_keeps_last_piece_whole_when_split_can_be_captured(self):
        assert chosen_move(max_depth=5) == SAFE_MOVE

    @pytest.mark.parametrize("max_depth", [3, 4, 5, 6])
    def test_get_move_matches_search_without_table_when_table_is_on(self, max_depth):
        assert chosen_move(max_depth=max_depth) == chosen_move(max_depth=max_depth, max_tt_size=0)


# ── Positions the simulator cannot expand ──────────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


class TestSimulatorLimit:
    """Game 30 of the 100-game series of 10 November 2026: five entangled queens. Some moves there need an
    entangled system above the engine's limit. Until 13 November the simulator raised, which aborted the search
    and, in `server.py`, dropped the request into the June code path. The harness itself swallows the engine's
    error and leaves the same side to move; the simulator now does the same (checked against the harness in
    `tests/compliance/test_cheqqers_harness_differential.py`), so the search sees such a move for what it is."""

    def test_get_move_returns_a_legal_move_when_a_branch_exceeds_the_engine_limit(self):
        payload = json.loads((FIXTURES / "cheqqers_entangled_queens_payload.json").read_text())
        env = OfficialCheqqersEnv.from_tournament_json(payload)
        player = ExpectiCheqqersPlayer(max_depth=5, time_limit_ms=1200.0)
        move = player.get_move(env, available_moves=payload["legalMoves"])
        assert move in payload["legalMoves"]
        assert player.unexpanded_branches == 0

    def test_refused_move_keeps_the_side_to_move_when_the_limit_is_exceeded(self, monkeypatch):
        from env.cheqqers_official import JointSystem

        # two split pieces already entangle four squares; with a limit of four squares a further split is refused
        monkeypatch.setattr(JointSystem, "max_dimension_limit", 16)
        env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
        refused = 0
        for _ in range(40):
            if env.done:
                break
            mover = env.turn
            splits = [m for m in env.get_legal_moves() if len(m) == 3]
            move = (splits or env.get_legal_moves())[0]
            for _, child in env.get_move_outcomes(move):  # must not raise
                assert child.turn in (mover, mover.other())
            entangled_before = sum(1 for sq in env.squares if sq.entangled_system is not None)
            env.step(move, return_state=False)
            if env.turn == mover and env.multi_jump_square is None and not env.done and entangled_before >= 4:
                refused += 1
                break
        assert refused == 1

    def test_search_still_survives_when_the_simulator_raises_the_limit_error(self, monkeypatch):
        # the safety net in the search stays: a simulator that does raise must not abort get_move()
        import search.expecti_cheqqers as module

        payload = json.loads((FIXTURES / "cheqqers_opening_black_payload.json").read_text())
        env = OfficialCheqqersEnv.from_tournament_json(payload)
        real = module.get_cheqqers_outcomes
        calls = {"n": 0}

        def sometimes_refuses(position, move):
            calls["n"] += 1
            if calls["n"] % 5 == 0:
                raise RuntimeError("QuantumLimitExceeded: dimension 65536 > 32768")
            return real(position, move)

        monkeypatch.setattr(module, "get_cheqqers_outcomes", sometimes_refuses)
        player = ExpectiCheqqersPlayer(max_depth=3, time_limit_ms=60000.0)
        move = player.get_move(env, available_moves=payload["legalMoves"])
        assert move in payload["legalMoves"]
        assert player.unexpanded_branches > 0


# ── One player object across moves and colours ─────────────────────────────────────────────────────────────────


class TestPlayerReuse:
    """Game 19 of the 100-game series of 10 November 2026: the series keeps one player for all games. Its table
    held scores from White's side of the previous game; as Black it read them as its own and played its worst
    first move in 39 ms."""

    def test_get_move_matches_a_fresh_player_when_the_player_was_used_as_the_other_colour(self):
        payload = json.loads((FIXTURES / "cheqqers_opening_black_payload.json").read_text())
        as_black = OfficialCheqqersEnv.from_tournament_json(payload)
        fresh = ExpectiCheqqersPlayer(max_depth=4, time_limit_ms=120000.0)
        expected = canonicalize_cheqqers_move(fresh.get_move(as_black, available_moves=payload["legalMoves"]))

        # White's first move of an earlier game, searched one ply deeper: its table now holds this very position
        # and its replies, scored from White's side.
        reused = ExpectiCheqqersPlayer(max_depth=5, time_limit_ms=120000.0)
        reused.get_move(OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2))
        reused.max_depth = 4
        as_black = OfficialCheqqersEnv.from_tournament_json(payload)
        got = canonicalize_cheqqers_move(reused.get_move(as_black, available_moves=payload["legalMoves"]))

        assert got == expected


# ── Quiescence: no static evaluation while a capture is pending (Track-Q, step 1) ──────────────────────────────


ARENA_GAMES = [json.loads(line) for line in
               (Path(__file__).parents[1] / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl")
               .read_text().splitlines() if line.strip()]


def arena_positions(plies=(6, 9, 12, 16), games=10) -> list[OfficialCheqqersEnv]:
    """Positions with a choice of moves from recorded arena games."""
    from arena.arena_records import replay

    out = []
    for game in ARENA_GAMES[:games]:
        for ply in plies:
            if ply < len(game["moves"]):
                env = replay({**game, "moves": game["moves"][:ply]})
                if not env.done and len(env.get_legal_moves()) > 1:
                    out.append(env)
    return out


def search_counting_leaves(env: OfficialCheqqersEnv, depth: int, **player_kwargs) -> dict:
    """Search `env` to a fixed depth; count the positions evaluated statically, and those among them in which the
    side to move had a capture to make."""
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=600000.0, **player_kwargs)
    seen = {"leaves": 0, "capture_pending": 0, "shallowest_pending": None}
    evaluate = player._evaluate_leaf

    def counting(position, root_color, ply):
        if not position.done:
            seen["leaves"] += 1
            if len(position.get_take_moves()) > 0:
                seen["capture_pending"] += 1
                if seen["shallowest_pending"] is None or ply < seen["shallowest_pending"]:
                    seen["shallowest_pending"] = ply
        return evaluate(position, root_color, ply)

    player._evaluate_leaf = counting
    seen["move"] = canonicalize_cheqqers_move(player.get_move(env.clone()))
    seen["nodes"] = player._nodes_visited
    return seen


class TestQuiescence:
    def test_plain_search_evaluates_positions_with_a_capture_pending(self):
        # the weakness this step removes: without it the test below would prove nothing
        pending = sum(search_counting_leaves(env, 2)["capture_pending"] for env in arena_positions())
        assert pending > 50

    def test_no_position_with_a_capture_pending_is_evaluated_before_the_budget_is_spent_when_quiescence_is_on(self):
        # Every extension adds a ply, and the first iteration of the search is one ply deep: a line that has spent
        # its budget of 8 is at least 1 + 8 plies long.
        # (Chains of captures on barely present pieces can be longer than any budget: it has to be finite.)
        positions = arena_positions()
        assert len(positions) >= 20
        for env in positions:
            seen = search_counting_leaves(env, 2, quiescence=True, max_capture_extension=8)
            assert seen["leaves"] > 0
            assert seen["shallowest_pending"] is None or seen["shallowest_pending"] >= 9

    def test_quiescence_stops_when_its_extension_budget_is_spent(self):
        positions = arena_positions(plies=(9,), games=6)
        plain = sum(search_counting_leaves(env, 2)["capture_pending"] for env in positions)
        short = sum(search_counting_leaves(env, 2, quiescence=True, max_capture_extension=1)["capture_pending"]
                    for env in positions)
        assert 0 < short < plain

    def test_default_player_searches_without_quiescence(self):
        assert ExpectiCheqqersPlayer().quiescence is False
        for env in arena_positions(plies=(9,), games=6):
            default = search_counting_leaves(env, 2)
            explicit = search_counting_leaves(env, 2, quiescence=False)
            assert (default["move"], default["nodes"]) == (explicit["move"], explicit["nodes"])


# ── Three-square moves only near the root (Track-Q, step 2) ────────────────────────────────────────────────────


def late_positions(min_three_square=6, games=30) -> list[OfficialCheqqersEnv]:
    """Recorded positions late in a game with many three-square moves among the legal ones (queens on the board)."""
    from arena.arena_records import replay

    out = []
    for game in ARENA_GAMES[:games]:
        for ply in range(24, len(game["moves"]), 6):
            env = replay({**game, "moves": game["moves"][:ply]})
            if not env.done and sum(len(m) == 3 for m in env.get_legal_moves()) >= min_three_square:
                out.append(env)
                break
    return out


def search_logging_moves(env: OfficialCheqqersEnv, depth: int, **player_kwargs) -> dict:
    """Search `env` to a fixed depth; the deepest ply at which a three-square move was tried, by ply of the
    position it leads to (1 = a move of the root), and the count of positions visited."""
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=600000.0, **player_kwargs)
    seen = {"three_square_plies": set()}
    chance = player._evaluate_chance_node

    def logging(**kwargs):
        if len(canonicalize_cheqqers_move(kwargs["move"])) == 3:
            seen["three_square_plies"].add(kwargs["ply"])
        return chance(**kwargs)

    player._evaluate_chance_node = logging
    seen["move"] = canonicalize_cheqqers_move(player.get_move(env.clone()))
    seen["nodes"] = player._nodes_visited
    return seen


class TestQuantumMovesNearTheRoot:
    def test_fixture_has_late_positions_with_many_three_square_moves(self):
        assert len(late_positions()) >= 5

    def test_search_tries_three_square_moves_at_every_ply_when_not_limited(self):
        assert ExpectiCheqqersPlayer().quantum_plies is None
        for env in late_positions()[:3]:
            assert max(search_logging_moves(env, 3)["three_square_plies"]) >= 3  # deeper where a line is extended

    @pytest.mark.parametrize("plies", [1, 2])
    def test_search_tries_three_square_moves_only_in_its_first_plies_when_limited(self, plies):
        for env in late_positions()[:5]:
            seen = search_logging_moves(env, 3, quantum_plies=plies)
            assert seen["three_square_plies"] and max(seen["three_square_plies"]) == plies

    def test_limited_search_visits_fewer_positions_at_the_same_depth(self):
        positions = late_positions()[:5]
        full = sum(search_logging_moves(env, 3)["nodes"] for env in positions)
        limited = sum(search_logging_moves(env, 3, quantum_plies=1)["nodes"] for env in positions)
        assert limited < 0.5 * full

    def test_limited_search_is_the_full_search_when_the_limit_is_beyond_its_depth(self):
        for env in late_positions()[:3]:
            full = search_logging_moves(env, 2)
            limited = search_logging_moves(env, 2, quantum_plies=5)
            assert limited["move"] == full["move"]


# ── A queen's long slide is not a capture ──────────────────────────────────────────────────────────────────────


def queen_position(with_enemy_on_the_diagonal: bool) -> OfficialCheqqersEnv:
    """A white queen on square 15 with its long diagonal (12, 10, 7, 5, 2) free, a black man out of reach on 0,
    and, if asked, a black man on 10 that the queen must capture."""
    board = [{"piece": None, "probability": 0} for _ in range(18)]
    board[15] = {"piece": "white", "probability": 1.0, "crowned": True}
    board[0] = {"piece": "black", "probability": 1.0}
    if with_enemy_on_the_diagonal:
        board[10] = {"piece": "black", "probability": 1.0}
    payload = {"player": "white", "currentTurn": "white", "board": board, "settings": {"boardSize": 6, "quantumness": 2}}
    return OfficialCheqqersEnv.from_tournament_json(payload)


class TestQueenSlidesAreNotCaptures:
    def test_long_slide_of_a_queen_is_not_a_capture_when_nothing_is_jumped(self):
        env = queen_position(with_enemy_on_the_diagonal=False)
        assert [15, 2] in env.get_legal_moves() and env.get_take_moves() == []
        player = ExpectiCheqqersPlayer()
        assert [player._is_capture_move(env, move) for move in env.get_legal_moves()] == [False] * len(env.get_legal_moves())

    def test_every_legal_move_is_a_capture_when_a_capture_is_pending(self):
        env = queen_position(with_enemy_on_the_diagonal=True)
        legal = env.get_legal_moves()
        assert [15, 7] in legal and [15, 12] not in legal
        player = ExpectiCheqqersPlayer()
        assert all(player._is_capture_move(env, move) for move in legal)

    def test_capture_pending_is_the_same_with_and_without_the_cache(self):
        for with_enemy in (False, True):
            env = queen_position(with_enemy)
            cached = env.clone()
            cached.cache_enabled = True
            assert env.capture_pending() is with_enemy
            assert cached.capture_pending() is with_enemy
            assert cached.capture_pending() is with_enemy  # the second answer comes from the cache

    def test_capture_pending_is_recomputed_after_a_move(self):
        env = queen_position(with_enemy_on_the_diagonal=True)
        env.cache_enabled = True
        assert env.capture_pending() is True
        env.step([15, 7], return_state=False)
        fresh = env.clone()
        fresh.cache_enabled = False
        assert env.capture_pending() is fresh.capture_pending()
