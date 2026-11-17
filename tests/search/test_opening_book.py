"""The Cheqqers opening book (Track-O): a table from positions to the move a long search chose.

Lookup must be exact (the stored move as the harness lists it, or nothing), the player must answer from the book
without searching and search as before otherwise, and `build` must cover every position the book's side can meet
in its first decisions. `root_values`, the judge used to compare a book move with the move the agent would play,
is checked against `get_move`.
"""

import json
from pathlib import Path

import pytest

from arena.arena_records import replay
from env.cheqqers_official import OfficialCheqqersEnv
from search.expecti_cheqqers import ExpectiCheqqersPlayer, canonicalize_cheqqers_move
from search.opening_book import OpeningBook, book_from_environment, build

FIXTURES = Path(__file__).parent / "fixtures"
PAYLOAD = json.loads((FIXTURES / "cheqqers_opening_black_payload.json").read_text())
ARENA = [json.loads(line) for line in
         (Path(__file__).parents[1] / "compliance" / "fixtures" / "arena_cheqqers_2026-11-12_sample.jsonl")
         .read_text().splitlines() if line.strip()]


def start() -> OfficialCheqqersEnv:
    return OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)


def payload_position() -> tuple[OfficialCheqqersEnv, list]:
    return OfficialCheqqersEnv.from_tournament_json(PAYLOAD), PAYLOAD["legalMoves"]


# ── Lookup ─────────────────────────────────────────────────────────────────────────────────────────────────────


class TestLookup:
    def test_lookup_returns_the_stored_move_as_the_harness_lists_it_when_position_is_in_the_book(self):
        env, legal = payload_position()
        wanted = legal[-1]
        book = OpeningBook({OpeningBook.key(env): [int(x) for x in wanted]})
        found = book.lookup(env, legal)
        assert found == wanted
        assert any(found is move for move in legal)

    def test_lookup_returns_nothing_when_position_is_not_in_the_book(self):
        env, legal = payload_position()
        assert OpeningBook({}).lookup(env, legal) is None
        assert OpeningBook({OpeningBook.key(start()): [15, 12]}).lookup(env, legal) is None

    def test_lookup_returns_nothing_when_the_stored_move_is_not_legal(self):
        env, legal = payload_position()
        book = OpeningBook({OpeningBook.key(env): [0, 17]})
        assert book.lookup(env, legal) is None

    def test_key_tells_an_open_capture_sequence_from_the_same_board_without_one(self):
        env = start()
        other = env.clone()
        other.multi_jump_square = 7
        assert OpeningBook.key(env) != OpeningBook.key(other)

    def test_key_is_the_same_for_a_position_from_a_payload_and_from_the_simulator(self):
        env, _ = payload_position()
        game = next(g for g in ARENA if len(g["moves"]) >= 1)
        replayed = replay({**game, "moves": []})
        assert OpeningBook.key(replayed) == OpeningBook.key(start())
        # the payload position is one ply into a game: the same key as the simulator after that ply, if any
        # arena game began with the move that leads to it
        after_one = {OpeningBook.key(replay({**g, "moves": g["moves"][:1]})) for g in ARENA}
        assert OpeningBook.key(env) in after_one


# ── The file ───────────────────────────────────────────────────────────────────────────────────────────────────


class TestBookFile:
    def test_book_is_the_same_after_saving_and_reading(self, tmp_path):
        book = OpeningBook({"T:0|1WM1": [15, 12], "T:1|2BM1": [3, 6, 7]}, {"seconds": 20.0})
        book.save(tmp_path / "book.json")
        again = OpeningBook.from_json(tmp_path / "book.json")
        assert again.moves == book.moves and again.settings == book.settings and len(again) == 2

    def test_reading_fails_when_the_file_is_not_a_book(self, tmp_path):
        (tmp_path / "weights.json").write_text(json.dumps({"men": 1.0}))
        with pytest.raises(ValueError):
            OpeningBook.from_json(tmp_path / "weights.json")

    def test_environment_names_no_book_when_variable_is_unset(self, monkeypatch):
        monkeypatch.delenv("CHEQQERS_OPENING_BOOK", raising=False)
        assert book_from_environment() is None

    def test_environment_book_is_read_when_variable_names_a_file(self, monkeypatch, tmp_path):
        OpeningBook({"T:0|1WM1": [15, 12]}).save(tmp_path / "book.json")
        monkeypatch.setenv("CHEQQERS_OPENING_BOOK", str(tmp_path / "book.json"))
        assert len(book_from_environment()) == 1

    def test_environment_book_fails_loudly_when_the_file_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHEQQERS_OPENING_BOOK", str(tmp_path / "nowhere.json"))
        with pytest.raises(OSError):
            book_from_environment()


# ── The player ─────────────────────────────────────────────────────────────────────────────────────────────────


class TestPlayerWithBook:
    def test_player_answers_from_the_book_without_searching_when_position_is_in_it(self):
        env, legal = payload_position()
        wanted = legal[-1]
        book = OpeningBook({OpeningBook.key(env): [int(x) for x in wanted]})
        player = ExpectiCheqqersPlayer(max_depth=3, time_limit_ms=600000.0, opening_book=book)
        assert player.get_move(env, available_moves=legal) == wanted
        assert player.from_book is True and player._nodes_visited == 0

    def test_player_searches_as_without_a_book_when_position_is_not_in_it(self):
        env, legal = payload_position()
        book = OpeningBook({OpeningBook.key(start()): [15, 12]})
        with_book = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0, opening_book=book)
        without = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0)
        assert with_book.get_move(env, available_moves=legal) == without.get_move(env, available_moves=legal)
        assert with_book.from_book is False and with_book._nodes_visited == without._nodes_visited > 0

    def test_player_reports_the_depth_and_score_of_its_last_search(self):
        env, legal = payload_position()
        player = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0)
        player.get_move(env, available_moves=legal)
        assert player.last_depth == 2
        assert -1000.0 <= player.last_score <= 1000.0


# ── Root values ────────────────────────────────────────────────────────────────────────────────────────────────


class TestRootValues:
    def test_root_values_cover_every_legal_move_and_the_chosen_move_has_the_highest(self):
        checked = 0
        for game in ARENA[:6]:
            env = replay({**game, "moves": game["moves"][:6]})
            legal = env.get_legal_moves()
            if env.done or len(legal) < 2:
                continue
            player = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0)
            chosen = canonicalize_cheqqers_move(player.get_move(env.clone()))
            depth, values = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0).root_values(env)
            assert depth == 2
            assert set(values) == {canonicalize_cheqqers_move(m) for m in legal}
            assert values[chosen] == pytest.approx(max(values.values()), abs=1e-9)
            assert player.last_score == pytest.approx(values[chosen], abs=1e-9)
            checked += 1
        assert checked >= 4

    def test_root_values_stop_at_the_last_finished_depth_when_time_runs_out(self):
        env = replay({**ARENA[0], "moves": ARENA[0]["moves"][:8]})
        depth, values = ExpectiCheqqersPlayer(max_depth=12, time_limit_ms=400.0).root_values(env)
        assert 1 <= depth < 12 and len(values) == len(env.get_legal_moves())


# ── Building ───────────────────────────────────────────────────────────────────────────────────────────────────


class TestBuild:
    def test_book_holds_a_legal_move_for_every_first_decision_of_either_colour(self):
        book = build(own_moves=1, seconds=600.0, depth=1, workers=1)
        root = start()
        assert book.lookup(root, root.get_legal_moves()) is not None
        replies = 0
        for move in root.get_legal_moves():
            for probability, child in root.get_move_outcomes(list(move)):
                legal = child.get_legal_moves()
                if child.done or len(legal) < 2:
                    continue
                assert book.lookup(child, legal) is not None
                replies += 1
        assert replies >= 5
        assert book.settings["own_moves"] == 1 and book.settings["depth"] == 1

    def test_book_follows_its_own_move_and_every_reply_when_it_goes_two_decisions_deep(self):
        book = build(own_moves=2, seconds=600.0, depth=1, workers=1)
        root = start()
        first = book.lookup(root, root.get_legal_moves())
        covered = 0
        for _, after_ours in root.get_move_outcomes(list(first)):
            for reply in after_ours.get_legal_moves():
                for _, position in after_ours.get_move_outcomes(list(reply)):
                    legal = position.get_legal_moves()
                    if position.done or len(legal) < 2 or position.turn != root.turn:
                        continue
                    assert book.lookup(position, legal) is not None
                    covered += 1
        # most replies to a first move force a capture (one legal move, nothing to store): few positions remain
        assert covered >= 1
        assert len(book) > len(build(own_moves=1, seconds=600.0, depth=1, workers=1))

    def test_book_stores_the_move_the_same_search_chooses(self):
        book = build(own_moves=1, seconds=600.0, depth=2, workers=1)
        root = start()
        chosen = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0).get_move(root.clone())
        assert book.lookup(root, root.get_legal_moves()) == list(chosen)


class TestRootValuesOfChosenMoves:
    def test_root_values_of_a_subset_are_the_values_those_moves_have_among_all(self):
        env = replay({**ARENA[0], "moves": ARENA[0]["moves"][:6]})
        legal = env.get_legal_moves()
        assert len(legal) >= 3
        depth, everything = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0).root_values(env)
        subset = [legal[0], legal[-1]]
        same_depth, some = ExpectiCheqqersPlayer(max_depth=2, time_limit_ms=600000.0).root_values(env, moves=subset)
        assert same_depth == depth == 2
        assert set(some) == {canonicalize_cheqqers_move(m) for m in subset}
        for key, value in some.items():
            assert value == pytest.approx(everything[key], abs=1e-9)

    def test_root_values_ignore_a_move_that_is_not_legal(self):
        env = replay({**ARENA[0], "moves": ARENA[0]["moves"][:6]})
        legal = env.get_legal_moves()
        _, some = ExpectiCheqqersPlayer(max_depth=1, time_limit_ms=600000.0).root_values(env, moves=[legal[0], [0, 17]])
        assert set(some) == {canonicalize_cheqqers_move(legal[0])}


def test_arena_positions_are_our_decisions_replayed_from_records(tmp_path) -> None:
    """Our side's positions (more than one legal move) before max_ply, replayed from the recorded moves; the other
    team's turns and games without our team are left out."""
    import json
    from search.opening_book import OpeningBook, arena_positions, start_position

    env = start_position()
    moves, ours_keys = [], []
    for _ in range(6):
        legal = env.get_legal_moves()
        by = "white" if env.turn.name == "WHITE" else "black"
        if by == "white" and len(legal) > 1:
            ours_keys.append(OpeningBook.key(env))
        move = [m for m in legal if len(m) == 2][0]  # a plain move: nothing is measured
        assert env.apply_move(list(move)) == 0
        moves.append({"by": by, "move": list(move), "outcomes": []})
    game = {"id": "g1", "settings": {"boardSize": 6, "startRows": 2, "quantumness": 2}, "moves": moves,
            "players": {"white": {"name": "Us"}, "black": {"name": "Them"}}}
    other = dict(game, id="g2", players={"white": {"name": "A"}, "black": {"name": "B"}})
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(game) + "\n" + json.dumps(other) + "\n")
    got = [OpeningBook.key(e) for e in arena_positions(path, "Us", 6)]
    assert sorted(got) == sorted(set(ours_keys)) and len(got) == 3
    assert arena_positions(path, "Us", 1) and len(arena_positions(path, "Us", 1)) == 1
    assert arena_positions(path, "Nobody", 6) == []
