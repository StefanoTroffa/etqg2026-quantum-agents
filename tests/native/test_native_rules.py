"""Gate 1 of the native Cheqqers engine: the rules are the Python's, bit for bit.

At every position of the arena games in tests/compliance/fixtures and of seeded random games (every quantumness,
6x6 and 8x8), with the per-position cache off (how the simulator runs) and on (how the search runs): the same legal
moves in the same order, the same outcome probabilities to the last bit with the same child positions, the same
finished state and winner, the same eval_features and evaluate_state (default and two other sets of weights). The
native replay of each arena game, with the arena's recorded measurements, stays equal to the Python's after every
move. The whole 599-game arena file of 13 November: scripts/native_identity_sweep.py.

Skips when the library cannot be built or loaded here.
"""

import pytest

from search import native_cheqqers as nc

if not nc.available():
    pytest.skip("native Cheqqers library unavailable (no compiler, or it does not load)", allow_module_level=True)

from native_identity import (  # noqa: E402
    ARENA_SAMPLE,
    BENCH,
    compare_position,
    load_games,
    random_positions,
    replay_positions,
    same_export,
    with_cache,
)

ARENA_GAMES = load_games(ARENA_SAMPLE)
BENCH_GAMES = load_games(BENCH)


@pytest.mark.parametrize("game", ARENA_GAMES + BENCH_GAMES, ids=lambda g: g["id"])
def test_every_arena_position_has_the_pythons_rules(game):
    problems = []
    for ply, env, _ in replay_positions(game):
        for label, position in (("", env), (" (cache on)", with_cache(env))):
            problems += [f"ply {ply}{label}: {p}" for p in compare_position(position)]
    assert problems == []


@pytest.mark.parametrize("game", ARENA_GAMES + BENCH_GAMES, ids=lambda g: g["id"])
def test_the_native_replay_of_an_arena_game_stays_the_pythons(game):
    native = None
    for ply, env, move in replay_positions(game):
        if native is None:
            native = nc.NativeEnv.from_env(env)
        assert same_export(nc.export_env(env), native.export()), f"ply {ply}"
        if move is not None:
            native.apply_move(list(move["move"]), forced_measurements=list(move["outcomes"]))


RANDOM_GAMES = [(seed, 6, q) for q in (0, 1, 2, 3) for seed in range(6)] + [(100 + seed, 8, 2) for seed in range(3)]


@pytest.mark.parametrize("seed,board_size,quantumness", RANDOM_GAMES)
def test_random_play_positions_have_the_pythons_rules(seed, board_size, quantumness):
    problems = []
    for k, env in enumerate(random_positions(seed, board_size, quantumness, max_plies=80)):
        for label, position in (("", env), (" (cache on)", with_cache(env))):
            problems += [f"position {k}{label}: {p}" for p in compare_position(position)]
    assert problems == []


def test_a_position_survives_the_round_trip_through_the_flat_state():
    game = ARENA_GAMES[0]
    for _, env, _ in replay_positions(game):
        flat = nc.export_env(env)
        back = nc.import_env(*nc.NativeEnv.from_flat(*flat).export())
        assert same_export(nc.export_env(back), flat)
        assert back.get_legal_moves() == env.get_legal_moves()


def test_a_tournament_payload_is_read_by_the_python_rules():
    # from_tournament_json can set a flag on the class (a server that fixed multi-jumps); leave it as it was found
    cls = type(next(replay_positions(BENCH_GAMES[0]))[1])
    saved = cls.__dict__.get("_detected_server_multijump_fix", None)
    try:
        _, env, _ = next(p for p in replay_positions(BENCH_GAMES[0], upto=BENCH_GAMES[0]["benchPly"]) if p[2] is None)
        payload = env.to_tournament_payload()
        result = nc.best_move(payload, time_limit_ms=60000, max_depth=2)
        assert result.move in payload["legalMoves"]
        reread = cls.from_tournament_json(payload)
        native = nc.NativeEnv.from_env(reread)
        assert same_export(nc.export_env(reread), native.export())
        assert native.legal_moves() == reread.get_legal_moves()
    finally:
        if saved is None:
            if "_detected_server_multijump_fix" in cls.__dict__:
                delattr(cls, "_detected_server_multijump_fix")
        else:
            cls._detected_server_multijump_fix = saved
