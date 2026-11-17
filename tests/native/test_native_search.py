"""Gate 2 of the native Cheqqers engine: at a fixed depth the native search is ExpectiCheqqersPlayer.

Without a time limit, on the six benchmark positions (tests/native/fixtures/bench_positions_2026-11-13.jsonl, the
positions of ~/work/_scratch/qg_cpp/bench_search.py) and on positions of the 44 arena games in
tests/compliance/fixtures: the same move, the same number of nodes (`_nodes_visited`), the same last finished depth
and the same score to the last bit, with the default options and with quantum_plies=2; on fewer positions also with
quiescence, other evaluation weights (linear and logit) and a smaller extension budget. The larger sweep (hundreds
of positions, depth 4 everywhere): scripts/native_identity_sweep.py.

Skips when the library cannot be built or loaded here.
"""

import struct

import pytest

from search import native_cheqqers as nc

if not nc.available():
    pytest.skip("native Cheqqers library unavailable (no compiler, or it does not load)", allow_module_level=True)

from native_identity import ARENA_SAMPLE, BENCH, ODD_WEIGHTS, load_games, replay_positions  # noqa: E402
from search.expecti_cheqqers import ExpectiCheqqersPlayer  # noqa: E402

BENCH_MOVES = [[6, 3], [7, 4], [10, 7], [12, 15], [0, 3], [4, 11]]
BENCH_NODES_D4_Q2 = 8067


def bench_positions():
    out = []
    for game in load_games(BENCH):
        for ply, env, move in replay_positions(game, upto=game["benchPly"]):
            if move is None:
                out.append(env)
    return out


def arena_positions(every: int, offset: int):
    out = []
    for game in load_games(ARENA_SAMPLE):
        for ply, env, move in replay_positions(game):
            if ply % every == offset and not env.is_terminal() and len(env.get_legal_moves()) > 1:
                out.append(env.clone())
    return out


def same_search(env, depth: int, **options) -> list[str]:
    player = ExpectiCheqqersPlayer(max_depth=depth, time_limit_ms=1e9, **options)
    py_move = player.get_move(env.clone())
    native = nc.best_move(env.clone(), time_limit_ms=1e9, max_depth=depth, **options)
    problems = []
    if list(py_move) != list(native.move):
        problems.append(f"move {py_move} != {native.move}")
    if player._nodes_visited != native.nodes:
        problems.append(f"nodes {player._nodes_visited} != {native.nodes}")
    if player.last_depth != native.last_depth:
        problems.append(f"depth {player.last_depth} != {native.last_depth}")
    a = struct.pack("d", player.last_score) if player.last_score is not None else None
    b = struct.pack("d", native.last_score) if native.last_score is not None else None
    if a != b:
        problems.append(f"score {player.last_score!r} != {native.last_score!r}")
    return problems


BENCH_POSITIONS = bench_positions()
ARENA_POSITIONS = arena_positions(every=25, offset=11)


def test_the_benchmark_is_the_one_measured_on_13_november():
    total = 0
    for env, expected in zip(BENCH_POSITIONS, BENCH_MOVES):
        result = nc.best_move(env.clone(), time_limit_ms=1e9, max_depth=4, quantum_plies=2)
        assert list(result.move) == expected
        total += result.nodes
    assert total == BENCH_NODES_D4_Q2


@pytest.mark.parametrize("k", range(6))
@pytest.mark.parametrize("options", [{"quantum_plies": 2}, {}], ids=["quantum_plies=2", "default"])
def test_benchmark_positions_search_as_python(k, options):
    assert same_search(BENCH_POSITIONS[k], 4 if options else 3, **options) == []


@pytest.mark.parametrize("k", range(len(ARENA_POSITIONS)))
def test_arena_positions_search_as_python(k):
    env = ARENA_POSITIONS[k]
    problems = same_search(env, 4, quantum_plies=2) + same_search(env, 3)
    assert problems == []


OPTION_SETS = [
    {"quiescence": True, "quantum_plies": 2},
    {"eval_weights": ODD_WEIGHTS[0], "quantum_plies": 2},
    {"eval_weights": ODD_WEIGHTS[1], "quantum_plies": 2},
    {"max_capture_extension": 1, "quiescence": True},
    {"star1_pruning": False, "quantum_plies": 1},
    {"use_cache": False, "quantum_plies": 2},
]


@pytest.mark.parametrize("options", OPTION_SETS, ids=lambda o: ",".join(f"{k}={'w' if k == 'eval_weights' else v}" for k, v in o.items()))
def test_search_options_as_python(options):
    problems = []
    for k, env in enumerate(BENCH_POSITIONS + ARENA_POSITIONS[::4]):
        problems += [f"position {k}: {p}" for p in same_search(env, 3, **options)]
    assert problems == []


def test_a_time_limit_is_honoured():
    env = BENCH_POSITIONS[3]
    result = nc.best_move(env.clone(), time_limit_ms=200, max_depth=30)
    assert result.move in env.get_legal_moves()
    assert result.elapsed_ms < 200 * 0.9 + 60
    assert result.last_depth >= 1


def test_one_legal_move_is_answered_without_a_search():
    env = BENCH_POSITIONS[0]
    only = env.get_legal_moves()[:1]
    result = nc.best_move(env.clone(), time_limit_ms=1e9, max_depth=4, available_moves=only)
    assert result.move == only[0] and result.nodes == 0


def test_a_value_network_is_refused():
    with pytest.raises(nc.NativeUnsupported):
        nc.best_move(BENCH_POSITIONS[0].clone(), model=object(), nn_blend=0.25)


def test_self_play_records_the_moves_the_python_search_plays(tmp_path):
    import json

    out = tmp_path / "selfplay.jsonl"
    written = nc.selfplay(out, games=2, seed=11, time_limit_ms=1e9, max_depth=3, quantum_plies=2, write_state=True)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    positions = [line for line in lines if "move" in line]
    assert written == len(positions) > 10 and sum("result" in line for line in lines) == 2
    for rec in positions[::3]:
        env = nc.import_env(*rec["state"])
        assert env.get_legal_moves() == rec["moves"]
        player = ExpectiCheqqersPlayer(max_depth=3, time_limit_ms=1e9, quantum_plies=2)
        assert list(player.get_move(env.clone())) == rec["move"]
        assert player._nodes_visited == rec["nodes"]
