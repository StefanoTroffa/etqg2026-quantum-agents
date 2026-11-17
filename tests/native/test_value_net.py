"""The value network at the native search's leaves (native/cheqqers/value_net.{h,cpp}).

- encoding: the C++ input vector equals encode_position(record) flattened (src/models/cheqqers_policy_value.py), the
  record built from the Python engine's own pieces, presences and eval_features, on every position of the arena
  fixture games (and on native self-play records, which are what the network was trained on);
- forward pass: the C++ logit equals a float64 numpy forward of the same JSON on the same inputs (|diff| < 1e-5);
- search: with a network the search returns a legal move, and at depth 1 (no pruning, no extension) each root score
  is the probability-weighted 900 tanh(logit / 2) of the outcomes, from the root player's side;
- without a network nothing changes: value_net="" and an unset CHEQQERS_VALUE_NET give the same result bits as the
  call without the option (the other identity tests keep checking the default path against Python).

The real network is used if present (CHEQQERS_VALUE_NET_TEST or ~/work/_scratch/qg_cpp/value_net/value_mlp.json);
a seeded random network of the same shape always runs.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from env.cheqqers_official import PieceColor  # noqa: E402
from models.cheqqers_policy_value import encode_position  # noqa: E402
from native_identity import ARENA_SAMPLE, BENCH, load_games, replay_positions  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402

pytestmark = pytest.mark.skipif(not nc.available(), reason="native library unavailable")

REAL_NET = Path(os.getenv("CHEQQERS_VALUE_NET_TEST",
                          os.path.expanduser("~/work/_scratch/qg_cpp/value_net/value_mlp.json")))


def _random_net(path: Path, seed: int = 7, dims=(144, 64, 32, 1)) -> Path:
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        layers.append({"weight": (rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i])).tolist(),
                       "bias": (0.1 * rng.standard_normal(dims[i + 1])).tolist()})
    path.write_text(json.dumps({"inputs": dims[0], "activation": "relu", "layers": layers}))
    return path


@pytest.fixture(scope="module")
def nets(tmp_path_factory) -> list[nc.ValueNet]:
    out = [nc.ValueNet(_random_net(tmp_path_factory.mktemp("net") / "random_mlp.json"))]
    if REAL_NET.exists():
        out.append(nc.ValueNet(REAL_NET))
    return out


def numpy_forward(net: nc.ValueNet, x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, np.float64)
    for k, (w, b) in enumerate(net.layers):
        h = h @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)
        if k + 1 < len(net.layers):
            h = np.maximum(h, 0.0)
    return h[..., 0]


def python_record(env) -> dict:
    """The self-play record fields encode_position reads, from the Python engine (cq_selfplay's meaning)."""
    occ = env._occ_probs()
    pieces = []
    for i in range(env.total_squares):
        prop = env.piece_properties.get(i)
        pieces.append(0 if not prop else 1 + (2 if prop["color"] == PieceColor.BLACK else 0) + (1 if prop["crowned"] else 0))
    return {"turn": int(env.turn), "multi_jump": -1 if env.multi_jump_square is None else env.multi_jump_square,
            "features": list(env.eval_features(env.turn)), "presence": [float(p) for p in occ], "pieces": pieces}


def flat(record: dict) -> list[float]:
    sq, gl = encode_position(record)
    return [v for s in sq for v in s] + gl


@pytest.fixture(scope="module")
def positions() -> list:
    out = []
    for game in load_games(ARENA_SAMPLE):
        for _, env, _ in replay_positions(game):
            out.append(env.clone())
    return out


def test_encoding_and_forward_on_arena_positions(nets, positions):
    assert len(positions) >= 1000
    xs = []
    for env in positions:
        expect = flat(python_record(env))
        native = nc.NativeEnv.from_env(env)
        for net in nets:
            got, logit = net.encode(native)
            assert struct.pack(f"{len(got)}d", *got) == struct.pack(f"{len(expect)}d", *expect)
        xs.append(expect)
    x = np.asarray(xs)
    for net in nets:
        ref = numpy_forward(net, x)
        got = np.asarray(net.forward(xs))
        assert np.max(np.abs(got - ref)) < 1e-5
        # the encode entry point's logit is the same forward pass
        enc = np.asarray([net.encode(nc.NativeEnv.from_env(e))[1] for e in positions[:200]])
        assert np.array_equal(enc, got[:200])


def test_encoding_on_native_selfplay_records(nets, tmp_path):
    out = tmp_path / "sp.jsonl"
    nc.selfplay(out, games=3, seed=11, time_limit_ms=1e9, max_depth=2, write_state=True)
    n = 0
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        if "result" in rec:
            continue
        native = nc.NativeEnv.from_flat(*rec["state"])
        for net in nets:
            got, _ = net.encode(native)
            assert got == flat(rec)
        n += 1
    assert n > 20


def _bench_envs():
    out = []
    for game in load_games(BENCH):
        for _, env, move in replay_positions(game, upto=game["benchPly"]):
            if move is None:
                out.append(nc.NativeEnv.from_env(env))
    return out


def test_search_with_net_returns_a_legal_move(nets):
    for net in nets:
        for env in _bench_envs():
            res = nc.best_move(env, time_limit_ms=1e9, max_depth=3, quantum_plies=2, value_net=net)
            assert res.error is None
            assert list(res.move) in env.legal_moves()
            assert res.nodes > 0 and res.last_depth == 3
        res = nc.best_move(_bench_envs()[0], time_limit_ms=1e9, max_depth=3, quantum_plies=2, value_net=net.path)
        assert res.move is not None


def test_depth_one_root_scores_are_the_nets_values(nets):
    for net in nets:
        for env in _bench_envs():
            root = int(env.turn)
            res = nc.best_move(env, time_limit_ms=1e9, max_depth=1, star1_pruning=False, max_capture_extension=0,
                               value_net=net)
            assert res.root_scores
            for move, score in res.root_scores:
                outs = env.move_outcomes(move)
                tot = sum(p for p, _ in outs)
                expect = 0.0
                for p, child in outs:
                    if p / tot <= 1e-9:
                        continue
                    if child.terminal:
                        pytest.skip("terminal child: the leaf keeps the +-1000 scale")
                    v = 900.0 * np.tanh(0.5 * net.encode(child)[1])
                    expect += (p / tot) * (v if int(child.turn) == root else -v)
                assert abs(score - expect) < 1e-9


def test_without_net_nothing_changes(monkeypatch):
    monkeypatch.delenv("CHEQQERS_VALUE_NET", raising=False)
    for env in _bench_envs():
        a = nc.best_move(env, time_limit_ms=1e9, max_depth=4, quantum_plies=2)
        b = nc.best_move(env, time_limit_ms=1e9, max_depth=4, quantum_plies=2, value_net="")
        assert (a.move, a.nodes, a.last_depth) == (b.move, b.nodes, b.last_depth)
        assert struct.pack("d", a.last_score) == struct.pack("d", b.last_score)


def test_env_switch_selects_the_net(monkeypatch, nets):
    net = nets[0]
    env = _bench_envs()[1]
    want = nc.best_move(env, time_limit_ms=1e9, max_depth=3, quantum_plies=2, value_net=net)
    monkeypatch.setenv("CHEQQERS_VALUE_NET", str(net.path))
    got = nc.best_move(env, time_limit_ms=1e9, max_depth=3, quantum_plies=2)
    assert (got.move, got.nodes, got.last_score) == (want.move, want.nodes, want.last_score)
    off = nc.best_move(env, time_limit_ms=1e9, max_depth=3, quantum_plies=2, value_net="")
    assert off.last_score != want.last_score


def test_hub_spec_net_key():
    from arena.leiden_sparring_hub import _split_budget

    _, budget = _split_budget("A:expecti_heuristic@cpp=1,weights=config/w.json,net=/x/value_mlp.json")
    assert budget["net"] == "/x/value_mlp.json"
