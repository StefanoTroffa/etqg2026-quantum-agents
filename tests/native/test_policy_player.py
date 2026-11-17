"""The policy-value transformer as the player (search/cheqqers_policy_player.py, track 33).

- the native encoding of a position (NativeEnv.encode) is the trainer's input (flat_encoding of the self-play record),
  bit for bit, on positions of native self-play;
- the player returns one of the legal moves and plays through the hub's agent spec `pv=<model.pt>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]

torch = pytest.importorskip("torch")

from models.cheqqers_policy_value import CheqqersPolicyValue, flat_encoding, model_config  # noqa: E402
from search import native_cheqqers as nc  # noqa: E402

pytestmark = pytest.mark.skipif(not nc.available(), reason="native library not available")


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    out = tmp_path_factory.mktemp("sp") / "games.jsonl"
    nc.selfplay(out, games=3, seed=7, time_limit_ms=20, max_depth=2, write_state=True)
    return [json.loads(line) for line in open(out) if '"state"' in line and '"result"' not in line]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = CheqqersPolicyValue(d_model=32, nhead=4, layers=1, ff=64)
    path = tmp_path_factory.mktemp("m") / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model_config(model)}, path)
    return path


def test_native_encoding_is_the_training_input(records: list[dict]) -> None:
    assert len(records) > 20
    for rec in records:
        env = nc.NativeEnv.from_flat(*rec["state"])
        assert env.encode() == flat_encoding(rec)


def test_player_returns_its_argmax_legal_move(records: list[dict], tiny_model: Path) -> None:
    from search.cheqqers_policy_player import PolicyPlayer

    player = PolicyPlayer(tiny_model, device="cpu")
    for rec in records[:30]:
        env = nc.NativeEnv.from_flat(*rec["state"])
        legal = env.legal_moves()
        move = player.choose(env, legal)
        assert move in legal
        logits, wdl = player.evaluate([env], [legal])
        assert len(logits[0]) == len(legal) and abs(sum(wdl[0]) - 1) < 1e-5
        if len(legal) > 1:
            assert move == legal[max(range(len(legal)), key=lambda i: (logits[0][i], -i))]


def test_hub_builds_and_runs_a_pv_agent(tiny_model: Path) -> None:
    from arena.leiden_sparring_hub import _build_agent_from_spec
    from env.cheqqers_official import OfficialCheqqersEnv

    agent = _build_agent_from_spec(f"PV:expecti_heuristic@pv={tiny_model}", "cheqqers")
    assert agent.policy_model == str(tiny_model)
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=2)
    legal = env.get_legal_moves()
    move = agent._compute_move(env, {}, legal)
    assert list(move) in [list(m) for m in legal]
