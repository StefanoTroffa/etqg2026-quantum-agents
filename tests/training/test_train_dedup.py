"""Deduplicated training targets (src/training/train_cheqqers_transformer.py, Dedup; track 33).

- identical positions (same encoding and legal moves) become one row; the policy target is the count of moves played
  there, or, if any copy carries a search label (value -1), the count of labels only; the value target counts results;
- the row weight is the label weight for labelled positions and 1 otherwise;
- batches rebuild the soft targets in the order of each position's moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]

torch = pytest.importorskip("torch")

from training.train_cheqqers_transformer import Arrays, Dedup, position_keys  # noqa: E402

A = [[1, 2], [3, 4], [5, 6, 7]]
B = [[8, 9], [10, 11]]


def _npz(path: Path, rows: list[tuple[int, list[list[int]], int, int]]) -> Path:
    """rows: (encoding id, moves, target index, value)."""
    xs, mst, msq, mlen, tgt, val = [], [0], [], [], [], []
    for enc, moves, t, v in rows:
        x = np.zeros(144, dtype=np.float32)
        x[enc] = 1.0
        xs.append(x)
        for m in moves:
            msq.append(m + [0] * (3 - len(m)))
            mlen.append(len(m))
        mst.append(len(mlen))
        tgt.append(t)
        val.append(v)
    np.savez(path, x=np.stack(xs), mstart=np.asarray(mst, dtype=np.int64), msq=np.asarray(msq, dtype=np.int8),
             mlen=np.asarray(mlen, dtype=np.int8), target=np.asarray(tgt, dtype=np.int32),
             value=np.asarray(val, dtype=np.int8), game=np.zeros(len(rows), dtype=np.int32),
             stats=np.asarray('{"games": 1, "excluded_positions": 0, "unfinished": 0, "ply_limit": 0}'))
    return path


@pytest.fixture()
def dedup(tmp_path: Path) -> tuple[Arrays, Dedup, dict[int, int]]:
    played = _npz(tmp_path / "p.npz", [(0, A, 0, 0), (0, A, 1, 2), (0, A, 0, 1), (1, B, 1, 0), (2, B, 0, 0)])
    labels = _npz(tmp_path / "l.npz", [(1, B, 0, -1), (1, B, 0, -1)])
    a = Arrays([str(played), str(labels)])
    d = Dedup(a, position_keys(a), label_weight=3.0)
    by_enc = {int(np.argmax(a.x[r])): u for u, r in enumerate(d.rep)}
    return a, d, by_enc


def _pol(d: Dedup, u: int) -> list[float]:
    return d.pol[d.pstart[u]:d.pstart[u + 1]].tolist()


def test_one_row_per_position(dedup: tuple) -> None:
    _, d, by_enc = dedup
    assert d.u == 3 and sorted(by_enc) == [0, 1, 2]


def test_policy_target_counts_played_moves_and_labels_override(dedup: tuple) -> None:
    _, d, e = dedup
    assert _pol(d, e[0]) == [2.0, 1.0, 0.0]
    assert _pol(d, e[1]) == [2.0, 0.0]  # the two labels, not the move played there
    assert _pol(d, e[2]) == [1.0, 0.0]


def test_value_target_and_weights(dedup: tuple) -> None:
    _, d, e = dedup
    assert d.val[e[0]].tolist() == [1.0, 1.0, 1.0] and d.val[e[1]].tolist() == [1.0, 0.0, 0.0]
    assert d.weight[e[1]] == 3.0 and d.weight[e[0]] == 1.0 and d.weight[e[2]] == 1.0


def test_batch_rebuilds_targets_in_move_order(dedup: tuple) -> None:
    _, d, e = dedup
    _, _, mi, ml, pt, _, _ = d.batch(np.array([e[0], e[1]]), "cpu")
    assert pt[0, :3].tolist() == [2.0, 1.0, 0.0] and pt[1].tolist() == [2.0, 0.0, 0.0]
    assert ml[0].tolist() == [2, 2, 3] and ml[1].tolist() == [2, 2, 0]
    assert mi[0, 2].tolist() == [5, 6, 7]


def test_soft_policy_targets_are_summed_over_copies(tmp_path: Path) -> None:
    """Gumbel self-play shards carry a policy target per move ("policy"): copies of a position add their vectors."""
    path = _npz(tmp_path / "s.npz", [(0, A, 0, 0), (0, A, 2, 2)])
    blob = dict(np.load(path))
    blob["policy"] = np.asarray([0.5, 0.25, 0.25, 0.1, 0.1, 0.8], dtype=np.float32)
    np.savez(path, **blob)
    a = Arrays([str(path)])
    d = Dedup(a, position_keys(a), label_weight=3.0)
    assert d.u == 1 and _pol(d, 0) == pytest.approx([0.6, 0.35, 1.05])
