"""End-to-end enrichment on a tiny synthetic v3.0 dataset. Skips if placo is absent."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

placo = pytest.importorskip("placo")  # noqa: F841 — skip whole module if missing

from joint_to_ee.pipeline import process_dataset


def _make_dataset(root, n_frames=5):
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    rng = np.random.default_rng(0)
    state = rng.uniform(-0.5, 0.5, size=(n_frames, 19)).astype(np.float32)
    action = rng.uniform(-0.5, 0.5, size=(n_frames, 16)).astype(np.float32)
    tbl = pa.table({
        "episode_index": pa.array([0] * n_frames, pa.int64()),
        "frame_index": pa.array(list(range(n_frames)), pa.int64()),
        "observation.state": pa.array(state.tolist(), pa.list_(pa.float32())),
        "action": pa.array(action.tolist(), pa.list_(pa.float32())),
    })
    pq.write_table(tbl, root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [19], "names": None},
            "action": {"dtype": "float32", "shape": [16], "names": None},
        },
    }))


def test_process_dataset_adds_expected_columns(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds)
    process_dataset(ds, ds, ref_frame="robot_base", include_joint_repr=True, rot_repr="both")

    feats = json.loads((ds / "meta" / "info.json").read_text())["features"]
    for col, dim in [
        ("observation.ee_left", 8), ("action.ee_left", 8),
        ("action.ee_left.delta", 8), ("action.ee_left.delta.rotvec", 7),
        ("action.ee_left.relative", 8), ("action.ee_left.relative.rotvec", 7),
        ("action.delta", 16), ("action.relative", 16),
    ]:
        assert col in feats, f"missing feature {col}"
        assert feats[col]["shape"] == [dim], f"{col} shape {feats[col]['shape']} != {dim}"

    tbl = pq.read_table(next((ds / "data").rglob("*.parquet")))
    row = tbl.slice(0, 1).to_pylist()[0]
    assert len(row["observation.ee_left"]) == 8
    # frame 0: delta vs obs EE == relative (both reference obs EE at t=0)
    assert np.allclose(row["action.ee_left.delta"], row["action.ee_left.relative"], atol=1e-5)
    # gripper component normalized into [0,1]
    assert 0.0 <= row["observation.ee_left"][7] <= 1.0


def test_double_enrichment_guard(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds)
    process_dataset(ds, ds, ref_frame="arm", include_joint_repr=True, rot_repr="quat")
    with pytest.raises(ValueError, match="already exist"):
        process_dataset(ds, ds, ref_frame="arm", include_joint_repr=True, rot_repr="quat")
