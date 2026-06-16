# Contributing to Dataset Tools

## Layout

| Path | Responsibility |
|---|---|
| `joint_to_ee/` | EE conversion package — one module per concern |
| `viewer/` | Browser viewer + wizard |
| `viewer/js/` | ES modules (see VIEWER_README.md for the module map) |
| `server.py` | stdlib HTTP server + JSON API |
| `dataset-wizard.py` | Merge / enrich / compress / upload pipeline |
| `tests/` | pytest suite (pure-math + integration) |

## Running tests

```bash
pip install pytest
cd /path/to/lerobot
python -m pytest tools/tests -q
```

Pure-math tests (`test_orientation.py`, `test_poses.py`, `test_representations.py`) run without placo. The integration test (`test_pipeline_integration.py`) skips automatically if placo is unavailable.

## Verifying viewer changes

There is no JS build step or JS test runner. Verify manually:

```bash
python tools/server.py --cache cache
# Open http://localhost:8080/viewer in a browser
```

Check the browser console for JS errors. The viewer uses native ES modules — no bundler.

## Conventions

- **Quaternion storage**: `[qw, qx, qy, qz]`. scipy's `Rotation.as_quat()` returns `[qx,qy,qz,qw]` — convert at the boundary (see `orientation.py:_to_scipy` / `_canonical_wxyz`).
- **EE vectors**: 8-dim `[x,y,z,qw,qx,qy,qz,gripper]`; legacy datasets may be 7-dim (viewer handles both).
- **Joint layout** (arms-first, fixed): state[0..5]=left joints, state[6]=left gripper, state[7..12]=right joints, state[13]=right gripper.
- **Commit style**: short imperative subject, 50-char target. Co-author line required for AI contributions.
