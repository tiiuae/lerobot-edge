"""Dataset enrichment orchestration with atomic in-place swap."""
import json
import shutil
import tempfile
from pathlib import Path

import time

import pyarrow.parquet as pq
from rich.console import Console

from . import constants as C
from .enrich import enrich_table
from .kinematics import make_kinematics
from .metadata import update_info_json

console = Console()


def _guard_columns(include_joint_repr: bool, rot_repr: str) -> set:
    guard = {"observation.ee_left", "observation.ee_right",
             "action.ee_left", "action.ee_right"}
    if include_joint_repr:
        guard |= {"action.delta", "action.relative"}
    for side in ("left", "right"):
        for kind in ("delta", "relative"):
            guard.add(f"action.ee_{side}.{kind}")
    return guard


def process_dataset(dataset_root: Path, output_root: Path, ref_frame: str,
                    include_joint_repr: bool = True, rot_repr: str = "both") -> None:
    in_place = output_root.resolve() == dataset_root.resolve()

    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        existing = set(json.loads(info_path.read_text()).get("features", {}).keys())
        collision = _guard_columns(include_joint_repr, rot_repr) & existing
        if collision:
            raise ValueError(
                f"Features already exist: {sorted(collision)}.\n"
                "Remove them first or start from a freshly merged dataset.")

    console.print("  joint layout      : [dim]arms-first (ignores mislabeled metadata)[/]")
    console.print(f"  ref frame         : [bold]{ref_frame}[/]")
    console.print(f"  include joint repr: [bold]{include_joint_repr}[/]")
    console.print(f"  orientation repr  : [bold]{rot_repr}[/]")

    left_mount = C.LEFT_MOUNT_XYZ if ref_frame == "robot_base" else None
    right_mount = C.RIGHT_MOUNT_XYZ if ref_frame == "robot_base" else None

    console.print(f"  building placo kinematics from [bold]{C.WXAI_FOLLOWER_URDF.name}[/] …")
    kin = make_kinematics()

    pq_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")

    if in_place:
        tmp_root = Path(tempfile.mkdtemp(
            prefix=f"_{dataset_root.name}_ee_tmp_", dir=dataset_root.parent))
        shutil.copytree(dataset_root, tmp_root, dirs_exist_ok=True)
        work_root = tmp_root
    else:
        shutil.copytree(dataset_root, output_root, dirs_exist_ok=False)
        work_root = output_root

    n = len(pq_files)
    console.print(f"  Enriching {n} parquet file(s)…")
    t0 = time.monotonic()
    last_pct = -1
    for i, pq_src in enumerate(pq_files, 1):
        pq_dst = work_root / pq_src.relative_to(dataset_root)
        tbl = pq.read_table(pq_src)
        tbl = enrich_table(tbl, kin, left_mount, right_mount,
                           include_joint_repr=include_joint_repr, rot_repr=rot_repr)
        pq.write_table(tbl, pq_dst)
        pct = int(i / n * 100)
        if pct != last_pct and (pct % 10 == 0 or i == n):
            elapsed = time.monotonic() - t0
            if i < n and elapsed > 0:
                eta_sec = elapsed / i * (n - i)
                eta_str = f"  eta {int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
            else:
                eta_str = ""
            console.print(f"  [{i}/{n}] {pct}%{eta_str}")
            last_pct = pct

    update_info_json(work_root / "meta" / "info.json",
                     include_joint_repr=include_joint_repr, ref_frame=ref_frame,
                     rot_repr=rot_repr)

    if in_place:
        shutil.rmtree(dataset_root)
        work_root.rename(dataset_root)
        console.print(f"  [green]✔[/]  Dataset enriched in-place at: [bold]{dataset_root}[/]")
    else:
        console.print(f"  [green]✔[/]  Enriched dataset written to: [bold]{work_root}[/]")
