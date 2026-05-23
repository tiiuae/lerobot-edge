#!/usr/bin/env python3
"""dataset-wizard.py

Pipeline for managing and enriching robotic datasets.

Stages (in order):
  1. conversion   - Convert datasets from v2.1 to v3.0 format
  2. merge        - Load and merge individual datasets into one
  3. ee_conversion - Add EE poses + action representations via joint-to-ee.py
  4. upload       - Compress and upload the merged dataset to an SFTP server

Use --start-from / --stop-at to run any subset of the pipeline.

Examples:
  python tools/dataset-wizard.py
  python tools/dataset-wizard.py --config tools/config.yaml
  python tools/dataset-wizard.py --start-from merge --stop-at merge
  python tools/dataset-wizard.py --start-from ee_conversion --stop-at ee_conversion --ee-frame arm
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import paramiko
import yaml
from tqdm import tqdm

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

# joint_to_ee lives alongside this script
sys.path.insert(0, str(Path(__file__).parent))
import joint_to_ee  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

STAGES = ["conversion", "merge", "ee_conversion", "upload"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Dataset wizard — merge, enrich, and upload LeRobot datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml",
                   help="Path to YAML config file (default: tools/config.yaml)")
    p.add_argument("--start-from", choices=STAGES, default=None,
                   help="Override start stage from config file")
    p.add_argument("--stop-at",    choices=STAGES, default=None,
                   help="Override stop stage from config file")
    p.add_argument("--ee-frame",   choices=["arm", "robot_base"], default=None,
                   help="EE reference frame (overrides config ee_frame)")
    p.add_argument("--ee-include-action", action="store_true", default=None,
                   help="Also compute EE and representations for action joints")
    return p.parse_args()


args = parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

move_dataset_repo_ids = cfg["datasets"]
base_dataset_root     = Path(cfg["base_path"]).expanduser()
merged_repo_id        = cfg["merged_name"]
output_directory      = base_dataset_root / merged_repo_id

start_from = args.start_from or cfg.get("start_from", "conversion")
stop_at    = args.stop_at    or cfg.get("stop_at",    "upload")


def should_run(stage: str) -> bool:
    return STAGES.index(start_from) <= STAGES.index(stage) <= STAGES.index(stop_at)


GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"


# ── Stage 1: Conversion ───────────────────────────────────────────────────────

if should_run("conversion"):
    print("=" * 60)
    print("Stage 1: Dataset format conversion (v2.1 → v3.0)")
    print("=" * 60)
    conversion_ok     = []
    conversion_failed = []

    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        old_path     = base_dataset_root / f"{repo_id}_old"

        if dataset_path.is_dir():
            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                version = json.loads(info_path.read_text()).get("codebase_version", "unknown")
                if version != "v2.1":
                    conversion_ok.append((repo_id, f"already converted ({version})"))
                    continue
        elif old_path.is_dir():
            conversion_ok.append((repo_id, "already converted (_old present)"))
            continue
        else:
            print(f"  Warning: directory not found for {repo_id}, skipping.")
            continue

        print(f"  Converting {repo_id} …")
        try:
            convert_dataset(
                repo_id=f"{base_dataset_root.name}/{repo_id}",
                root=str(base_dataset_root.parent),
                push_to_hub=False,
            )
            conversion_ok.append((repo_id, "converted"))
        except Exception as e:
            conversion_failed.append((repo_id, dataset_path, e))

    print("\n── Conversion summary ──────────────────────────────────────────")
    for repo_id, label in conversion_ok:
        print(f"  {GREEN}✔ {repo_id}  ({label}){RESET}")
    for repo_id, _, e in conversion_failed:
        print(f"  {RED}✘ {repo_id}  {type(e).__name__}: {e}{RESET}")
    print("────────────────────────────────────────────────────────────────\n")

    if conversion_failed:
        answer = input(
            f"{len(conversion_failed)} dataset(s) failed. "
            f"Proceed with {len(conversion_ok)} succeeded? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborting.")
            raise SystemExit(0)
else:
    print("Skipping conversion stage.")


# ── Stage 2: Merge ────────────────────────────────────────────────────────────

if should_run("merge"):
    print("=" * 60)
    print("Stage 2: Merging datasets")
    print("=" * 60)
    datasets_to_merge = []
    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        if dataset_path.is_dir():
            print(f"  Loading {repo_id} …")
            ds = LeRobotDataset(repo_id=repo_id, root=dataset_path)
            print(f"    {ds.num_episodes} episodes, {ds.num_frames} frames")
            datasets_to_merge.append(ds)
        else:
            print(f"  Warning: {dataset_path} not found, skipping.")

    if output_directory.exists():
        print(f"\nRemoving existing output directory: {output_directory}")
        shutil.rmtree(output_directory)

    print(f"\nMerging {len(datasets_to_merge)} datasets → {merged_repo_id} …")
    merged = merge_datasets(
        datasets=datasets_to_merge,
        output_repo_id=merged_repo_id,
        output_dir=output_directory,
    )
    print(f"\n{GREEN}✔ Merged dataset: {merged.root}{RESET}")
    print(f"  Total episodes : {merged.meta.total_episodes}")
    print(f"  Total frames   : {merged.meta.total_frames}\n")
else:
    print("Skipping merge stage.")


# ── Stage 3: EE Conversion ────────────────────────────────────────────────────

if should_run("ee_conversion"):
    ee_cfg           = cfg.get("joint_to_ee", {})
    ref_frame        = args.ee_frame or ee_cfg.get("ee_frame") or cfg.get("ee_frame")
    include_action   = args.ee_include_action or ee_cfg.get("include_action", False) \
                       or cfg.get("ee_include_action", False)

    if ref_frame is None:
        print(
            "Skipping EE conversion: no ee_frame set. "
            "Use --ee-frame arm|robot_base or set ee_frame in config.yaml."
        )
    else:
        print("=" * 60)
        print("Stage 3: EE conversion + action representations")
        print("=" * 60)
        joint_to_ee.process_dataset(
            dataset_root=output_directory,
            output_root=output_directory,   # in-place
            ref_frame=ref_frame,
            include_action=include_action,
        )
else:
    print("Skipping EE conversion stage.")


# ── Stage 4: Upload ───────────────────────────────────────────────────────────

if should_run("upload"):
    print("=" * 60)
    print("Stage 4: Compress and upload")
    print("=" * 60)
    zip_path = output_directory.with_suffix(".zip")
    print(f"Compressing {output_directory} → {zip_path} …")
    shutil.make_archive(str(output_directory), "zip", str(output_directory))
    print(f"Compressed: {zip_path}\n")

    sftp_cfg    = cfg.get("sftp", {})
    hostname    = str(sftp_cfg["hostname"])
    port        = int(sftp_cfg.get("port", 22))
    username    = str(sftp_cfg["username"])
    password    = str(sftp_cfg["password"])
    remote_path = sftp_cfg.get("remote_path")

    if remote_path is None:
        raise ValueError("sftp.remote_path is not set in config.yaml.")
    if not str(remote_path).endswith("/"):
        remote_path = str(remote_path) + "/"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, password)
    sftp = ssh.open_sftp()
    print("SFTP connection established.")

    remote_file = remote_path + zip_path.name
    file_size   = os.path.getsize(str(zip_path))
    print(f"Uploading {zip_path.name} → {remote_file} …")
    with tqdm(total=file_size, unit="B", unit_scale=True, desc=zip_path.name) as pbar:
        last = {"sent": 0}

        def _cb(transferred: int, _total: int):
            pbar.update(transferred - last["sent"])
            last["sent"] = transferred

        sftp.put(str(zip_path), remote_file, callback=_cb)

    print(f"\n{GREEN}✔ Uploaded to {remote_file}{RESET}")
else:
    print("Skipping upload stage.")
