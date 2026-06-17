#!/usr/bin/env python3
"""dataset-wizard.py

Pipeline for managing and enriching robotic datasets.

Stages (in order):
  1. conversion    - Convert datasets from v2.1 to v3.0 format
  2. merge         - Load and merge individual datasets into one
  3. ee_conversion - Add EE poses + action representations via joint_to_ee.py
  4. upload        - Compress and upload the merged dataset to an SFTP server

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
import zipfile
from pathlib import Path

import paramiko
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

# joint_to_ee lives alongside this script
sys.path.insert(0, str(Path(__file__).parent))
import joint_to_ee  # noqa: E402

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

STAGES = ["conversion", "merge", "ee_conversion", "compress", "upload"]


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
    p.add_argument("--ee-rot-repr", choices=["quat", "rotvec", "both"], default=None,
                   help="EE orientation representation for delta/relative (default: both)")
    p.add_argument("--no-ee-joint-repr", action="store_true", default=False,
                   help="Skip joint-space action representations (action.delta, action.relative)")
    p.add_argument("--skip-ee", action="store_true", default=False,
                   help="Skip EE conversion stage regardless of start/stop range")
    return p.parse_args()


args = parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

move_dataset_repo_ids = cfg["datasets"]
base_dataset_root     = Path(cfg["base_path"]).expanduser()
merged_repo_id        = cfg.get("output_dir") or cfg.get("merged_name", "output_dataset")
output_directory      = base_dataset_root / merged_repo_id

start_from = args.start_from or cfg.get("start_from", "conversion")
stop_at    = args.stop_at    or cfg.get("stop_at",    "upload")
skip_ee    = args.skip_ee


def should_run(stage: str) -> bool:
    if stage == "ee_conversion" and skip_ee:
        return False
    return STAGES.index(start_from) <= STAGES.index(stage) <= STAGES.index(stop_at)


# When merge or EE run they produce output_directory, so compress/upload read from there.
# When both are skipped the first selected dataset is the source (it already exists).
_has_producing_stage = should_run("merge") or should_run("ee_conversion")
compress_source = (
    output_directory
    if _has_producing_stage
    else (base_dataset_root / move_dataset_repo_ids[0] if move_dataset_repo_ids else output_directory)
)


# ── Stage 1: Conversion ───────────────────────────────────────────────────────

if should_run("conversion"):
    console.rule("[bold cyan]Stage 1 — Dataset format conversion (v2.1 → v3.0)[/]")
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
            console.print(f"  [yellow]⚠[/]  Directory not found for [bold]{repo_id}[/], skipping.")
            continue

        console.print(f"  Converting [bold]{repo_id}[/] …")
        try:
            convert_dataset(
                repo_id=f"{base_dataset_root.name}/{repo_id}",
                root=str(base_dataset_root.parent),
                push_to_hub=False,
            )
            conversion_ok.append((repo_id, "converted"))
        except Exception as e:
            conversion_failed.append((repo_id, dataset_path, e))

    table = Table(show_header=False, box=None, padding=(0, 1))
    for repo_id, label in conversion_ok:
        table.add_row("[green]✔[/]", f"[bold]{repo_id}[/]", f"[dim]{label}[/]")
    for repo_id, _, e in conversion_failed:
        table.add_row("[red]✘[/]", f"[bold]{repo_id}[/]", f"[red]{type(e).__name__}: {e}[/]")
    console.print(table)

    if conversion_failed:
        if not Confirm.ask(
            f"[yellow]{len(conversion_failed)} dataset(s) failed.[/] "
            f"Proceed with [green]{len(conversion_ok)}[/] succeeded?"
        ):
            console.print("[red]Aborting.[/]")
            raise SystemExit(0)
else:
    console.print("[dim]Skipping conversion stage.[/]")


# ── Stage 2: Merge ────────────────────────────────────────────────────────────

if should_run("merge"):
    console.rule("[bold cyan]Stage 2 — Merging datasets[/]")
    datasets_to_merge = []
    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        if dataset_path.is_dir():
            console.print(f"  Loading [bold]{repo_id}[/] …")
            ds = LeRobotDataset(repo_id=repo_id, root=dataset_path)
            console.print(f"    [dim]{ds.num_episodes} episodes, {ds.num_frames} frames[/]")
            datasets_to_merge.append(ds)
        else:
            console.print(f"  [yellow]⚠[/]  {dataset_path} not found, skipping.")

    if output_directory.exists():
        console.print(f"\n  Removing existing output directory: [dim]{output_directory}[/]")
        shutil.rmtree(output_directory)

    console.print(f"\n  Merging [bold]{len(datasets_to_merge)}[/] datasets → [bold]{merged_repo_id}[/] …")
    merged = merge_datasets(
        datasets=datasets_to_merge,
        output_repo_id=merged_repo_id,
        output_dir=output_directory,
    )
    console.print(Panel(
        f"[green]✔[/]  [bold]{merged.root}[/]\n"
        f"   Total episodes : [bold]{merged.meta.total_episodes}[/]\n"
        f"   Total frames   : [bold]{merged.meta.total_frames}[/]",
        title="[green]Merge complete[/]",
        border_style="green",
    ))
else:
    console.print("[dim]Skipping merge stage.[/]")


# ── Stage 3: EE Conversion ────────────────────────────────────────────────────

if should_run("ee_conversion"):
    ee_cfg = cfg.get("joint_to_ee", {})
    ref_frame = args.ee_frame or ee_cfg.get("ee_frame") or cfg.get("ee_frame")
    rot_repr = args.ee_rot_repr or ee_cfg.get("rot_repr", "both")
    include_joint_repr = (
        not args.no_ee_joint_repr
        and ee_cfg.get("include_joint_repr", True)
    )

    if ref_frame is None:
        console.print(Panel(
            "No [bold]ee_frame[/] configured.\n\n"
            "Set it via [bold cyan]--ee-frame arm|robot_base[/] "
            "or add [bold cyan]ee_frame[/] under [bold cyan]joint_to_ee:[/] in config.yaml.",
            title="[yellow]EE Conversion Skipped[/]",
            border_style="yellow",
        ))
    else:
        console.rule("[bold cyan]Stage 3 — EE conversion + action representations[/]")
        # When merge ran, its output is the EE input (in-place).
        # When merge was skipped, the first selected dataset is the input and
        # output_directory is where the enriched copy will be written.
        if should_run("merge"):
            ee_input_directory = output_directory
        else:
            ee_input_directory = (
                base_dataset_root / move_dataset_repo_ids[0]
                if move_dataset_repo_ids else output_directory
            )
        console.print(f"  Input  : [dim]{ee_input_directory}[/]")
        console.print(f"  Output : [dim]{output_directory}[/]")
        try:
            joint_to_ee.process_dataset(
                dataset_root=ee_input_directory,
                output_root=output_directory,
                ref_frame=ref_frame,
                include_joint_repr=include_joint_repr,
                rot_repr=rot_repr,
            )
        except ValueError as exc:
            console.print(Panel(
                f"[bold]{exc}[/]\n\n"
                "[dim]The output directory was already enriched by a previous run.[/]\n\n"
                "To fix, delete the output directory and re-run, or change the "
                "[bold cyan]Output Dir[/] field in the wizard to a fresh name.",
                title="[bold red]EE Conversion Failed — Features Already Exist[/]",
                border_style="red",
            ))
            if Confirm.ask("Skip EE conversion and continue to the next stage?", default=False):
                console.print("[dim]EE conversion skipped — continuing.[/]")
            else:
                console.print("[red]Aborting.[/]")
                raise SystemExit(1)
else:
    console.print("[dim]Skipping EE conversion stage.[/]")


# ── Stage 4: Compress ─────────────────────────────────────────────────────────

if should_run("compress"):
    console.rule("[bold cyan]Stage 4 — Compress[/]")
    zip_path = compress_source.with_suffix(".zip")

    if not compress_source.is_dir():
        console.print(Panel(
            f"[bold]{compress_source}[/] does not exist.\n\n"
            "Update the [bold cyan]Output Dir[/] field in the wizard to match "
            "the target directory name, then save the config and re-run.",
            title="[bold red]Compress Failed — Directory Not Found[/]",
            border_style="red",
        ))
        raise SystemExit(1)

    files = sorted(f for f in compress_source.rglob("*") if f.is_file())
    total = len(files)

    if total == 0:
        console.print(Panel(
            f"[bold]{compress_source}[/] exists but contains no files.\n\n"
            "Check that the [bold cyan]Output Dir[/] field points to a "
            "valid LeRobot dataset directory.",
            title="[bold red]Compress Failed — Empty Directory[/]",
            border_style="red",
        ))
        raise SystemExit(1)

    console.print(f"  Compressing [bold]{total}[/] files → [bold]{zip_path.name}[/] …\n")

    step = max(1, total // 25)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(files):
            zf.write(f, f.relative_to(compress_source))
            n = i + 1
            if n % step == 0 or n == total:
                pct    = n / total * 100
                filled = int(30 * n / total)
                bar    = "█" * filled + "░" * (30 - filled)
                console.print(
                    f"  [[[cyan]{bar}[/cyan]] [cyan]{pct:5.1f}%[/]  [dim]{n}/{total} files[/]"
                )

    zip_mb = zip_path.stat().st_size / 1024**2
    console.print(f"\n  [green]✔[/]  Compressed: [dim]{zip_path}[/]  ({zip_mb:.1f} MB)\n")
else:
    console.print("[dim]Skipping compress stage.[/]")


# ── Stage 5: Upload ───────────────────────────────────────────────────────────

if should_run("upload"):
    console.rule("[bold cyan]Stage 5 — Upload (SFTP)[/]")
    zip_path = compress_source.with_suffix(".zip")

    if not zip_path.exists():
        console.print(f"  [red]✘[/]  Archive not found: [bold]{zip_path}[/]")
        console.print("  Run the compress stage first (or start from compress).")
        raise SystemExit(1)

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
    console.print("  [green]✔[/]  SFTP connection established.")

    remote_file = remote_path + zip_path.name
    file_size   = os.path.getsize(str(zip_path))
    total_mb    = file_size / 1024**2
    console.print(
        f"  Uploading [bold]{zip_path.name}[/] → [dim]{remote_file}[/]  ({total_mb:.1f} MB)\n"
    )

    last_pct = [0]

    def _cb(transferred: int, total: int):
        pct = int(transferred / total * 100)
        if pct >= last_pct[0] + 5 or transferred == total:
            mb     = transferred / 1024**2
            filled = int(30 * transferred / total)
            bar    = "█" * filled + "░" * (30 - filled)
            console.print(
                f"  [[[cyan]{bar}[/cyan]] [cyan]{pct:3d}%[/]  {mb:.1f}/{total_mb:.1f} MB"
            )
            last_pct[0] = pct

    sftp.put(str(zip_path), remote_file, callback=_cb)
    sftp.close()
    ssh.close()

    console.print(Panel(
        f"[green]✔[/]  [bold]{remote_file}[/]",
        title="[green]Upload complete[/]",
        border_style="green",
    ))
else:
    console.print("[dim]Skipping upload stage.[/]")
