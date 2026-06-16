"""Standalone CLI: python -m joint_to_ee --dataset ..."""
import argparse
from pathlib import Path

from .pipeline import process_dataset
from .representations import ROT_REPRS


def _parse_args():
    p = argparse.ArgumentParser(
        description="Enrich a LeRobot dataset with EE poses and action representations.")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="Output dir (default: modify dataset in-place).")
    p.add_argument("--frame", choices=["arm", "robot_base"], default="robot_base")
    p.add_argument("--rot-repr", choices=list(ROT_REPRS), default="both",
                   help="Orientation representation for EE delta/relative (default: both).")
    p.add_argument("--no-joint-repr", action="store_true",
                   help="Skip joint-space action.delta / action.relative.")
    return p.parse_args()


def main():
    args = _parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    output_root = args.output.expanduser().resolve() if args.output else dataset_root
    process_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        ref_frame=args.frame,
        include_joint_repr=not args.no_joint_repr,
        rot_repr=args.rot_repr,
    )


if __name__ == "__main__":
    main()
