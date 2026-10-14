"""Ratchet: lint violations may not exceed the count on the base branch.

Runs ruff on the working tree and on `origin/main` (via `git worktree`), compares counts per
rule set, and fails if the PR adds violations. This is how a brownfield repo adopts the team
rules without a big-bang cleanup: existing debt is frozen, new code must be clean, and the
count only goes down. Two rule sets: SIZE (complexity/size, always enforced) and ALL (every
rule the repo config enables). Set RATCHET_RULES=size|all|both (default both).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SIZE_RULES = "PLR0915,PLR0912,PLR0911,C901"
MODE = os.environ.get("RATCHET_RULES", "both")
PATHS = os.environ.get("OWNED_PATHS", "src").split()   # CI passes the repo's src_paths
BASE = os.environ.get("RATCHET_BASE", "origin/main")


CONFIG = Path.cwd() / ".dev-standard" / "ruff.toml"   # head's team config, applied to BOTH sides so a config change never trips the ratchet


def count(root: Path, select: str | None) -> int:
    """Number of violations under root; `select` limits the rules, None = the team config."""
    paths = [str(root / p) for p in PATHS if (root / p).exists()]
    cmd = ["ruff", "check", "--config", str(CONFIG), "--output-format", "json", "--exit-zero", *(["--select", select] if select else []), *paths]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return len(json.loads(out.stdout or "[]"))


def sets() -> list[tuple[str, str | None]]:
    """Rule sets to compare, by mode."""
    size, everything = ("size", SIZE_RULES), ("all", None)
    return {"size": [size], "all": [everything]}.get(MODE, [size, everything])


def main() -> int:
    """Compare head vs base; return 1 if head has more violations."""
    head = {name: count(Path.cwd(), sel) for name, sel in sets()}
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp) / "base"
        wt = subprocess.run(["git", "worktree", "add", "--detach", str(base_dir), BASE], capture_output=True, text=True)
        if wt.returncode:
            print(f"::error::cannot check out {BASE}: {wt.stderr.strip()} (fetch-depth: 0 and origin/main required)")
            return 1
        try:
            base = {name: count(base_dir, sel) for name, sel in sets()}
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(base_dir)], capture_output=True)
    failed = False
    for name in head:
        print(f"ratchet[{name}]: base={base[name]} head={head[name]}")
        if head[name] > base[name]:
            failed = True
            what = "complexity/size (split the function or file)" if name == "size" else "lint (run `ruff check --fix` and fix the rest)"
            print(f"::error::this PR adds {head[name] - base[name]} {what} violation(s) — AGENTS.md")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
