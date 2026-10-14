This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

<!-- dev-standard:begin — managed by install.sh, do not edit by hand -->

## Team standard (dev-standard v1)

Full text: `STANDARD.md` in `tiiuae/dev-standard`. Read by Claude Code (`CLAUDE.md` imports this file), Cursor, Codex and OpenCode.

**Workflow.** Design before code with OpenSpec (team schema `dev-standard`): `/opsx:explore` → `/opsx:propose <feature>` (creates `openspec/changes/<feature>/{proposal,design,tasks}.md` and `explainer.html`, an animated page that explains the change — tell the user its path) → `/opsx:apply` → PR → `/opsx:archive` after merge. Read the change folder before touching code; if scope changes, update `design.md`, its diagrams and `explainer.html` in the same PR. Branch `feat/<ticket>-<slug>` · `fix/…` · `chore/…`; Conventional Commits; squash-merge, PR title = commit. Never `--no-verify`, never push to `main`.

**PR body** = an _At a glance_ Mermaid diagram of what the diff touches, then `proposal.md` + `design.md` (with its Mermaid diagrams) under the template headings, plus evidence (tests, pipeline output — see Commands above, real-hardware log/video if `hardware` applies). Open the PR as a draft and run the agent review locally (skill `pr-review`: it posts the rubric table on the PR); fix until every row passes, then mark it ready. One CODEOWNER approves; docs-only PRs need only the agent review.

**Definition of done.** Every new function/class has a test; no test deleted, skipped or loosened; lint and tests green (the Commands above); ratchet not worse than `main` (≤ 50 statements per function, complexity ≤ 10, ≤ 400 lines per file unless the project says stricter); docs level updated; durable decision → `docs/adr/NNNN-<slug>.md` (MADR); out-of-scope findings filed with `gh issue create --label backlog --label <perf|debt|security|test-gap|docs>`, never fixed in this PR.

**Diagrams.** In Markdown (design.md, PR bodies, docs) diagrams are Mermaid in ` ```mermaid ` fences, which GitHub renders: real module/service/device names, about 15 nodes at most, what a change adds, alters or removes marked with the `new` / `changed` / `removed` classes from the design template — not ASCII art.

**Code.** Google style for Python and C++, enforced by the linters in `.dev-standard/`; types and docstrings on every public function; no I/O at import time; no secrets, hard-coded URLs, ports or model names; no new dependency without a line in `design.md` saying why. Before editing a file read it; before changing a function grep its callers; for codebase questions query `graphify-out/graph.json` (`graphify query "…"`) before grepping.

**Restricted without explicit human approval in the PR.** `.github/workflows/**`, lockfiles, `infra/`, `deploy/`, migrations, secrets, `vendor/`, `third_party/`, anything that runs on a robot, and — on forks — every upstream file (ours live in owned paths or `patches/`).

**UI.** Any HTML/CSS/JS change uses the team design system (`.dev-standard/ui/styles.css`, rules in `.dev-standard/ui/GUIDELINES.md`, skill `ui-design-system`): tokens only, no literal colours/fonts/shadows, state through attributes, both themes checked. A change's `explainer.html` is the exception: it comes from the OpenSpec template, which carries the tokens itself.

**Tools.** graphify graph and archify diagrams under `graphify-out/` and `docs/diagrams/` are regenerated by CI on every merge to `main`; never regenerate or commit them from a branch. Shared skills: `npx skills add tiiuae/dev-standard -y`.

<!-- dev-standard:end -->
