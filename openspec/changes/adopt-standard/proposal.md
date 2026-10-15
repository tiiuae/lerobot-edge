# Proposal

## Why

lerobot-edge is the team's fork of huggingface/lerobot, and the team's own code in it (the pi0.5 deployment
server under `deployment/`) had no shared workflow, lint baseline or CI gate. Adopting the team standard
(tiiuae/dev-standard v1) gives it the OpenSpec workflow, the team ruff base, the central CI with a lint ratchet,
and code ownership, without a big-bang cleanup of existing code.

## What Changes

- Added by `install.sh`: `.dev-standard/` (vendored ruff, mypy, prettier and clang-format configs, the ratchet
  script and `STANDARD.md`), `.editorconfig`, `CODEOWNERS`, `.github/ISSUE_TEMPLATE/{feature,backlog}.md`,
  `.github/workflows/ci.yml`, `.github/workflows/graph-on-merge.yml` (removed again, see below), `docs/adr/0000-template.md`, `openspec/`
  (config, team schema `dev-standard`), the OpenSpec commands and skills under `.claude/` and `.agents/`, the
  shared skills, and `skills-lock.json`.
- Edited by `install.sh`: the team block appended to `AGENTS.md`; `extend = ".dev-standard/ruff.toml"` added to
  `[tool.ruff]` in `pyproject.toml` (removed again by hand, see below).
- Edited by hand:
  - `AGENTS.md`: removed the `@AGENTS.md` line the installer prepended. `CLAUDE.md` is a symlink to `AGENTS.md`,
    so the import it writes into `CLAUDE.md` lands in `AGENTS.md` as a self-import.
  - `pyproject.toml`: the installer's ruff `extend` removed, so lerobot's own ruff settings stay exactly as upstream
    has them. Extending the team base applied the team rules to all of upstream through pre-commit (lerobot's
    Quality check: 10,086 findings). The team rules still gate new code: `ci.yml`'s ratchet lints `deployment/` with
    `.dev-standard/ruff.toml` directly.
  - `docs/adr/0001-adopt-dev-standard-in-a-public-fork.md`: records the durable decisions (copied CI workflow, graph job left out, team
    rules through the ratchet only, owned paths only).
  - `.editorconfig`: `mdx` added to the 2-space group. Prettier reads `.editorconfig`; with 4 spaces it re-indented
    lerobot's `.mdx` docs, which fails Quality.
  - `pyproject.toml`: `[tool.coverage.run] relative_files = true`, which the CI's coverage comment needs: it reads
    the coverage data inside a container, where the runner's absolute paths don't exist.
  - `.github/workflows/graph-on-merge.yml`: removed. Copied like `ci.yml`, it would run after every merge to `main`;
    its graphify step writes a ~21 MB `graphify-out/graph.json` plus `GRAPH_REPORT.md`, which fail lerobot's pre-commit hooks (1 MB large-file limit, end-of-file, typos, Prettier), and its auto-commit step fails on the missing `docs/diagrams/` paths, so every merge would
    fail and the committed graph would break lerobot's Quality check.
  - `.github/workflows/ci.yml`: the installer's stub replaced with a copy of the team workflow (dev-standard
    v1.3.1), because a public repository cannot call reusable workflows from the private
    tiiuae/dev-standard. Inputs filled in: lint, ratchet and tests on `deployment/` and `tests/deployment/` (the
    team-owned code in this fork); install the `dev` and `test` extras plus the server's `websockets` and `msgpack`;
    Python `3.12` (`requires-python >= 3.12`); docs off. Actions are pinned to commit SHAs, permissions are per job,
    and checkouts that don't push set `persist-credentials: false`, as this repository's zizmor policy requires.
  - `CODEOWNERS`: `@salehaalbreiki` and `@ibrahimkhadraoui` own every path; the `<docs-bot>` entries are dropped.
  - `.pre-commit-config.yaml`: the bandit and mypy hooks exclude `.dev-standard/`, which is vendored and overwritten
    by `install.sh` (the team ruff base already excludes it). They flagged the vendored `ratchet.py`.

## Capabilities

### New Capabilities

None. This change is tooling only.

### Modified Capabilities

None.

## Impact

- No runtime code, API or dependency of the package changes.
- lerobot's Quality check (pre-commit on all files) behaves as on `main`, which passes since #5: editors and
  pre-commit keep lerobot's ruff rules, and the team rules apply in CI, through the ratchet, to new code in
  `deployment/`.
- New CI (`ci.yml`) lints, type-checks, ratchets and tests only `deployment/` and `tests/deployment/`; ruff and
  mypy are non-blocking until that code is clean.
- `deployment/` is on `main` (#2), so CI compares like for like: every blocking `ci / python` step passes locally
  against `origin/main` (ratchet 7 → 7 size, 88 → 88 lint; tests; patch coverage; gitleaks; change folder).
- The copied `ci.yml` runs on this public repository today, but no longer updates when dev-standard does: re-copy
  it on every dev-standard upgrade, and switch back to the stub once tiiuae/dev-standard is public (see tasks).
- No graph or diagrams are regenerated after merges (`graph-on-merge.yml` left out); `AGENTS.md`'s managed team
  block still mentions them, because the installer owns that text.
