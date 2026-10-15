# Proposal

## Why

lerobot-edge is the team's fork of huggingface/lerobot, and the team's own code in it (the pi0.5 deployment
server under `deployment/`) had no shared workflow, lint baseline or CI gate. Adopting the team standard
(tiiuae/dev-standard v1) gives it the OpenSpec workflow, the team ruff base, the central CI with a lint ratchet,
and code ownership, without a big-bang cleanup of existing code.

## What Changes

- Added by `install.sh`: `.dev-standard/` (vendored ruff, mypy, prettier and clang-format configs, the ratchet
  script and `STANDARD.md`), `.editorconfig`, `CODEOWNERS`, `.github/ISSUE_TEMPLATE/{feature,backlog}.md`,
  `.github/workflows/ci.yml`, `.github/workflows/graph-on-merge.yml`, `docs/adr/0000-template.md`, `openspec/`
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
  - `.editorconfig`: `mdx` added to the 2-space group. Prettier reads `.editorconfig`; with 4 spaces it re-indented
    lerobot's `.mdx` docs, which fails Quality.
  - `pyproject.toml`: `[tool.coverage.run] relative_files = true`, which the CI's coverage comment needs: it reads
    the coverage data inside a container, where the runner's absolute paths don't exist.
  - `.github/workflows/ci.yml` and `graph-on-merge.yml`: the installer's stubs replaced with copies of the team
    workflows (dev-standard v1.3.1), because a public repository cannot call reusable workflows from the private
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
- The copied workflows run on this public repository today, but no longer update when dev-standard does: re-copy
  them on every dev-standard upgrade, and switch back to the stubs once tiiuae/dev-standard is public (see tasks).
