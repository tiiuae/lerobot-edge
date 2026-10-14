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
  `[tool.ruff]` in `pyproject.toml`.
- Edited by hand:
  - `AGENTS.md`: removed the `@AGENTS.md` line the installer prepended. `CLAUDE.md` is a symlink to `AGENTS.md`,
    so the import it writes into `CLAUDE.md` lands in `AGENTS.md` as a self-import.
  - `pyproject.toml`: `[tool.ruff.lint]` `select` / `ignore` renamed to `extend-select` / `extend-ignore`, so
    the repository's rules add to the team base instead of replacing it.
  - `.github/workflows/ci.yml`: `src_paths` / `cov_source` set to `deployment` and `test_paths` to
    `tests/deployment` (the team-owned code in this fork); `install_cmd` installs the `dev` and `test` extras
    plus the server's `websockets` and `msgpack`; `python_version` set to `3.12` (`requires-python >= 3.12`).
  - `CODEOWNERS`: `@salehaalbreiki` and `@ibrahimkhadraoui` own every path; the `<docs-bot>` entries are dropped.
  - `.github/workflows/ci.yml`: no `secrets: inherit` (zizmor `secrets-inherit`): it would pass every repository
    secret to a workflow in another repository at the movable `@v1` tag. `GITHUB_TOKEN` is always available to the
    called workflow; its only other secret, `DOCS_APPROVER_TOKEN`, is optional.
  - `.pre-commit-config.yaml`: the bandit and mypy hooks exclude `.dev-standard/`, which is vendored and overwritten
    by `install.sh` (the team ruff base already excludes it). They flagged the vendored `ratchet.py`.

## Capabilities

### New Capabilities

None. This change is tooling only.

### Modified Capabilities

None.

## Impact

- No runtime code, API or dependency of the package changes.
- The pre-commit ruff hooks now apply the team base on top of the repository's rules: whole-repo violations go
  from 42 to 10,070 (ruff 0.14.1, mostly `ANN*` and complexity rules in upstream code). `quality.yml` runs
  `pre-commit run --all-files` on every PR, so that job fails until this is resolved (see tasks).
- New CI (`ci.yml`) lints, type-checks, ratchets and tests only `deployment/` and `tests/deployment/`; ruff and
  mypy are non-blocking until that code is clean.
- Without `secrets: inherit`, docs-only PRs are not auto-approved even if `DOCS_APPROVER_TOKEN` is set.
- Depends on #2 (`clean-up`), which brings `deployment/` to `main`. Before #2 merges, CI compares against a `main`
  without `deployment/`: patch coverage fails (58% < 80%) and the ratchet passes only through a dev-standard bug that
  lints the whole repository on the base (fixed in tiiuae/dev-standard#6). After #2 merges, every blocking check
  passes locally (ratchet 7 → 7 size, 88 → 88 lint).
