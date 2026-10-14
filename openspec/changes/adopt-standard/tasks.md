# Tasks

## 1. Adoption

- [x] 1.1 Run `install.sh` from tiiuae/dev-standard v1.3.1 and verify a re-run reports no notes
- [x] 1.2 Rename the repository's ruff `select` / `ignore` to `extend-select` / `extend-ignore` and verify `ruff check --show-settings deployment/model_server/server_policy.py` shows `linter.mccabe.max_complexity = 10` and `linter.pylint.max_statements = 50`
- [x] 1.3 Remove the `@AGENTS.md` self-import written through the `CLAUDE.md` symlink and verify `AGENTS.md` starts with its original first line
- [x] 1.4 Set the CI inputs (owned paths, install command, Python 3.12) and verify the `install_cmd` and the test step pass in a scratch environment (`uv sync --locked --extra dev --extra test`, then `uv run pytest tests/deployment --cov=deployment`: 45 passed)
- [x] 1.5 Fill `CODEOWNERS` with `@salehaalbreiki` and `@ibrahimkhadraoui` and verify no `<team>` or `<docs-bot>` placeholder remains
- [x] 1.6 Replace the `ci.yml` and `graph-on-merge.yml` stubs with copies of the dev-standard v1.3.1 workflows (inputs filled in, actions pinned to SHAs, per-job permissions) and verify `actionlint` and `zizmor --offline` report nothing, and that the copied jobs differ from the source only in the filled-in inputs, check names and those zizmor fixes
- [x] 1.7 Exclude the vendored `.dev-standard/` from the bandit and mypy hooks and verify `pre-commit run bandit --files .dev-standard/ratchet.py` and `pre-commit run mypy --files .dev-standard/ratchet.py` skip it
- [x] 1.8 Write `design.md` (flowchart and Before/After diagrams) and `explainer.html` from the team templates and verify `openspec validate adopt-standard` passes and the explainer renders without an error banner

## 2. Decisions to confirm

- [ ] 2.1 Decide how to handle the existing `quality.yml` pre-commit job, which now fails on 10,070 whole-repo ruff violations (options: scope the ruff hooks to team-owned paths, keep the team base out of the hooks' config, or accept the job failing), and verify the job's result on the first PR
- [ ] 2.2 Confirm `ci.yml` covering only `deployment/` and `tests/deployment/` (upstream's suite stays in `fast_tests.yml` / `full_tests.yml`, which overlap `ci.yml` for the deployment tests) and verify on the first PR's `ci / python` check
- [ ] 2.3 Confirm leaving `dataset-wizard.py` (team-owned, root script) out of `src_paths` and verify the choice in `ci.yml`
- [ ] 2.4 Review the ruff deviations kept from the repository (`line-length = 110` vs the team's 100; `extend-ignore` of `E501`) and verify the decision in `pyproject.toml`
- [ ] 2.5 Note that the repository's `[tool.ruff.lint.per-file-ignores]` replaces the team base's (tests and scripts lose the base's `ANN`, `PLR`, `S` exemptions locally; CI's ratchet uses the team base alone) and verify the decision in `pyproject.toml`

## 3. Only a human can do

- [ ] 3.1 Set branch protection on `main` (PR required, checks `ci / python` and `ci / change-present`, 1 approval, review from Code Owners, dismiss stale approvals, squash-merge only) and verify it in the repository settings
- [ ] 3.2 Create the labels `backlog perf debt security test-gap docs feature` and verify with `gh label list`
- [ ] 3.3 Optionally add the `ANTHROPIC_API_KEY` (archify diagrams on merge) and `DOCS_APPROVER_TOKEN` (docs-only auto-approve) secrets and verify in the repository settings

## 4. Follow-ups

- [ ] 4.1 Report to tiiuae/dev-standard that `install.sh` writes the `@AGENTS.md` import through a `CLAUDE.md -> AGENTS.md` symlink (it reappears on every re-run or upgrade) and verify an issue is filed
- [x] 4.2 Fix `ratchet.py` linting the whole repository on the base side when an owned path does not exist there, upstream, and verify tiiuae/dev-standard#6 is open with tests
- [x] 4.3 Merge #2 (`clean-up`, brings `deployment/` to `main`) before this PR, and verify this PR's `ci / python` ratchet and patch-coverage steps pass against the updated `main` (locally: ratchet 7 → 7, 88 → 88; no covered lines in the diff)
- [ ] 4.4 Archive this change after the PR merges (`npx -y @fission-ai/openspec@1 archive adopt-standard --yes`) and verify `openspec/changes/adopt-standard/` has moved under `openspec/changes/archive/`
- [ ] 4.5 Once tiiuae/dev-standard is public and has a `v1` ref, switch `ci.yml` and `graph-on-merge.yml` back to the installer's stubs (`uses: tiiuae/dev-standard/.github/workflows/…@v1`) and verify `ci / python` and `ci / change-present` run from the stub
