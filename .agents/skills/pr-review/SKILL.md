---
name: pr-review
description: The team's first review of a pull request, run locally by your own coding agent — no CI job, no API key. Applies the fixed ten-row rubric to a PR (needs `gh`) or to the current branch before a PR exists (git only), reproduces CI's checks locally, posts one rubric comment on the PR, and marks a draft PR ready for human review once every row passes. Use when asked to review a PR or a branch, before opening or marking a PR ready for review, or when a CODEOWNER wants a first pass on someone else's PR.
---

# pr-review

Every PR gets the same first review before a human looks at it: a fixed rubric, applied by whichever coding agent the author or reviewer uses, on their own machine. PRs are opened as drafts; marking one ready for review is the signal for the CODEOWNER, and it happens only when this review passes.

## Steps

1. **Target.**
   - **PR mode** — a PR number or URL: `gh pr view <n> --json number,title,body,author,isDraft,headRefOid,files` and `gh pr view <n> --comments` (`gh auth status` must pass).
   - **Branch mode** — no PR yet: the current branch's last commit against `origin/main` (`git fetch origin` first). Needs no `gh`. Uncommitted changes are not reviewed; say so if there are any.
2. **Work outside the user's checkout.** Review in detached worktrees under a scratch directory outside the repository, so the user's files and environment stay untouched:
   - head — PR: `git fetch origin +pull/<n>/head`, then `git worktree add --detach <scratch>/head FETCH_HEAD`; branch: `git worktree add --detach <scratch>/head HEAD`.
   - base (for the mypy comparison): `git worktree add --detach <scratch>/base origin/main`.
   - a fresh virtualenv in `<scratch>` with the `python_version` from `.github/workflows/ci.yml` (default 3.11), installing what CI's `install_cmd` installs plus `pytest-cov` and `diff-cover`. Never install into the user's active environment.
   - when done: `git worktree remove --force` both worktrees.
3. **Read** AGENTS.md (root and any directory the change touches); the change folder — the one the PR body names, else the one the diff adds or modifies under `openspec/changes/` — with its `proposal.md`, `design.md` (and its diagrams), `tasks.md` and `explainer.html`; then the diff (`gh pr diff <n>`, or `git diff origin/main...HEAD`). Read a changed file in full when the diff doesn't show enough.
4. **Checks** — what CI runs, reproduced in the head worktree (in PR mode, `gh pr checks <n>` first; run them yourself for anything not green yet). Paths come from the inputs in `.github/workflows/ci.yml`.
   - tests and patch coverage: `pytest -q <test_paths> --cov=<cov_source> --cov-report=xml`, then `diff-cover coverage.xml --compare-branch=origin/main --fail-under=80`.
   - ratchet: `OWNED_PATHS="<src_paths>" RATCHET_RULES=both python .dev-standard/ratchet.py` (with `ruff` on PATH) — no new lint or size violations. Ruff must be fully clean only when `ruff_blocking: true`.
   - mypy: `mypy <src_paths>` in head may not report more errors than in base.
   - secrets: `gitleaks git --log-opts="origin/main..HEAD"` if gitleaks is installed; otherwise read the diff for keys, tokens and passwords.
5. **Rubric.** Each row gets ✅ pass, ❌ fail, ➖ not applicable (also a conditional row whose condition isn't met) or ⚠️ accepted by the user, and one line of evidence:

   | #   | Check                                                                                                       |
   | --- | ----------------------------------------------------------------------------------------------------------- |
   | 1   | Change folder has proposal + design (and the PR body carries both); the diff stays inside that scope        |
   | 2   | Every new public function/class has a test; no test deleted, skipped, xfail'd or loosened                   |
   | 3   | CI green: tests, patch coverage ≥ 80 %, ratchet (no new lint/size violations), no new mypy errors vs main   |
   | 4   | Restricted paths untouched (STANDARD + AGENTS.md), or a CODEOWNER approved that change on the PR            |
   | 5   | Evidence: test and pipeline output; real-hardware log/video if the change touches hardware (`hardware-dod`) |
   | 6   | Docs level / ADR updated when architecture or a durable decision changed                                    |
   | 7   | No fix outside the proposal's scope in the diff; out-of-scope findings are filed as issues instead          |
   | 8   | No secrets; no new dependency without a reason in `design.md`                                               |
   | 9   | PR title is a Conventional Commit ≤ 70 chars                                                                |
   | 10  | Diagrams: _At a glance_ diagram in the PR body matches the diff; design.md has Mermaid; explainer matches   |
   - Branch mode stand-ins: row 1 checks the change folder only; row 4 is ❌ whenever a restricted path is touched, because the approval can only be given on the PR; row 5 checks that the evidence exists and is ready to paste; row 9 — propose a title; row 10 checks `design.md` and the explainer, and drafts the _At a glance_ diagram from the diff for the PR body.
   - Row 10's design and explainer parts apply to change folders whose `.openspec.yaml` says `schema: dev-standard`; an older change needs only the _At a glance_ diagram. A diagram matches when every file or module it marks `changed` or `new` is in the diff and nothing significant in the diff is missing from it.
   - Hardware means anything that runs on or configures a robot, sensor or firmware, calibration files included. If AGENTS.md has no pipeline command, row 5 needs the test output (and hardware evidence where it applies).
   - Out-of-scope things you notice yourself go under findings as "to file"; they don't fail row 7.
   - Docs-only change (every changed file is Markdown, as CI defines it): rows 4, 6, 8 and 9 apply; the rest are ➖.
   - The user may accept a ❌ on rows 1, 3, 6, 7, 9 or 10 with a one-line reason, which goes in the table as ⚠️. Rows 2, 4, 5 and 8 can't be accepted.

   Then at most five findings, most severe first, one per defect (naming the rows it fails): `file:line` — what, why, fix.

6. **Show the user** `Agent review of <7-char head sha>` (with `PR #<n>` or `branch <name>, no PR yet`), the table, the findings and the verdict. Nothing is posted until they agree. If the change folder has an `explainer.html`, offer to open it in their browser (`xdg-open` / `open`) — it is the fastest way into the change.
7. **Post** (PR mode only): one comment whose first line is `<!-- dev-standard:pr-review -->`, followed by what you showed in step 6. On a re-run, update that comment instead of adding another — find it with `gh api --paginate repos/{owner}/{repo}/issues/<n>/comments --jq '.[] | select(.body | startswith("<!-- dev-standard:pr-review")) | .id'` and replace it with `gh api -X PATCH repos/{owner}/{repo}/issues/comments/<id> -F body=@<file>`; the first time, `gh pr comment <n> --body-file <file>`.
8. **Verdict.** Every row ✅, ➖ or ⚠️:
   - PR mode — if the PR is a draft and the user is its author, `gh pr ready <n>`: the CODEOWNER takes it from here.
   - branch mode — ready to open as a draft PR (`gh pr create --draft`, title from row 9, body from the PR template with the _At a glance_ diagram from row 10); then run this review on the PR to post it.

   Any ❌ → the author fixes it and runs this review again; the PR stays a draft.

## Rules

- Review, don't rewrite: never change the author's code or push to their branch, even to fix a finding.
- Don't comment on style the linters already enforce.
- Never approve or request changes on GitHub for the user — approval is the CODEOWNER's own act. The rubric comment is the agent review.
- An out-of-scope finding becomes an issue (`gh issue create --label backlog --label <perf|debt|security|test-gap|docs>`) once the user agrees — never a fix in this PR.
