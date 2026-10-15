# ADR-0001: Adopt dev-standard in lerobot-edge, a public fork

- Status: accepted
- Date: 2026-10-15
- Origin: openspec/changes/adopt-standard/ (PR #4)

## Context and problem statement

lerobot-edge is a public GitHub fork of huggingface/lerobot; the team's own code is `deployment/` and
`tests/deployment/`, everything else is upstream. The team standard (tiiuae/dev-standard v1.3.1) ships workflow stubs
that call reusable workflows in tiiuae/dev-standard, which is private, and GitHub lets a public repository call
reusable workflows only from public repositories. Its installer also makes the repository's ruff config extend the
team rules, which pre-commit then applies to every file, upstream included: lerobot's Quality check failed on
10,086 findings in upstream code.

## Considered options

1. Keep the installer's stubs and wait until tiiuae/dev-standard is public (no team CI meanwhile).
2. Make lerobot-edge private (a fork of a public repository cannot be made private without leaving the fork network).
3. Copy the team workflows into the repository, and apply the team ruff rules only through CI's ratchet, on the
   team-owned paths.
4. Option 3, but with the team ruff rules in a config scoped to `deployment/` (pre-commit would then fail Quality on
   the 88 existing team-rule findings there, 7 of them over the size limits).

## Decision

Option 3: it gives the team CI on this repository today and holds new team code to the team rules, without touching
upstream code or failing lerobot's own checks.

- `.github/workflows/ci.yml` and `graph-on-merge.yml` are copies of the dev-standard v1.3.1 workflows with this
  repository's inputs filled in; the check names stay `ci / python` and `ci / change-present`.
- CI lints, ratchets and tests only `deployment/` and `tests/deployment/`; upstream code is left to lerobot's own CI.
- `pyproject.toml` keeps lerobot's ruff settings; the ratchet lints `deployment/` with `.dev-standard/ruff.toml`
  directly, so a PR may not add team-rule violations there.

## Consequences

- Good: the team gate (ratchet, tests, patch coverage, secret scan, change folder) runs on every PR now, next to
  lerobot's unchanged Quality, Fast Tests and Security checks, and all of them pass.
- Good: upstream files and upstream syncs are not affected by team rules.
- Bad: the copied workflows no longer update when dev-standard does; re-copy them on every upgrade, and switch back
  to the stubs once tiiuae/dev-standard is public and has a `v1` ref.
- Bad: editors and pre-commit show lerobot's rules, not the team's; only CI enforces the team rules.
- Bad: every `install.sh` run adds the ruff `extend` back to `pyproject.toml`; it has to be removed again.
