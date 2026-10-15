---
name: docs-site
description: Create or rebuild a repository's documentation site in the team style — Sphinx with Markdown (MyST) pages laid out like the ROS 2 docs (Getting started, Tutorials, How-to guides, Concepts, Reference with an API reference generated from docstrings, Decisions, Project), Mermaid diagrams, the team design system's look in light and dark. Use ONLY when the user explicitly asks for this documentation site (or types /docs-site). Never on your own initiative, never as part of installing or upgrading the standard, never as a side effect of another task — not even a docs edit.
---

# docs-site

A documentation site for one repository, the same in every repository: the sections of the [ROS 2 docs](https://docs.ros.org/) (Diátaxis), built with Sphinx like ROS 2 itself, pages in Markdown (MyST), the pydata-sphinx-theme dressed in the team design system, published to GitHub Pages by the reusable `graph-on-merge.yml` and checked on every PR by the reusable `ci.yml`. The look is managed by the standard (`.dev-standard/docs-site/theme`, refreshed by `install.sh`); the pages belong to the repository.

Invoked as `/docs-site` (Claude Code, Cursor), `$docs-site` (Codex), or by asking for the documentation site in words.

## Hard rules

1. **Opt-in only.** Act only when the user has explicitly asked for this documentation site to be created, rebuilt or restructured. Anything else — installing or upgrading the standard, a docs fix, a README edit, a feature that "needs docs" — is not a request for a site: don't create `docs/conf.py`, don't move pages, and at most mention that `/docs-site` exists.
2. **Ask about the existing documentation first, and wait.** Before changing any file, list what you found and ask the user which of these they want (one question, these three options, plus anything specific to the repository):
   - **Keep in place** — every existing file stays where it is; the site only adds it to its navigation. No file moved, no link broken.
   - **Reorganise** — move or split the old pages into the Diátaxis sections, rewriting every link that pointed at them (in docs, README, code comments, AGENTS.md).
   - **Replace** — new pages written from the old ones; the old files archived (`docs/archive/`) or removed — ask which.

   Do nothing until the user answers. If the repository has no documentation beyond `docs/README.md` and `docs/adr/`, say so and ask for a go-ahead anyway.

3. **The repository's AGENTS.md applies.** Design first: an OpenSpec change (`/opsx:propose docs-site`) before any file; its `design.md` names the new docs dependencies and why (copy _Dependencies_ below). No edit to `.github/workflows/**` or to a lockfile without the user's explicit approval in this conversation. Branch `feat/…-docs-site` or `docs/…`, Conventional Commits.

## Steps

1. **Inventory** (read only): `docs/**`, Markdown at the root (`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`), an existing docs config (`docs/conf.py`, `mkdocs.yml` — a site already exists: this is a rebuild, read it all), the source layout (`src/<pkg>/` or packages at the root), the docstring style, `.github/workflows/{ci,graph-on-merge}.yml`, a `Makefile`, `.gitignore`, `docs/adr/` numbering.
2. **Ask** — rule 2. Wait.
3. **Templates.** They are vendored in `.dev-standard/docs-site/`. If that directory is missing, re-run the installer with the flag from a dev-standard checkout: `bash <SRC>/install.sh --src <SRC> --docs-site` (`<SRC>` is usually `~/dev-standard`; without a checkout, ask before `git clone https://github.com/tiiuae/dev-standard ~/dev-standard`). It also refreshes the repository's other vendored files to that checkout's version — say so in the report.
4. **Propose** the OpenSpec change: proposal, `design.md` (the user's answer from step 2, the section map — which existing page goes where — the dependencies, the workflow edits you will ask for), tasks, explainer. Show the user the section map before writing pages if it moves or splits anything.
5. **Scaffold** — copy, never overwrite: `cp -Rn .dev-standard/docs-site/docs/ docs/` (keeps every file that already exists). `theme/` stays in `.dev-standard/` — `docs/conf.py` points at it.
   - Replace every placeholder in `docs/conf.py` and the pages: `PROJECT NAME`, `ORG`/`REPO`, `ORG/REPO`, `TODO …` (`grep -rn 'TODO\|ORG\|PROJECT NAME' docs` comes back empty, except text the repository already had).
   - `autoapi_dirs` in `docs/conf.py`: `["../src"]` for a src layout; for packages at the root, one entry per package (`["../my_pkg"]`) — never `[".."]`, which would document tests and scripts.
   - `docs/adr/NNNN-documentation-site.md` → the next free number; fill it in (Date, Origin, the existing-docs choice).
   - `.gitignore`: add `docs/_build/` and `docs/reference/api/` (the generated API pages; Sphinx removes them after a build, the line covers an interrupted one).
6. **Existing pages**, as the user chose:
   - _Keep in place_: add each existing page to the `toctree` of the section it fits, by its path from that page (`../architecture`), or to the hidden toctree in `docs/index.md`; leave the file and its links alone. `docs/README.md` stays the GitHub folder index and is excluded from the site (`exclude_patterns`); link its useful content from `index.md` instead of copying it. If `graph-on-merge.yml` has `archify: true`, it writes `docs/architecture.md` on merge: list it in Concepts (create it with a title if it is missing), or the strict build fails on `main`.
   - _Reorganise_: `git mv` each page into its section folder (`getting-started/`, `tutorials/`, `how-to/`, `concepts/`, `reference/`) and add it to that folder's `toctree`; split a page that mixes kinds (a tutorial with reference tables → two pages). Then `grep -rn '<old path>'` across the repository and fix every link.
   - _Replace_: write the new pages; archive or remove the old ones as the user said.

   Every page must be in a `toctree` — the strict build fails on one that isn't, which is how nothing gets lost. The API reference lists itself (`api/*/index` in `reference/index.md`); ADRs too (`[0-9]*` in `adr/index.md`).

7. **Writing pages** (MyST Markdown):
   - Links inside `docs/`: relative `.md` links. To anything outside `docs/` (source, root README, `AGENTS.md`, config): the file's GitHub URL, `https://github.com/ORG/REPO/blob/main/<path>` (`/tree/main/<dir>` for a directory). A relative link to a Markdown file outside `docs/` fails the strict build; one to any other file (`../src/x.py`) silently becomes a download of a copy — also wrong.
   - Diagrams: ` ```mermaid ` fences, real component names, about 15 nodes at most. No images of diagrams.
   - Admonitions: `:::{note}` … `:::` (also `tip`, `warning`, `danger`); with a title: `:::{admonition} Title` and `:class: warning` on the next line.
   - Cross-reference the API: `` {py:class}`pkg.module.Class` ``, `` {py:func}`pkg.module.func` ``.
   - Keep lists flat: `prettier` (pre-commit) re-indents nested lists to 2 spaces, which changes their meaning.
8. **Makefile** — add these targets (create a Makefile only if the user agrees; otherwise put the commands in AGENTS.md):

   ```make
   # Docs site (skill docs-site): its own pinned environment, never the project's.
   DOCS := uv run --no-project --isolated --python 3.12 --with-requirements docs/requirements.txt

   .PHONY: docs docs-serve
   docs:        ## build the docs site; any warning fails
   	$(DOCS) sphinx-build -W --keep-going -b html -d docs/_build/doctrees docs docs/_build/html
   docs-serve: docs  ## build, then serve at http://127.0.0.1:8000
   	python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
   ```

   Add both to _Commands (exact)_ in AGENTS.md.

9. **Build**: `make docs` must end with `build succeeded` and no warning. Open the site (`make docs-serve`) and look at the home page, one page per section and an API page, in light and in dark (the theme switch in the header), and at phone width. Attach the screenshots to the PR.
10. **Workflows — ask first** (`.github/workflows/**` is restricted). Propose these edits and make them only after the user approves:
    - `.github/workflows/ci.yml`, under `with:` — `docs: true`: the reusable workflow runs `make docs` on every PR, so a broken docs build blocks the merge.
    - `.github/workflows/graph-on-merge.yml` — `pages: write` under `permissions:`, and under `with:` `docs: true` and `docs_cmd: make docs`: after each merge it builds `docs/_build/html` and publishes it with GitHub's Pages actions.

    Under "Only a human can do": Settings → Pages → Source: **GitHub Actions**; branch protection: add the required check `ci / docs`.

11. **Report**: the user's choice and what it meant (files kept, moved, replaced), the section map, `make docs` output, the screenshots, the workflow edits (made or pending approval), and anything left as TODO.

## Dependencies (for design.md)

Docs only, pinned in `docs/requirements.in` (exact) and `docs/requirements.txt` (`uv pip compile` output), installed in their own environment — the project never imports them, and they never enter `pyproject.toml` or `uv.lock`:

- `sphinx` — the site generator (what ROS 2, Python and NumPy use).
- `myst-parser` — Markdown pages, with directives and Mermaid fences.
- `pydata-sphinx-theme` — the theme: top section tabs, sidebar, light/dark; restyled by the team theme.
- `sphinx-autoapi` — the API reference from the source by static analysis (the project is not imported, so its dependencies need not be installed); Google docstrings through Sphinx's own `napoleon`.
- `sphinxcontrib-mermaid` — renders the Mermaid fences.

To update them: edit the pins in `requirements.in`, then `uv pip compile docs/requirements.in -o docs/requirements.txt --python-version 3.12` (a lockfile: the user approves the change).

## Don't

- Don't edit `.dev-standard/docs-site/` — `install.sh` overwrites it. A look change goes to `dev-standard` (`templates/docs-site/theme/`) by PR.
- Don't add CSS with literal colours or fonts, and don't restyle the team theme from `conf.py`.
- Don't commit `docs/_build/` or the generated API pages; don't hand-write API pages — fix the docstring.
- Don't install the docs dependencies into the project's environment.
- Don't silence warnings (`suppress_warnings`, dropping `-W`) to get a green build: fix the page.

## Definition of done

The user's existing-docs choice carried out exactly · `make docs` passes with zero warnings · every section has a landing page and the API reference lists the packages · no placeholder left · no link into `docs/` broken, none out of it relative · light, dark and phone-width screenshots on the PR · ADR added · workflow edits made only with the user's approval.
