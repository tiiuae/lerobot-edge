---
name: ui-design-system
description: Build or change any web UI (dashboard, control panel, chat/agent interface, settings, monitor, admin page) with the team design system — tokens, class vocabulary, light/dark, shell layout. Use whenever a task adds or edits HTML, CSS, JSX, templates or components in any repo, or asks for a new screen, panel, widget or page; also when reviewing UI code for consistency.
---

# ui-design-system

Every UI the team ships uses one stylesheet and one class vocabulary, so screens from different repos read as one product and any engineer or agent can work on any of them. The system lives in `tiiuae/dev-standard/ui/`; this skill tells you how to use it and what not to do.

Not for a change's `openspec/changes/*/explainer.html`: that page comes from the OpenSpec template, which carries the tokens itself and animates on purpose.

## Before writing any UI code

1. Make sure the system is present: `ui/styles.css` linked in the page (or `.dev-standard/ui/` vendored by `install.sh --ui`). If not, add it — do not write your own colours, fonts or buttons.
2. Read `ui/GUIDELINES.md` (rules, tokens, class vocabulary, layout) and, for an existing screen, the page's current markup to reuse its patterns.
3. Decide the shell: does the screen fit `.topbar` + `.layout` (sidebar · main · drawer)? Almost always yes. A new layout pattern needs a one-line justification in the PR.

## Rules

- **Tokens only.** Never write `#hex`, `rgb()`, a font name, or a shadow that is not `var(--scrim)`. Missing a value → propose a token in `dev-standard`, don't inline. Own layout glue (flex, grid, gaps 4–16px) is fine.
- **One primary action per surface.** `.btn-primary` once; everything else `.btn-quiet`, `.btn-ghost`, `.btn-icon`. Status colours only on things that report state.
- **State through attributes**, not classes: `aria-expanded`, `aria-pressed`, `aria-selected`, `aria-current`, `data-status`, `data-state`, `data-level`, `data-open`, `hidden`. The CSS already selects on them.
- **Both themes, always.** Toggle `html.dark` and look again before calling a screen done. Canvas/WebGL reads the theme from a helper, never from a literal.
- **Small type, quiet motion.** No text above 18px, no entrance animations, no hover effects beyond colour.
- **Compose before you create.** A `.card` + `.section-header` + `.field`s + `.actions` covers most panels. A new component goes into `dev-standard/ui/components.css` via PR, not into a product's CSS.
- **Accessibility is not optional**: `aria-label` on icon buttons, real `<button>`/`<label>` elements, focus visible, colour never the only channel.

## Idiomatic snippet

```html
<section class="card">
  <div class="pane-header">Policy server</div>
  <div class="card-body">
    <label class="field"
      ><span>Endpoint</span
      ><input class="input input-mono" value="10.0.0.4:8000"
    /></label>
    <label class="field"
      ><span>Mode</span
      ><select class="input">
        <option>real</option>
        <option>replay</option>
      </select></label
    >
    <div class="chip-row">
      <span class="pill"
        ><span class="k">warmup</span
        ><em class="v" data-kind="true">true</em></span
      >
      <span class="pill"
        ><span class="k">steps</span
        ><em class="v" data-kind="number">50</em></span
      >
    </div>
    <div class="meter" data-level="warn">
      <div class="meter-head">GPU <em>78%</em></div>
      <div class="bar"><div class="bar-fill" style="width:78%"></div></div>
    </div>
    <div class="actions">
      <button class="btn btn-quiet">Reset</button
      ><button class="btn btn-primary">Connect</button>
    </div>
  </div>
</section>
```

## Definition of done for a UI change

Screen uses only tokens and vocabulary classes (run `npx stylelint "**/*.css" --config .dev-standard/ui/.stylelintrc.json` on your CSS) · looks right in light and dark (attach both screenshots to the PR) · keyboard-navigable, focus visible · one primary action per surface · no new literal colour, font or shadow · new components, if any, proposed to `dev-standard`.
