---
name: hardware-dod
description: Definition-of-done evidence for a change that touches a robot, sensor, firmware or real hardware — plan the real-hardware run, capture the log or video, and attach it to the PR body. Use before opening a PR whenever the proposal or design mentions a robot, arm, drone, camera, sensor, board, firmware, or a make test-hw target; also when the PR template's "Touches hardware" is yes.
---

# hardware-dod

Unit tests and a replay pipeline prove the code; only a run on the real hardware proves the change. This skill produces the evidence the standard requires and refuses to open the PR without it.

## Steps

1. **Decide scope.** From `openspec/changes/<feature>/design.md`, list what physically moves or reads: which robot/board, which sensors, which safety interlock (e-stop, workspace limits, speed cap). If nothing physical is affected, say so and stop — this skill does not apply.
2. **Pre-run checklist** (ask the human to confirm each): workspace clear, e-stop reachable, speed/force limits at test values, firmware/config versions noted, a second person present if the arm moves.
3. **Run** the repo's real-hardware target (`make test-hw` or the command in `AGENTS.md §Commands`) with the smallest instruction that exercises the change (e.g. `home`, one gesture, one short episode). Capture: command, start/end time, git SHA, hardware identifiers, the JSON/log output, and a short video if motion is involved.
4. **Evidence block** — append to the PR body under _Evidence → hardware run_:
   ```
   hardware: <robot/board> · fw <version> · sha <7 chars> · <YYYY-MM-DD HH:MM TZ>
   command: make test-hw PROMPT="home"
   result: episode_done state=holding in 12.4 s · log: <path or link> · video: <link>
   observed: <one line: what happened, anything unexpected>
   ```
5. **Failure** = a finding, not a retry loop: file it (`gh issue create --label backlog --label debt`) or fix it in scope, re-run once, and record both runs.

## Rules

- Never mark hardware evidence as done from a simulation, replay or unit test.
- Never raise speed/force limits or bypass an interlock to make a run pass.
- The human runs the physical test; the agent prepares, records and formats.
