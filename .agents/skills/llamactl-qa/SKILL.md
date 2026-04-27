---
name: llamactl-qa
description: Plan and run CLI-side QA for llamactl changes. Cooperatively builds a small test matrix from a diff or set of acceptance criteria, executes against a chosen backend (prod test project by default, local kind+tilt for new API contract changes), and writes a transcript report to `thoughts/shared/qa/`. Use this for changes to `llamactl` commands, output formats, auth, or control-plane API contracts. For UI/template smoke tests, use `llamactl_browser_test` instead.
---

# llamactl CLI QA

Two halves. First, a planning pass that turns "I changed X" into a small, sign-off-able test matrix. Second, an execution pass that runs the matrix and writes a transcript report.

The point is deliberateness, not exhaustiveness. Every row in the matrix is there because we agreed it covers a real risk in the diff.

llamactl is churning right now (multi-slice rework: output modes, apply/delete, template/apply-loop). Treat this skill as a living reference. When a run teaches you something about llamactl that isn't in here, edit the skill before closing the task. See "Self-update" at the bottom.

## When to use

- Changes to `llamactl` commands (new flags, new output modes, command splits, behavior changes).
- Changes to control-plane API contracts that llamactl consumes.
- Changes to auth / profile / project resolution.

Skip when the change is fully covered by pytest, or when the failure mode is in the UI (use `llamactl_browser_test`).

## Pick a backend

1. **Prod against a test project.** Default. Read-only commands, and write commands targeting a deployment in a project explicitly designated for testing. Confirm the project name with the user before running anything mutating. Pin every command with `--project <test-project-id>` so an active-profile slip can't write into a real project.
2. **Local kind + tilt.** When the change is a new API contract (new endpoint, new field, new behavior on the control plane) and you need to exercise it against a control plane that actually has the change. `uv run operator/dev.py up`. See `AGENTS.md` and `operator/AGENTS.md` for setup. Local mode runs with no auth: the `http://localhost:8011` env is preconfigured; switch to it with `llamactl auth env switch http://localhost:8011` (arg is the API URL, not a name) and a `default` profile is created automatically. No tokens, no `auth login`. Before drafting the matrix, list existing deployments with `kubectl get llamadeployments -A` — there's often something already in the cluster from prior work that you can target with `--project <id>`, saving the cost of a fresh `deployments create`.
3. **Staging.** Rarely the right answer for this skill. Use prod-test-project or local instead.

Ask the user which backend before drafting the matrix. The answer changes which rows are realistic.

## llamactl mental model

The bits that matter when designing a matrix.

A profile is `(env, oauth tokens, active org, active project)`. The env is a control-plane URL. `--project <id>` overrides the profile's active project for one call. `llamactl auth list` shows the stack. `llamactl auth env list` shows envs.

Read commands accept `-o text|json|yaml`. Text is human-facing; json/yaml are assertable. Prefer json for QA — exact-match assertions catch field-rename regressions text formatting hides.

Several commands branch on a TTY check or `--no-interactive`. If a command can launch a TUI or open a browser, exercise both branches. Always pass `--no-interactive` (or `-o json`) for QA runs so a TUI doesn't appear to hang the session.

The command surface is moving. Run `--help` on any command you're testing before assuming flag names; don't go from memory.

## Cooperative planning

Don't run anything until the user has signed off on the matrix.

1. Read the diff. `git diff <base>...HEAD --stat` and targeted reads on the command files. Identify the behavior surface: which subcommands, flags, output formats, endpoints changed.
2. List risks, not features. One line each. "JSON output for `deployments get` could omit fields the schema declares." "`--project` override could be ignored when an env var is also set." A risk is what a reviewer would reasonably ask "did you check".
3. Map risks to a small matrix. One row per case. 4–10 rows is usually right; 20 means you're testing features.
4. Present the matrix inline. Wait for the user to keep, cut, or add rows.

Matrix row format:

| # | Command | Backend | Expect | Covers |
|---|---|---|---|---|
| 1 | `deployments get -o json <id> --project <test>` | prod-test | exit 0; stdout parses as json with `id`, `status`, `latest_release` keys | json mode shape regression |
| 2 | `deployments get --project <test>` | prod-test | exit 0; lists ≥1 row; no TUI | new list-mode path |
| 3 | `deployments get <id> --project <other-test>` | prod-test | returns deployment from `<other-test>`, not active project | `--project` override |

If a row's "Covers" reads as "verify the command works", drop it.

## Execution

Run rows top to bottom. Capture for each:

- The exact command, including any inline env vars.
- Exit code and elapsed time (latency is part of the UX).
- **Verbatim stdout and stderr.** Don't trim, don't summarize, don't replace with "all keys present". The point of the report is the user reading the actual output and judging the design. Truncation only when output is genuinely huge (200+ line log dumps); even then, keep the head, the tail, and a `... <N> lines elided ...` marker so the reader can see what got cut.
- Anything you'd call out in a design review: column widths and truncation, field naming, key ordering, redundant data, error message wording, how the output looks when something is missing.

When you make an assertion (`jq`, `python -c`), include both the assertion command and the verbatim output so the reader can see the proof, not just your verdict.

If a row fails or surprises, decide before continuing: fix-and-rerun, file-and-skip, or stop and ask. Don't paper over by re-running with different flags until the symptom goes away.

For comparison rows (`--project` override, `list` vs `get`, text vs json), capture both calls in the same row and show them adjacent so the reader can eyeball the difference.

Long-running commands (`serve`, `logs --follow`) go in the background with `run_in_background: true`. Stop them at end of row before moving on.

## Report

The report is a design and interaction review surface, not a test result. The reader should be able to scroll through it and form their own opinion: are the columns aligned, is the field naming consistent, does this command feel right, is the error message clear. Your job is to put the real evidence in front of them, not to pre-judge it.

Write to `thoughts/shared/qa/<date>-llamactl-<scope>.md`. Post a short pointer inline (one or two sentences plus the file link). The user reads the file, not your inline summary.

Template:

````
# llamactl QA: <one-line scope>

Backend: <prod-test-project | local kind | other>
Profile: <name> (<env URL>)
Project: <test project id/name>
Branch: <branch>  Commit: <short sha>

## Highlights
<2–4 bullets pointing at things the reader should look at: surprises, design questions, anything that needs a decision. Do not summarize "what passed". Lean toward open questions, not conclusions.>

## Rows

### 1. <one-line description, e.g. "deployments get -o json (list mode)">

Command:
```
$ <exact command>
```

Output (exit=<n>, <elapsed>):
```
<verbatim stdout>
```
<if stderr non-empty:>
Stderr:
```
<verbatim stderr>
```

<if you ran an assertion:>
Check:
```
$ <assertion command>
<verbatim assertion output>
```

<if anything design-relevant:>
Notes: <bullet or two on what the reader should notice>

### 2. <next row>
... same shape ...

## Followups

- <questions surfaced by the run that need a human answer>
- <regressions or design choices that should turn into issues / changes>
````

Rules:

- **Show, don't tell.** "JSON keys are id, status, git_sha" → no. Paste the JSON. The reader can see the keys.
- **Keep verbatim formatting.** Tables in fenced blocks preserve the column rendering. Don't reformat or "clean up" output.
- **Comparison rows live in one section.** Two adjacent fenced blocks labeled "active" and "override" beat two separate rows.
- **Highlights surface decisions, not pass/fail.** "`auth env list -o json` includes `min_llamactl_version` — keep in the public contract or drop?" beats "all rows pass".
- **No celebration.** If everything looks reasonable, the report just looks reasonable. Don't add "all green" framing.

The inline chat reply is one or two sentences pointing at the file. Don't restate the report.

## Common gotchas

- Stale profile. `llamactl auth list` first, every session. A profile pointing at last week's env produces 401s that look like a code bug.
- Active project drift. A test that "works on my machine" can fail elsewhere because the active project differs. Pin every prod row with `--project`.
- Non-test project mutation. If you mutate without `--project`, the active profile decides where it lands. Always pin write commands.
- TUI launches in non-tty. Some commands open a textual app when stdin is a TTY. Pass `--no-interactive` or `-o json` for QA.
- `serve` vs tilt. `llamactl serve` runs the appserver locally with no kubernetes. Tilt runs the full cloud stack in kind. Different scopes; pick one.
- Don't leak IDs. Deployment and org IDs are fine in your terminal and in `thoughts/`, not fine in a public PR description. Strip or alias them in anything that might leave the repo.

## Self-update

llamactl's command surface is moving. When a run teaches you a fact this skill doesn't have, edit it.

Triggers for an edit:

- A flag name, command path, or output shape was different from what the skill implied.
- A failure mode showed up that isn't in "Common gotchas".
- A backend setup step (especially local tilt) needed something the skill didn't say.

What to add:

- New gotchas go to "Common gotchas" as one-line entries.
- New mental-model facts go to "llamactl mental model".
- Tilt or backend changes go to "Pick a backend".

Edit the file at `.agents/skills/llamactl-qa/SKILL.md` (the `.claude/skills/` entry is a symlink into here, maintained by `uv run dev sync-skills`). Keep the voice terse; don't pad. If you're unsure whether the fact is durable or just incidental to one run, leave it out. False additions are worse than missing ones.
