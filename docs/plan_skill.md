# Plan Skill

> **When to use:** Any task with 3+ distinct steps, or when the user says
> "plan", "project", "set up", "migrate", "refactor", etc.

## Plan format

Plans are saved as Markdown files (default: `plan.md` in the working directory).
Each plan has a title, optional description, and a numbered task list with
status checkboxes.

```markdown
# <Plan Title>

<Optional one-line description>

## Tasks

- [ ] 1. First task
- [ ] 2. Second task
- [x] 3. Completed task
- [>] 4. Currently active task
- [ ] 5. Future task
```

**Status markers:**
| Marker | Meaning |
|--------|---------|
| `[ ]`  | Pending |
| `[>]`  | Active (currently working on) |
| `[x]`  | Completed |
| `[-]`  | Skipped |

Only **one** task should be `[>]` at a time.

## Using the `plan_update` tool

You have a `plan_update` tool with four actions:

### `create` — Start a new plan

```json
{
  "action": "create",
  "title": "Set up Python project with CI",
  "tasks": [
    "Initialize git repo and .gitignore",
    "Create pyproject.toml with dependencies",
    "Set up pytest and write initial tests",
    "Add GitHub Actions CI workflow",
    "Run tests and verify CI passes"
  ],
  "file": "plan.md"
}
```

The first task is automatically marked `[>]` (active).

### `next` — Mark current task done, advance to the next

```json
{
  "action": "next",
  "note": "Repo initialized with Python .gitignore"
}
```

Optional `note` is appended to the completed task for context.

### `complete` — Mark the entire plan complete

```json
{
  "action": "complete",
  "note": "All tasks done, CI is green"
}
```

### `status` — Read current plan state

```json
{
  "action": "status"
}
```

Returns the current plan Markdown and scratchpad summary.

## Scratchpad integration

`plan_update` automatically maintains these scratchpad keys:

| Key | Value |
|-----|-------|
| `plan_file` | Path to the current plan file |
| `current_task` | Text of the currently active task |
| `task_status` | e.g. `3/7 — Set up pytest` |

These survive conversation compaction, so you always know where you are.

## Guidelines

1. **Create plans proactively** for multi-step work — don't wait to be asked.
2. **Call `next` after completing each task** before starting the next one.
3. **Keep tasks atomic** — each should be completable in 1-3 tool rounds.
4. **5-12 tasks** is ideal. More than 15 means you should split into sub-plans.
5. **Use `status`** if you've lost context (e.g. after compaction) to re-orient.
6. When the user gives vague instructions, create a plan first, show it, then
   ask "Shall I proceed?" before executing.
