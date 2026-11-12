# Agent Skills

`alpaca` supports reusable procedural **Agent Skills** (`SKILL.md` folders) with progressive loading so only short metadata consumes context tokens until the model invokes a skill.

## Directory Layout

Skills live inside `skills/<skill-name>/SKILL.md` in your workspace (or app root):

```text
skills/
  code-review/
    SKILL.md
  create-skill/
    SKILL.md
  scaffold-project/
    SKILL.md
```

Each `SKILL.md` begins with YAML front matter followed by concise Markdown instructions:

```markdown
---
name: my-workflow
description: One-line summary telling the assistant when to call use_skill.
---

# Workflow Steps
1. ...
```

## Starter Skills

On first agent initialization, `alpaca` seeds three starter skills:

- `create-skill` — How to structure, validate, and stage reusable skills via `save_skill`.
- `scaffold-project` — Checklist for initializing clean projects with docs and smoke tests.
- `code-review` — Structured code review workflow for correctness, security, and tests.

If you delete a starter skill, `alpaca` moves it to Trash cleanly and does not re-seed it unexpectedly.

## How Models Use Skills

1. **Discovery**: Enabled skills appear in `<available_skills>` inside the system prompt (name + description only).
2. **Activation**: The model calls `use_skill` with the skill's exact `name` to load full `SKILL.md` steps on demand, and `read_file` for any companion files inside `skills/<name>/`.
3. **Creation (`save_skill`)**: When the model discovers a repeatable procedure, it may propose a new skill via `save_skill`. All proposals stage to `.alpaca/skills-staging/` and require **explicit user approval** before writing to `skills/`.

## Managing Skills in Chat

- `/skills` — Open interactive skills manager (view, enable/disable, rename, create, or delete).
- `/skills reload` — Rescan `skills/` folders immediately.
- `/skill <name>` — Preview or inspect a specific skill.
