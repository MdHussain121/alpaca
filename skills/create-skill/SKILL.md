---
name: create-skill
description: "How to structure, validate, and stage reusable Agent Skills via save_skill."
---

# Create Skill Guide

Use this guide when converting a repeatable workflow or domain playbook into a reusable Agent Skill via `save_skill`.

## Naming & Front Matter Rules
- `name`: lowercase ASCII letters, digits, and hyphens only (`^[a-z0-9-]{1,40}$`), matching folder `skills/<name>/`.
- `description`: concise one-line summary telling the assistant *when* to load the skill.
- Keep instructions clear, step-by-step, and grounded in concrete workspace commands or checks.

## Size & Safety Constraints
- Body must stay under 16 KB of Markdown.
- Up to 10 helper files (`files` map), each under 16 KB, total bundle <= 64 KB.
- Never embed secrets, API keys, or instructions to bypass permissions.
- `save_skill` always stages into `.llm/skills-staging/` for user preview and approval before saving.
