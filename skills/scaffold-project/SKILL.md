---
name: scaffold-project
description: "Checklist for scaffolding clean, testable project layouts with starter docs and checks."
---

# Scaffold Project

Follow this sequence when asked to initialize a new project or module cleanly inside the active workspace:

1. Inspect existing workspace files with `list_dir` so you do not overwrite existing work unintentionally.
2. Confirm language/runtime, entrypoints, and test setup if ambiguous using `ask_user`.
3. Create directories cleanly with `make_dir` and write minimal idiomatic entry files plus a small README and smoke test.
4. Verify build/tests via `run_command` before finishing and report exact verified paths and command outputs.
