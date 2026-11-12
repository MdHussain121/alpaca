---
name: code-review
description: "Structured diff and code review focusing on correctness, edge cases, security, and tests."
---

# Code Review

When reviewing changes in the workspace:

1. Gather context using `glob`, `grep`, and `read_file` (or `git diff` via `run_command` if needed).
2. Check correctness first: nil/boundary checks, resource leaks, error propagation, concurrency hazards, and path/SSRF boundaries.
3. Verify test coverage exercises failure paths as well as happy paths.
4. Present findings grouped by severity (`Critical`, `Warning`, `Suggestion`) citing exact file paths and line ranges.
