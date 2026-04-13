# OmniChat Agent Rules

## Python Environment

- Always use the repository-local `.venv` at the workspace root for Python commands, package installs, tests, and scripts.
- Do not depend on the VS Code selected interpreter or any external virtual environment.
- On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\pip.exe` explicitly when running Python or pip commands.
- Treat `.venv-1` as non-default and ignore it unless the user explicitly asks to use it.
- If `.venv` is missing or broken, stop and report that as the blocker instead of silently falling back to another interpreter.

## Operational Expectation

- Any future setup, automation, or command examples for this repo should assume the root `.venv` is the only supported Python environment.