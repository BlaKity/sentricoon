# Contributing

Sentricoon targets both Linux and Windows. The current implementation focus is
Ubuntu first so the safety, verification, and plugin loop can become reliable
before the Windows surface expands.

## Development Setup

Use `uv` from the repo root:

```bash
uv run python -m unittest discover
uv run python -m sentricoon.main --help
```

Run benchmark tests:

```bash
uv run python -m unittest tests.test_benchmarks_ubuntu
```

## Safety Rules For New Tools

New tools must preserve Sentricoon's safety model:

- Add every action to `sentricoon/safety/policies.py`.
- Add every planner-visible action to `sentricoon/catalog.py`.
- Register native tools in `sentricoon/main.py`.
- Mark write-effect tools with `is_read_only = False`.
- Implement `dry_run_describe()` for every write-effect tool.
- Add risky tools to `REQUIRES_CONFIRMATION`.
- Add tests for allowlist, confirmation, dry-run, and routing behavior.

Do not add generic escape-hatch actions such as:

```text
run_anything
execute_python
repair_all
```

Prefer narrow named actions:

```text
ubuntu.systemd.failed_units
ubuntu.systemd.status
ubuntu.systemd.restart
```

## Verification Expectations

Tools should return evidence that can be verified. A diagnostic action should
not fail merely because the diagnosed system state is unhealthy.

Example: `systemctl status` returns exit code `3` for failed units. For
`ubuntu.systemd.status`, that is successful diagnostic evidence, not a tool
crash.

## Pull Request Checklist

Before opening a PR:

- Run `uv run python -m unittest discover`.
- Add or update focused tests.
- Add benchmark coverage for safety-critical behavior.
- Update docs if user-facing behavior changed.
- Keep dependencies minimal; the current runtime is stdlib-only.

## Scope

Good contributions:

- Ubuntu diagnostics
- safety and verification improvements
- behavior benchmarks
- clearer reports
- plugin metadata and loader work

Out of scope for now:

- broad general-assistant features
- unrestricted shell wrappers
- unattended high-risk repair flows
- GUI automation without explicit guardrails
