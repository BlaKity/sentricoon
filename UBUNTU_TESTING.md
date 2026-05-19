# Ubuntu Testing Guide

This guide explains how to test Sentricoon safely on Ubuntu.

Ubuntu is the best first developer-alpha target because common repair surfaces
are inspectable through stable CLI tools such as `systemctl`, `journalctl`,
`ip`, `df`, `free`, and `ss`.

## Current Ubuntu Capability

Sentricoon can currently:

- detect system information
- list processes
- run controlled shell commands with confirmation
- read, write, and delete files through controlled tools
- dry-run risky actions
- create logs and reports
- snapshot some file state before risky file actions
- inspect failed systemd units through `ubuntu.systemd.failed_units`
- inspect a specific systemd unit through `ubuntu.systemd.status`
- restart a systemd unit through `ubuntu.systemd.restart` with confirmation

Sentricoon cannot fully do these yet:

- repair every package-manager failure
- guarantee rollback for service changes
- understand every journal error automatically
- operate graphical Ubuntu Settings
- act as a polished desktop support app

## Prerequisites

Install:

- Git
- Python 3.10+
- `uv`

Clone the repo:

```bash
git clone https://github.com/BlaKity/sentricoon.git
cd sentricoon
```

## Basic Smoke Tests

Run the test suite:

```bash
uv run python -m unittest discover
```

View CLI help:

```bash
uv run python -m sentricoon.main --help
```

These two commands should pass before trying real tasks.

## Configure API Key

Set your cloud LLM API key:

```bash
export SENTRICOON_CLOUD_API_KEY="your-api-key"
```

Optional OpenAI-compatible settings:

```bash
export SENTRICOON_CLOUD_BASE_URL="https://api.openai.com/v1"
export SENTRICOON_CLOUD_MODEL="gpt-4o-mini"
```

## Safe Read-Only Diagnostics

Start with simple read-only tasks:

```bash
uv run python -m sentricoon.main "report the OS version and kernel"
```

```bash
uv run python -m sentricoon.main "show system information"
```

```bash
uv run python -m sentricoon.main "list running processes"
```

## Ubuntu Systemd Diagnostics

Failed services:

```bash
uv run python -m sentricoon.main "diagnose failed systemd services"
```

Specific service:

```bash
uv run python -m sentricoon.main "check the status of ssh.service"
```

Dry-run repair planning:

```bash
uv run python -m sentricoon.main --dry-run "diagnose and repair failed systemd services"
```

## Other Dry-Run Repair Planning

Disk usage:

```bash
uv run python -m sentricoon.main --dry-run "diagnose disk usage"
```

Memory pressure:

```bash
uv run python -m sentricoon.main --dry-run "diagnose memory usage"
```

Network connectivity:

```bash
uv run python -m sentricoon.main --dry-run "diagnose network connectivity"
```

DNS:

```bash
uv run python -m sentricoon.main --dry-run "diagnose DNS resolution"
```

## Real Repair Attempts

Start with dry-run. For real repair attempts, avoid `--yes` until you trust the
generated plan.

Example:

```bash
uv run python -m sentricoon.main "restart ssh.service if it is failed"
```

If Sentricoon proposes `ubuntu.systemd.restart`, it should ask for confirmation.

Do not start with:

```bash
uv run python -m sentricoon.main --yes "fix failed services"
```

`--yes` skips confirmation prompts.

## Recommended First Test Sequence

```bash
git clone https://github.com/BlaKity/sentricoon.git
cd sentricoon

uv run python -m unittest discover
uv run python -m sentricoon.main --help

export SENTRICOON_CLOUD_API_KEY="your-api-key"

uv run python -m sentricoon.main "report the OS version and kernel"
uv run python -m sentricoon.main "diagnose failed systemd services"
uv run python -m sentricoon.main --dry-run "diagnose disk usage"
uv run python -m sentricoon.main --dry-run "diagnose network connectivity"
```

## What To Watch For

During Ubuntu testing, check:

- Does the planner prefer `ubuntu.systemd.*` actions over generic `shell.run`
  for service tasks?
- Does it diagnose before restarting a service?
- Does it ask before `ubuntu.systemd.restart`?
- Does dry-run avoid real mutations?
- Does verification clearly explain whether the step worked?
- Does escalation produce a useful report when it cannot continue?

## Current Milestone

The next milestone is **Ubuntu Developer Alpha**:

```text
A technical user can clone Sentricoon, run it with uv, safely diagnose common
Ubuntu problems, and understand every action it proposes.
```
