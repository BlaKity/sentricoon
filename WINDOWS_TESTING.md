# Windows Testing Guide

This guide explains what Sentricoon can do on Windows today and how to test it
safely.

Sentricoon is currently best used on Windows for:

- safe diagnostics
- dry-run repair planning
- read-only system inspection
- early confirmation-gated repair experiments

It is not yet a full Windows GUI repair agent.

## Current Windows Capability

Sentricoon can currently:

- detect system information
- list processes using `tasklist`
- run planned PowerShell or `cmd.exe` commands
- read, write, and delete files through controlled tools
- kill processes only behind confirmation
- dry-run risky actions
- create logs and reports
- snapshot some file state before risky file actions

Sentricoon cannot fully do these yet:

- click through Windows Settings
- use Device Manager UI
- handle permission popups visually
- inspect screenshots
- reliably repair drivers
- guarantee rollback for services or registry changes
- act like a polished Windows support app

## Prerequisites

Install:

- Git
- Python 3.10+
- `uv`

Clone the repo:

```powershell
git clone https://github.com/BlaKity/sentricoon.git
cd sentricoon
```

## Basic Smoke Tests

Run the test suite:

```powershell
uv run python -m unittest discover
```

View CLI help:

```powershell
uv run python -m sentricoon.main --help
```

These two commands should pass before trying real tasks.

## Configure API Key

Set your cloud LLM API key for the current PowerShell session:

```powershell
$env:SENTRICOON_CLOUD_API_KEY = "your-api-key"
```

Alternative OpenAI-compatible settings:

```powershell
$env:SENTRICOON_CLOUD_BASE_URL = "https://api.openai.com/v1"
$env:SENTRICOON_CLOUD_MODEL = "gpt-4o-mini"
```

## Safe Read-Only Diagnostics

Start with simple read-only tasks:

```powershell
uv run python -m sentricoon.main "report the OS version and CPU architecture"
```

```powershell
uv run python -m sentricoon.main "show system information"
```

```powershell
uv run python -m sentricoon.main "list running processes"
```

## Dry-Run Repair Planning

Dry-run is the best Windows mode right now. It lets Sentricoon plan and
simulate write-effect actions without changing your system.

Audio diagnosis:

```powershell
uv run python -m sentricoon.main --dry-run "diagnose why Windows Audio is not running"
```

Network diagnosis:

```powershell
uv run python -m sentricoon.main --dry-run "diagnose why my internet connection is not working"
```

Slow computer diagnosis:

```powershell
uv run python -m sentricoon.main --dry-run "check why my computer is slow"
```

Low disk space diagnosis:

```powershell
uv run python -m sentricoon.main --dry-run "diagnose why my C drive is low on space"
```

## Real Repair Attempts

After dry-run testing, you can try simple real tasks without `--yes`:

```powershell
uv run python -m sentricoon.main "diagnose why Windows Audio is not running"
```

If Sentricoon proposes a risky action, it should ask for confirmation.

Do not start Windows testing with:

```powershell
uv run python -m sentricoon.main --yes "fix my audio"
```

`--yes` skips confirmation prompts. Use it only after you trust the generated
plan and understand the machine state.

## Recommended First Test Sequence

```powershell
git clone https://github.com/BlaKity/sentricoon.git
cd sentricoon

uv run python -m unittest discover
uv run python -m sentricoon.main --help

$env:SENTRICOON_CLOUD_API_KEY = "your-api-key"

uv run python -m sentricoon.main "report the OS version and CPU architecture"
uv run python -m sentricoon.main --dry-run "diagnose why Windows Audio is not running"
uv run python -m sentricoon.main --dry-run "diagnose why my internet is not working"
```

## What To Watch For

During Windows testing, check:

- Does the planner choose Windows-native commands?
- Does it avoid Linux commands like `systemctl`?
- Does it use `cmd.exe /c ...` or `powershell.exe -NoProfile -Command ...` for
  complex shell tasks?
- Does it ask before risky actions?
- Does dry-run avoid real mutations?
- Does verification clearly explain whether the step worked?
- Does escalation produce a useful report when it cannot continue?

## Current Project Level

Sentricoon is currently a **serious prototype / early alpha**.

The next target is **developer alpha**:

```text
A technical user can clone it, run it with uv, test it on Windows/Linux,
perform safe diagnostics, and understand every action it proposes.
```

Windows testing is important, but Ubuntu is currently the first development
target.
