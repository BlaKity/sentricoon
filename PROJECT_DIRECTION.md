# Sentricoon Project Direction

## Core Vision

Sentricoon should become a **safe, plugin-powered OS technician agent**.

It should not become a general assistant, chatbot, AutoGPT clone, or broad
computer-use agent. Its strongest identity is narrower:

> Sentricoon diagnoses computer problems, proposes safe repair plans, executes
> approved actions, verifies results, and escalates when repair should leave
> agent control.

In practice, Sentricoon should feel like a cautious junior IT technician running
locally on a machine.

## North Star

Sentricoon is not an assistant that happens to run commands.

It is a repair loop with:

- safety
- verification
- repair
- memory
- rollback
- escalation
- plugin-based OS skills

The core loop is:

```text
TASK
  -> PLAN
  -> EXECUTE SAFELY
  -> VERIFY
  -> DIAGNOSE FAILURE
  -> REPAIR OR REPLAN
  -> ESCALATE WITH EVIDENCE
```

## Keep The Core Small

The core should own the universal control loop, not every OS-specific repair
trick.

Core responsibilities:

- decide whether an action is allowed
- enforce risk policy
- request confirmation
- support dry-run mode
- route tools
- execute actions
- verify results
- retry or replan when appropriate
- stop on unsafe or repeated failure
- write logs, snapshots, and reports

OS-specific repair knowledge should live in plugins.

## Plugin Direction

Plugins should extend Sentricoon with declared tools, not arbitrary execution.

A plugin should mean:

```text
declared tools + metadata + risk level + permissions + verification hints
```

This is a **tool-plugin system**, not a general extension system.

Good plugin action names:

```text
ubuntu.systemd.failed_units
ubuntu.systemd.status
ubuntu.network.dns_check
windows.audio.status
windows.audio.restart_services
support.bundle.create
```

Bad plugin action names:

```text
run_anything
execute_python
do_task
repair_pc
```

Each plugin action should declare:

- action name
- description
- supported OS
- input schema
- output schema
- risk level
- whether it is read-only
- whether it requires confirmation
- dry-run behavior
- verification hint

Important rule:

> Plugins may extend Sentricoon's tools, but they must never bypass
> SafetyGuard, confirmation, dry-run, verification, or escalation.

## Ubuntu-First Focus

Ubuntu should be the first developer-alpha target.

Why Ubuntu first:

- the current development environment is Ubuntu
- Linux has clean CLI diagnostic surfaces
- services, logs, packages, disk, network, and processes are inspectable
- verification is easier with `systemctl`, `journalctl`, `ip`, `df`, `free`,
  and `ss`
- useful repair plugins can be built quickly

Good first Ubuntu workflows:

- system info report
- failed services report
- disk usage diagnosis
- memory usage report
- high CPU process diagnosis
- network status diagnosis
- DNS diagnosis
- package-manager health check
- journal error summary
- service restart with confirmation

First Ubuntu plugin groups:

```text
ubuntu.system
ubuntu.systemd
ubuntu.disk
ubuntu.network
ubuntu.logs
ubuntu.apt
support.bundle
```

## Windows Later

Windows is still a strong long-term opportunity, but it should come after the
Ubuntu developer-alpha loop is stable.

Good first Windows workflows later:

- audio not working
- network connected but no internet
- high CPU usage
- low disk space
- startup apps slowing boot
- Windows service not running
- driver/device problem report

Windows will eventually need GUI support because many repairs require Settings,
Device Manager, permission prompts, or installer windows.

## GUI Capability Later

CLI alone cannot repair every real OS problem. GUI control should come later,
after the CLI/safety/verification loop is solid.

Suggested GUI stages:

```text
Stage 1: open apps/settings only
Stage 2: screenshot observation
Stage 3: guided UI instructions
Stage 4: approved click/type actions
Stage 5: verified GUI workflows
```

Rule:

> GUI plugins must be even more restricted than CLI plugins.

## Verification As A Signature Feature

Sentricoon should be known for this principle:

> I do not assume. I check.

Examples:

After restarting a service:

```text
- verify the unit is active
- verify no new critical journal errors appeared
```

After writing a config file:

```text
- re-read the file
- compare content
```

After a network fix:

```text
- retry DNS lookup
- retry gateway ping
- retry public endpoint
```

This is what makes Sentricoon trustworthy.

## Benchmarks As The Quality Bar

Benchmarks should become part of the project identity.

For Sentricoon, benchmarks should not be generic model scores. They should ask:

> Can the agent safely diagnose and repair known OS problems, with evidence,
> without causing damage?

Important benchmark types:

- diagnostic benchmarks
- safety benchmarks
- dry-run benchmarks
- verification benchmarks
- cross-OS prompt benchmarks
- escalation benchmarks

The first benchmark suite lives in:

```text
tests/test_benchmarks_ubuntu.py
```

See [Benchmarks](BENCHMARKS.md).

## Escalation Is A Product Feature

Escalation is not failure. It is responsible behavior.

Sentricoon should stop when:

- risk is high
- admin permission is required
- the same strategy failed repeatedly
- hardware failure is likely
- driver reinstall is needed
- the user denies confirmation
- verification is ambiguous

The goal is to turn "agent failed" into "agent produced useful support
evidence."

## Practical Roadmap

### Phase 1: Ubuntu Developer Alpha

Focus:

- tests
- Ubuntu docs
- `uv` workflow
- safe CLI execution
- logs
- dry-run
- rollback
- reports
- `ubuntu.systemd` tools

Goal:

```text
Sentricoon can safely run common read-only diagnostics on Ubuntu and explain
every action it proposes.
```

### Phase 2: Plugin System

Focus:

- plugin manifest
- plugin loader
- action registration
- risk metadata
- input/output schemas
- dry-run hooks
- verification hints

Goal:

```text
New repair domains can be added without editing core agent logic.
```

### Phase 3: First Useful Ubuntu Plugins

Build:

- `ubuntu.systemd`
- `ubuntu.disk`
- `ubuntu.network`
- `ubuntu.logs`
- `ubuntu.apt`
- `support.bundle`

Goal:

```text
Sentricoon can diagnose common Ubuntu server/workstation problems safely.
```

### Phase 4: Windows Technician Mode

Focus:

- PowerShell-safe commands
- service workflows
- network workflows
- device status reports
- admin-boundary handling
- clear Windows docs

### Phase 5: GUI Awareness

Focus:

- open settings
- screenshots
- OCR/vision
- guided actions
- approval-gated clicking

### Phase 6: Local App

Focus:

- local dashboard
- plugin manager
- run history
- report export
- approval UI

## Main Rule

Do not chase general autonomy.

Chase:

```text
safe repair
verified action
plugin skill
human-readable evidence
```

That is the future worth focusing on.
