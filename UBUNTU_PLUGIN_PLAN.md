# Ubuntu Plugin Plan

Sentricoon should become Ubuntu-first before expanding into Windows GUI repair.

The immediate goal is:

> Sentricoon can safely diagnose common Ubuntu server/workstation problems and
> produce useful technician evidence without changing the machine by default.

## Plugin Principles

Ubuntu plugins should expose narrow named actions, not generic shell access.

Good:

```text
ubuntu.systemd.failed_units
ubuntu.systemd.status
ubuntu.network.dns_check
ubuntu.disk.usage
ubuntu.logs.journal_errors
```

Bad:

```text
ubuntu.run_anything
ubuntu.repair_all
ubuntu.execute_script
```

Each action should declare:

- supported OS
- input schema
- output schema
- risk level
- read-only status
- confirmation requirement
- dry-run behavior
- verification hint

## First Plugin Group: ubuntu.systemd

Implemented first because services are common, useful, and easy to verify.

Actions:

```text
ubuntu.systemd.failed_units
ubuntu.systemd.status
ubuntu.systemd.restart
```

Risk model:

```text
ubuntu.systemd.failed_units = low, read-only
ubuntu.systemd.status = low, read-only
ubuntu.systemd.restart = high, requires confirmation
```

Expected workflow:

```text
1. List failed units
2. Check status for the relevant unit
3. Restart only if appropriate and approved
4. Verify status after restart
5. Escalate if restart fails or evidence is ambiguous
```

Example:

```bash
uv run python -m sentricoon.main "diagnose failed systemd services"
```

## Next Ubuntu Plugin Groups

### ubuntu.disk

Read-only first:

```text
ubuntu.disk.usage
ubuntu.disk.inodes
ubuntu.disk.large_dirs
```

Repair later:

```text
ubuntu.disk.clean_apt_cache
ubuntu.disk.clean_journal
```

High-risk cleanup actions must require confirmation.

### ubuntu.network

Read-only first:

```text
ubuntu.network.ip_addr
ubuntu.network.routes
ubuntu.network.dns_status
ubuntu.network.ping_gateway
ubuntu.network.dns_lookup
ubuntu.network.open_ports
```

Repair later:

```text
ubuntu.network.restart_network_manager
ubuntu.network.flush_dns_cache
```

### ubuntu.logs

Read-only:

```text
ubuntu.logs.journal_errors
ubuntu.logs.unit_errors
ubuntu.logs.boot_errors
```

This plugin should produce compact evidence, not dump huge logs.

### ubuntu.apt

Read-only first:

```text
ubuntu.apt.check_locks
ubuntu.apt.pending_updates
ubuntu.apt.broken_packages
```

Repair later:

```text
ubuntu.apt.fix_broken
ubuntu.apt.autoremove
ubuntu.apt.clean_cache
```

Package repair should be high-risk and confirmation-gated.

### support.bundle

Read-only export:

```text
support.bundle.create
```

Should collect:

- system info
- failed services
- recent journal errors
- disk usage
- memory usage
- network summary
- Sentricoon run report

## Ubuntu Developer Alpha Definition

Sentricoon reaches Ubuntu Developer Alpha when:

- tests pass on Ubuntu
- README and Ubuntu testing guide are accurate
- `ubuntu.systemd` actions are wired and tested
- common diagnostics work without mutation
- risky repairs require confirmation
- dry-run previews write-effect actions
- failure reports are understandable

## Near-Term Task List

1. Stabilize `ubuntu.systemd`.
2. Add `ubuntu.disk.usage`.
3. Add `ubuntu.network.dns_status`.
4. Add `ubuntu.logs.journal_errors`.
5. Build a support bundle action.
6. Add example run outputs to docs.
