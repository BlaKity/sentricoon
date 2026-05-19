"""Planner — turns a task string into a list[Step] via the cloud LLM.

Routes via LLMRole.PLANNER (cloud, infrequent, expensive). The planner
consults procedural memory's quarantine list BEFORE generating, so the
prompt tells the LLM which approaches have already failed on this machine.

Plan validation is strict: anything outside the action allowlist is
rejected, and every step MUST commit to a concrete expected_state — that
predicate is what the SemanticVerifier checks against. A planner that
emits vague expected_state values would let the verifier rubber-stamp
broken steps. Catching this here fails fast and cheap.
"""

from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass

from .catalog import catalog_for_prompt
from .llm.role import LLMRole
from .llm.router import LLMRouter
from .llm.types import LLMRequest, Message
from .memory import Memory
from .safety.policies import ACTION_ALLOWLIST
from .state import Step


class PlanError(Exception):
    """Raised when the planner produces an invalid or unusable plan."""


# Per-OS command guidance the planner LLM sees. The same agent code runs
# on Linux, Windows, and macOS; what changes is which command vocabulary
# the planner emits for `shell.run` steps. Without these hints, an LLM
# without strong OS context can pick the wrong family (`net start` on
# Linux, `systemctl` on Windows, etc.) and every shell step fails.
_OS_COMMAND_HINTS: dict[str, str] = {
    "Linux": (
        "  - Service management: systemctl status/start/restart <name>, "
        "systemctl --user for per-user services\n"
        "  - Process info: ps aux, top, journalctl -u <name>\n"
        "  - Package status: dpkg -l, apt list --installed, rpm -qa\n"
        "  - Audio (PipeWire/PulseAudio): pactl list, "
        "systemctl --user status pipewire, pavucontrol\n"
        "  - Network: ss -tulpn, ip addr, networkctl status\n"
    ),
    "Windows": (
        "  - Service management: sc query <name>, net start <name>, "
        "net stop <name>, sc start <name>\n"
        "  - PowerShell variant: Get-Service <name>, "
        "Restart-Service <name> (requires elevation for system services)\n"
        "  - Process info: tasklist, tasklist /FI \"IMAGENAME eq X\"\n"
        "  - Audio: Get-Service Audiosrv, net start Audiosrv, net start AudioEndpointBuilder\n"
        "  - For complex commands prefer: cmd.exe /c \"...\" or "
        "powershell.exe -NoProfile -Command \"...\"\n"
    ),
    "Darwin": (
        "  - Service management: launchctl list, launchctl bootstrap, launchctl kickstart\n"
        "  - Process info: ps aux, top, log show --predicate '...'\n"
        "  - Audio: SystemSoundServer is managed by launchd; check /Library/Audio/\n"
        "  - Package status: brew list, pkgutil --pkgs\n"
    ),
}

_GENERIC_HINT = "  - (no OS-specific command guidance available — fall back to POSIX commands)\n"


def host_context_block(
    *,
    system: str | None = None,
    platform_string: str | None = None,
    machine: str | None = None,
) -> str:
    """Format host environment context for the planner system prompt.

    All args optional; defaults read from the `platform` module. Tests
    inject specific values to assert per-OS prompt content.
    """
    system = system or platform.system()
    platform_string = platform_string or platform.platform()
    machine = machine or platform.machine()
    hints = _OS_COMMAND_HINTS.get(system, _GENERIC_HINT)
    return (
        f"HOST CONTEXT\n"
        f"  Operating system: {system}\n"
        f"  Platform: {platform_string}\n"
        f"  Architecture: {machine}\n"
        f"\n"
        f"When running shell.run, prefer commands native to {system}:\n"
        f"{hints}"
    )


SYSTEM_PROMPT_TEMPLATE = """You are the planner for an OS-Technician AI agent on a personal computer.

Your job: turn the user's task into a JSON plan. The executor will run each step,
the verifier will check it, and the diagnoser will analyze any failures.

STRICT RULES
1. Use ONLY actions from the allowlist below. Inventing actions will be rejected.
2. Every step MUST include "expected_state" — a concrete, observable predicate
   that names a specific value, regex, or comparison the verifier can check
   without an LLM.

   ACCEPT (specific value / regex / existence check):
     - "exit code is 0 AND stdout contains 'Active: active (running)'"
     - "system.info output has key 'system' equal to 'Linux'"
     - "free -h stdout has a line starting with 'Mem:' followed by a size"
     - "file '/etc/resolv.conf' exists and contains the word 'nameserver'"
     - "process.list output contains an entry with name matching /pipewire/"

   REJECT (cannot be checked — verifier would rubber-stamp anything):
     - "kernel version is displayed"          -> displayed WHERE? what value?
     - "OS information is retrieved"          -> tautological with success=True
     - "memory usage is shown"                -> no checkable assertion
     - "the command runs successfully"        -> that's just exit code == 0
     - "service works"                        -> name a state or output
     - "should be running"                    -> hopeful, not assertive
     - any predicate shorter than 30 chars    -> too vague

   When in doubt, write the predicate as the regex / value comparison you'd
   put in an assertion. If you can't, your predicate is too weak.

3. Plan MINIMALLY. Only include steps that directly contribute to the user's
   stated request. Do not add diagnostic or "context" steps that gather data
   the user did not ask for.

   Examples of over-planning:
     Task: "How much memory is in use?"
       BAD:  [system.info, process.list, shell.run free -h]
             -- process.list adds nothing; the user did not ask
                which processes use memory.
       GOOD: [shell.run free -h]   (one step, answers the question directly)

     Task: "Report the OS distribution and kernel version."
       BAD:  [system.info, shell.run uname -r]
             -- system.info already contains release (kernel version).
       GOOD: [system.info]   (one step covers both)

     Task: "Fix the audio service."  (repair, not info)
       OK to add diagnostic steps: list services, check audio service status,
       THEN apply a fix. Repair tasks need observation before mutation.

   Rule of thumb: if removing a step would change what the user sees as the
   answer, keep it. If it would not, drop it.

4. For REPAIR tasks (the user asked to fix / restart / configure something),
   diagnose before mutating: read state first, then apply write-effect steps.
   For INFO tasks (the user asked for a report / status / value), do NOT
   include diagnostic steps that don't bear on the answer.

5. Prefer reversible operations. Mark each step with risk_level: low | medium | high.

6. AVOID privilege escalation unless the user explicitly authorized it.
   For read-only inspection, prefer non-sudo / non-elevated commands.
   If some files are inaccessible, accept partial results and report them.
   Do NOT silently retry with sudo when permission errors occur. If sudo
   really is required, mark risk_level: "high" and explain why in
   expected_state. (On Windows: avoid running unprompted as Administrator.)

7. Output format MUST be valid JSON.

   The `args` field is ALWAYS a JSON object with named keys — NEVER a bare
   list. Each action has its own expected keys:

     shell.run        -> args: {{"argv": ["echo", "hello"]}}
                          (a list named 'argv' INSIDE the args object;
                           use OS-specific commands from the HOST CONTEXT
                           block below, not literally 'echo hello')

     file.read        -> args: {{"path": "/etc/resolv.conf"}}
     file.write       -> args: {{"path": "/tmp/x.txt", "content": "hello"}}
     file.delete      -> args: {{"path": "/tmp/x.txt"}}

     process.kill     -> args: {{"pid": 1234}}

     system.info      -> args: {{}}    (empty object, NEVER null, NEVER omit)
     process.list     -> args: {{}}

   Common mistakes to NOT make:
     BAD:  "args": ["some", "command", "tokens"]    <- list, not object; wrap in {{"argv": [...]}}
     BAD:  "args": "/tmp/x.txt"                      <- string, not object
     BAD:  "args": null                              <- use {{}} instead

   Full step shape:
   {{
     "action": "<name>",
     "args": {{<action-specific keys>}},
     "expected_state": "<predicate>",
     "risk_level": "low" | "medium" | "high"
   }}

   Top-level response:
   {{"steps": [<step1>, <step2>, ...]}}

{host_context}
ACTION CATALOG
{catalog}

CONTEXT
Goal class: {goal_class}
{quarantine_block}
"""


# Heuristic patterns for weak expected_state predicates. Used by
# `weak_predicate_warnings()` — emits warnings, never rejects the plan.
# The list is intentionally tight: false-positives are worse than misses
# because rejecting good predicates costs a real LLM call to fix.
_WEAK_PREDICATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bis\s+(displayed|shown|reported|retrieved|returned|printed)\b", re.IGNORECASE),
        "passive '{m}' — what value? checked how?",
    ),
    (
        re.compile(r"\b(works|succeeds|completes|runs)(\s+(successfully|normally|correctly|fine))?\b", re.IGNORECASE),
        "generic '{m}' — tautological with returncode == 0",
    ),
    (
        re.compile(r"\b(looks|seems|appears)\s+(good|normal|correct|right|fine|ok)\b", re.IGNORECASE),
        "subjective '{m}' — no concrete observable",
    ),
    (
        re.compile(r"\bshould\b", re.IGNORECASE),
        "hopeful 'should' — name the actual condition, not the intent",
    ),
]


_WEAK_PREDICATE_MIN_LEN = 30


def weak_predicate_warnings(plan: list[Step]) -> list[str]:
    """Scan a plan for weak `expected_state` predicates.

    Returns a list of warning strings. Empty when every predicate looks
    concrete. Warnings only — the agent does not reject the plan, but
    surfaces them in the audit log and terminal output so the user sees
    when the verifier is going to rubber-stamp ambiguous steps.
    """
    warnings: list[str] = []
    for i, step in enumerate(plan, start=1):
        pred = (step.expected_state or "").strip()
        if len(pred) < _WEAK_PREDICATE_MIN_LEN:
            warnings.append(
                f"step {i} ({step.action}): expected_state {pred!r} "
                f"is too short ({len(pred)} chars) — name a specific observable"
            )
            continue
        for pattern, reason_template in _WEAK_PREDICATE_PATTERNS:
            match = pattern.search(pred)
            if match:
                reason = reason_template.format(m=match.group(0))
                warnings.append(
                    f"step {i} ({step.action}): expected_state {pred!r} — {reason}"
                )
                break  # one warning per step is enough
    return warnings


# ---- Risk warnings (separate concern from predicate quality) ----

_PRIVILEGE_ESCALATION_TOKENS: frozenset[str] = frozenset({
    "sudo",       # Linux / macOS
    "su",         # Linux
    "doas",       # *BSD / occasional Linux
    "runas",      # Windows
    "gsudo",      # Windows (community sudo equivalent)
})


def plan_risk_warnings(plan: list[Step]) -> list[str]:
    """Scan a plan for risky patterns that don't show up as weak predicates.

    Currently detects:
      - shell.run steps whose argv begins with a privilege-escalation token
        (sudo, su, doas, runas, gsudo). Even when the user authorized
        elevation in the task description, surfacing the fact that THIS
        plan uses it gives the user a chance to deny at the prompt.

    Extensible: other risks (raw network commands, rm -rf, registry
    writes on Windows) can be added here as patterns surface.
    """
    warnings: list[str] = []
    for i, step in enumerate(plan, start=1):
        if step.action != "shell.run":
            continue
        argv = step.args.get("argv")
        if not isinstance(argv, list) or not argv:
            continue
        first = str(argv[0]).lower()
        if first in _PRIVILEGE_ESCALATION_TOKENS:
            preview = " ".join(str(a) for a in argv)[:80]
            warnings.append(
                f"step {i} (shell.run): uses privilege escalation ({argv[0]!r}) — "
                f"`{preview}`. Approve only if intentional."
            )
    return warnings


@dataclass
class _ParsedPlan:
    steps: list[Step]


class Planner:
    def __init__(self, router: LLMRouter, *, memory: Memory | None = None) -> None:
        self.router = router
        self.memory = memory or Memory()

    # ----- public API -----

    def create_plan(self, task: str, goal_class: str | None = None) -> list[Step]:
        prompt = self._build_system_prompt(goal_class)
        return self._call_and_parse(prompt, user_content=task, temperature=0.2)

    def replan(
        self,
        task: str,
        failed_step: Step,
        diagnosis: dict,
        *,
        goal_class: str | None = None,
    ) -> list[Step]:
        prompt = self._build_system_prompt(goal_class or "replan")
        user_content = (
            f"Original task: {task}\n\n"
            f"A previous step FAILED:\n"
            f"  action: {failed_step.action}\n"
            f"  args: {json.dumps(failed_step.args, default=str)}\n"
            f"  expected_state: {failed_step.expected_state}\n\n"
            f"Diagnosis: {json.dumps(diagnosis, default=str)}\n\n"
            "Generate a NEW plan that avoids this failure mode. Do not repeat "
            "the failed step verbatim. Prefer a different approach."
        )
        return self._call_and_parse(prompt, user_content=user_content, temperature=0.3)

    # ----- internals -----

    def _build_system_prompt(self, goal_class: str | None) -> str:
        system = platform.system()
        sequences = self.memory.quarantined_action_sequences()
        if sequences:
            lines = ["Avoid these failed approaches (action sequences known to fail on this machine):"]
            for seq in sequences:
                lines.append(f"  - {' -> '.join(seq)}")
            quarantine_block = "\n".join(lines)
        else:
            quarantine_block = "No prior failed approaches recorded for this machine."
        return SYSTEM_PROMPT_TEMPLATE.format(
            host_context=host_context_block(system=system),
            catalog=catalog_for_prompt(system=system),
            goal_class=goal_class or "unspecified",
            quarantine_block=quarantine_block,
        )

    def _call_and_parse(self, system_prompt: str, *, user_content: str, temperature: float) -> list[Step]:
        req = LLMRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ],
            temperature=temperature,
            json_mode=True,
        )
        try:
            resp = self.router.complete(LLMRole.PLANNER, req)
        except Exception as e:
            raise PlanError(f"planner backend unreachable: {e}") from e

        return _parse_and_validate(resp.content)


def _parse_and_validate(content: str) -> list[Step]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise PlanError(f"planner returned non-JSON: {content[:200]!r}") from e

    if not isinstance(parsed, dict):
        raise PlanError(f"planner response is not a JSON object: got {type(parsed).__name__}")

    if "steps" not in parsed:
        raise PlanError("planner response missing required 'steps' key")

    steps_raw = parsed["steps"]
    if not isinstance(steps_raw, list):
        raise PlanError(f"'steps' must be a list, got {type(steps_raw).__name__}")

    if not steps_raw:
        raise PlanError("plan is empty — agent has nothing to do")

    return [_validate_step(raw, i) for i, raw in enumerate(steps_raw)]


def _validate_step(raw: object, idx: int) -> Step:
    if not isinstance(raw, dict):
        raise PlanError(f"step {idx} is not a JSON object: {raw!r}")

    action = raw.get("action")
    if not isinstance(action, str) or not action:
        raise PlanError(f"step {idx} missing non-empty 'action' string")

    if action not in ACTION_ALLOWLIST:
        raise PlanError(
            f"step {idx} uses action {action!r} not in allowlist. "
            f"Allowed: {sorted(ACTION_ALLOWLIST)}"
        )

    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise PlanError(f"step {idx} 'args' must be an object, got {type(args).__name__}")

    expected_state = raw.get("expected_state")
    if not isinstance(expected_state, str) or not expected_state.strip():
        raise PlanError(
            f"step {idx} ({action}) missing non-empty 'expected_state' — "
            "every step must commit to a concrete observable predicate"
        )

    risk_level = raw.get("risk_level", "low")
    if risk_level not in {"low", "medium", "high"}:
        raise PlanError(f"step {idx} risk_level must be low/medium/high, got {risk_level!r}")

    return Step(
        action=action,
        args=args,
        expected_state=expected_state.strip(),
        risk_level=risk_level,
    )
