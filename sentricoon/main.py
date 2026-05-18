"""CLI entry point for sentricoon.

Three modes:
  - Default: run a task end-to-end against real LLMs.
  - --dry-run: build the plan, simulate write-effect tools, no side effects.
  - --rollback PATH: apply a snapshot to restore prior state.

Heavy lifting lives in `build_agent`, `run_task`, `run_rollback`, which
are independently testable. `main()` is just the dispatcher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from .agent import Agent, TaskOutcome
from .confirmation import AutoApproveProvider, CLIConfirmationProvider, ConfirmationProvider
from .config import MyBotConfig
from .diagnoser import Diagnoser
from .escalation import TaskBudget
from .executor import Executor
from .llm import LLMRouter, OpenAICompatBackend, RouterConfig
from .llm.openai_compat import OpenAICompatConfig
from .memory import EpisodicLog, Memory, default_run_path, new_run_id
from .planner import Planner
from .repair import Repairer
from .router import ToolRouter
from .safety.guard import SafetyGuard
from .snapshot import Restorer, Snapshotter
from .tools.file_ops import FileDeleteTool, FileReadTool, FileWriteTool
from .tools.process import ProcessKillTool, ProcessListTool
from .tools.shell import ShellTool
from .tools.system import SystemInfoTool
from .verifier import (
    HardVerifier,
    ObservationVerifier,
    SemanticVerifier,
    Verifier,
)


# ============================================================================
# Wiring
# ============================================================================


def build_llm_router(config: MyBotConfig) -> LLMRouter:
    """Build the LLM router. When local is disabled (default), every role
    routes to cloud so input_repair / fix_input still work — costs more
    per task but keeps the system functional without Ollama."""
    cloud = OpenAICompatBackend(OpenAICompatConfig(
        base_url=config.cloud_base_url,
        model=config.cloud_model,
        api_key=config.cloud_api_key,
        timeout_s=config.cloud_timeout_s,
        name="cloud",
    ))
    if not config.local_configured():
        # All roles go to cloud. Caller skips SemanticVerifier separately.
        router = LLMRouter(RouterConfig.all_one("cloud"))
        router.register(cloud)
        return router

    local = OpenAICompatBackend(OpenAICompatConfig(
        base_url=config.local_base_url,
        model=config.local_model,
        api_key=None,  # Ollama OpenAI-compat endpoint accepts unauth'd
        timeout_s=config.local_timeout_s,
        name="local",
    ))
    router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    router.register(cloud)
    router.register(local)
    return router


def build_tool_router() -> ToolRouter:
    """Register every native tool. MCP adapters are added by the user code
    that wires a real MCP server (see plan-mcp-integration)."""
    r = ToolRouter()
    r.register(SystemInfoTool())
    r.register(ProcessListTool())
    r.register(ProcessKillTool())
    r.register(FileReadTool())
    r.register(FileWriteTool())
    r.register(FileDeleteTool())
    r.register(ShellTool())
    return r


def build_agent(
    config: MyBotConfig,
    *,
    run_id: str,
    dry_run: bool = False,
    auto_approve: bool = False,
    llm_router: LLMRouter | None = None,
    tool_router: ToolRouter | None = None,
    confirmation_provider: ConfirmationProvider | None = None,
) -> Agent:
    """Wire every subsystem and return an Agent ready to .run().

    `llm_router` / `tool_router` / `confirmation_provider` injectable for tests.
    """
    llm = llm_router or build_llm_router(config)
    tools = tool_router or build_tool_router()
    guard = SafetyGuard()
    executor = Executor(tools, guard, dry_run=dry_run)

    # Semantic verifier is ALWAYS constructed; the LLMRouter handles
    # whether `LLMRole.VERIFIER` lands on local (when Ollama configured)
    # or cloud (when not). Skipping the layer entirely would mean ambiguous
    # steps fail verification with no semantic check at all — losing real
    # verification capability for a cost saving the user didn't ask for.
    verifier = Verifier(
        hard=HardVerifier(),
        observation=ObservationVerifier(executor),
        semantic=SemanticVerifier(llm),
    )

    episodic = EpisodicLog(default_run_path(run_id))
    if config.runs_dir.parent != default_run_path(run_id).parent.parent:
        # Honor custom data root from config — rebuild path.
        episodic = EpisodicLog(config.runs_dir / f"{run_id}.jsonl")

    if confirmation_provider is None:
        confirmation_provider = (
            AutoApproveProvider() if auto_approve else CLIConfirmationProvider()
        )

    return Agent(
        run_id=run_id,
        planner=Planner(llm),
        executor=executor,
        verifier=verifier,
        diagnoser=Diagnoser(llm),
        repairer=Repairer(llm, max_attempts_per_step=config.max_attempts_per_step),
        episodic=episodic,
        snapshotter=Snapshotter(config.snapshots_dir),
        confirmation_provider=confirmation_provider,
        memory=Memory(),
        budget=TaskBudget(),
        reports_dir=config.failures_dir,
    )


# ============================================================================
# Modes
# ============================================================================


def run_task(
    config: MyBotConfig,
    task: str,
    *,
    goal_class: str | None = None,
    dry_run: bool = False,
    auto_approve: bool = False,
    out: Callable[[str], None] = print,
    builder: Callable[..., Agent] | None = None,
) -> int:
    """Run one task end-to-end. Returns 0 on done, 1 on escalation."""
    if not config.cloud_configured():
        out(
            "ERROR: cloud LLM not configured. Set SENTRICOON_CLOUD_API_KEY "
            "(or OPENAI_API_KEY / ANTHROPIC_API_KEY) before running."
        )
        return 2

    run_id = new_run_id()
    out(f"[sentricoon] run_id={run_id}  task={task!r}  dry_run={dry_run}")

    build = builder or build_agent
    agent = build(config, run_id=run_id, dry_run=dry_run, auto_approve=auto_approve)

    outcome = agent.run(task, goal_class=goal_class)
    _print_outcome(outcome, out)
    return 0 if outcome.status == "done" else 1


def run_rollback(
    snapshot_path: Path,
    *,
    out: Callable[[str], None] = print,
) -> int:
    """Apply a previously-captured snapshot to restore prior state."""
    if not snapshot_path.exists():
        out(f"ERROR: snapshot not found: {snapshot_path}")
        return 2
    restorer = Restorer()
    try:
        snap = restorer.load(snapshot_path)
    except (OSError, json.JSONDecodeError) as e:
        out(f"ERROR: could not load snapshot {snapshot_path}: {e}")
        return 2

    out(f"[sentricoon] rolling back snapshot {snap.snapshot_id} (run={snap.run_id}, "
        f"{len(snap.entries)} entries)")
    results = restorer.restore(snap)
    failed = 0
    for r in results:
        marker = "OK" if r.success else "FAIL"
        out(f"  [{marker}] {r.kind} {r.target!r} action={r.action} — {r.reason or ''}")
        if not r.success and r.action != "noop":
            failed += 1
    out(f"[sentricoon] rollback complete: {len(results) - failed}/{len(results)} succeeded")
    return 0 if failed == 0 else 1


def _print_outcome(outcome: TaskOutcome, out: Callable[[str], None]) -> None:
    out(f"[sentricoon] status={outcome.status}  steps={len(outcome.state.history)}  "
        f"strategies={len(outcome.state.strategies_tried)}")

    for warning in outcome.state.plan_warnings:
        out(f"[sentricoon] WARNING (plan quality): {warning}")

    if outcome.state.history:
        out("")
        out("Steps:")
        for i, record in enumerate(outcome.state.history, start=1):
            out(f"  [{i}] {_format_step_record(record)}")

    if outcome.status == "escalated" and outcome.failure_report is not None:
        report = outcome.failure_report
        out("")
        out(f"[sentricoon] escalation reason: {report.reason.value}")
        out(f"[sentricoon] summary: {report.summary}")
        if outcome.report_paths:
            md, js = outcome.report_paths
            out(f"[sentricoon] report written: {md}")
            out(f"[sentricoon] report json:    {js}")


def _format_step_record(record) -> str:
    marker = "OK" if record.success else "FAIL"
    args_brief = _format_args_brief(record.step.action, record.step.args)
    head = f"[{marker}] {record.step.action}" + (f" {args_brief}" if args_brief else "")
    body = _format_output(record.output)
    return f"{head} -> {body}" if body else head


def _format_args_brief(action: str, args: dict) -> str:
    if action == "shell.run":
        argv = args.get("argv", [])
        return " ".join(str(a) for a in argv) if isinstance(argv, list) else str(argv)
    if action in {"file.read", "file.write", "file.delete"}:
        return str(args.get("path", ""))
    if action == "process.kill":
        return f"pid={args.get('pid')}"
    if action in {"process.list", "system.info"}:
        return ""
    return ""


_OUTPUT_LINE_LIMIT = 140


def _format_output(output) -> str:
    """Render a step's output as a single short line for terminal display."""
    if output is None:
        return ""
    if isinstance(output, dict):
        # Shell results: surface as much of stdout as fits. _truncate
        # collapses newlines to spaces and caps length — taking only
        # splitlines()[0] would lose the most useful line for outputs like
        # `free -h` where the data is on line 2, not the header.
        if "stdout" in output:
            stdout = (output.get("stdout") or "").strip()
            if stdout:
                return _truncate(stdout)
            stderr = (output.get("stderr") or "").strip()
            if stderr:
                return _truncate(f"(stderr) {stderr}")
            return "(no output)"
        # Other dicts: show first 2-3 key=value pairs
        pairs = []
        for k, v in list(output.items())[:3]:
            pairs.append(f"{k}={v!s}")
        return _truncate(", ".join(pairs))
    if isinstance(output, list):
        if not output:
            return "(empty list)"
        return _truncate(f"[{len(output)} entries] " + repr(output[0]))
    return _truncate(str(output))


def _truncate(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= _OUTPUT_LINE_LIMIT:
        return s
    return s[: _OUTPUT_LINE_LIMIT - 3] + "..."


# ============================================================================
# CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentricoon",
        description="OS-Technician AI agent — semi-autonomous Windows/Linux PC repair",
    )
    p.add_argument("task", nargs="?", help="task description, e.g. 'fix my speakers'")
    p.add_argument("--goal-class", help="optional goal class for procedural memory keying")
    p.add_argument("--dry-run", action="store_true",
                   help="plan + simulate write-effect tools without executing them")
    p.add_argument("--yes", "--auto-approve", dest="auto_approve", action="store_true",
                   help="auto-approve high-risk actions (skips confirmation prompts)")
    p.add_argument("--rollback", type=Path, metavar="SNAPSHOT_PATH",
                   help="restore a previously-captured snapshot and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.rollback is not None:
        return run_rollback(args.rollback)

    if not args.task:
        _build_parser().print_help()
        return 0

    config = MyBotConfig.from_env()
    return run_task(
        config,
        args.task,
        goal_class=args.goal_class,
        dry_run=args.dry_run,
        auto_approve=args.auto_approve,
    )


if __name__ == "__main__":
    sys.exit(main())
