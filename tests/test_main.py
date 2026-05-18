"""Tests for the CLI entry point — focuses on plumbing, not agent behavior.

Real LLM calls would require network + API keys, so these tests inject a
MockBackend-backed LLMRouter into `build_agent` via the `llm_router=` and
`confirmation_provider=` hooks.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sentricoon.agent import Agent
from sentricoon.confirmation import AlwaysDenyProvider, AutoApproveProvider
from sentricoon.config import MyBotConfig
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.main import (
    _build_parser,
    build_agent,
    build_tool_router,
    main,
    run_rollback,
    run_task,
)
from sentricoon.snapshot import Snapshotter


# ============================================================================
# MyBotConfig
# ============================================================================


class TestConfig(unittest.TestCase):
    def test_defaults_when_env_empty(self):
        c = MyBotConfig.from_env(env={})
        self.assertIsNone(c.cloud_api_key)
        self.assertEqual(c.cloud_base_url, "https://api.openai.com/v1")
        self.assertEqual(c.cloud_model, "gpt-4o-mini")
        self.assertEqual(c.local_base_url, "http://localhost:11434/v1")
        self.assertEqual(c.local_model, "llama3.2:3b")
        self.assertFalse(c.cloud_configured())
        # Local is OFF by default
        self.assertFalse(c.local_configured())

    def test_local_enabled_by_setting_url(self):
        c = MyBotConfig.from_env(env={"SENTRICOON_LOCAL_BASE_URL": "http://my-ollama:11434/v1"})
        self.assertTrue(c.local_configured())
        self.assertEqual(c.local_base_url, "http://my-ollama:11434/v1")

    def test_local_enabled_by_explicit_flag(self):
        c = MyBotConfig.from_env(env={"SENTRICOON_LOCAL_ENABLED": "1"})
        self.assertTrue(c.local_configured())
        # URL uses default since user didn't override
        self.assertEqual(c.local_base_url, "http://localhost:11434/v1")

    def test_local_flag_variants(self):
        for v in ("1", "true", "yes", "TRUE", "Yes"):
            c = MyBotConfig.from_env(env={"SENTRICOON_LOCAL_ENABLED": v})
            self.assertTrue(c.local_configured(), msg=v)
        for v in ("0", "false", "no", "", "maybe"):
            c = MyBotConfig.from_env(env={"SENTRICOON_LOCAL_ENABLED": v})
            self.assertFalse(c.local_configured(), msg=v)

    def test_mybot_key_takes_precedence_over_openai(self):
        c = MyBotConfig.from_env(env={
            "SENTRICOON_CLOUD_API_KEY": "sentricoon-key",
            "OPENAI_API_KEY": "openai-key",
        })
        self.assertEqual(c.cloud_api_key, "sentricoon-key")

    def test_openai_key_fallback(self):
        c = MyBotConfig.from_env(env={"OPENAI_API_KEY": "openai-key"})
        self.assertEqual(c.cloud_api_key, "openai-key")
        self.assertTrue(c.cloud_configured())

    def test_anthropic_key_fallback(self):
        c = MyBotConfig.from_env(env={"ANTHROPIC_API_KEY": "anthropic-key"})
        self.assertEqual(c.cloud_api_key, "anthropic-key")

    def test_overrides_from_env(self):
        c = MyBotConfig.from_env(env={
            "SENTRICOON_CLOUD_API_KEY": "k",
            "SENTRICOON_CLOUD_BASE_URL": "https://example.com/v1",
            "SENTRICOON_CLOUD_MODEL": "gpt-5",
            "SENTRICOON_LOCAL_BASE_URL": "http://192.168.1.5:11434/v1",
            "SENTRICOON_LOCAL_MODEL": "qwen2.5:7b",
            "SENTRICOON_DATA_ROOT": "/var/lib/sentricoon",
            "SENTRICOON_MAX_RETRIES_PER_STEP": "5",
        })
        self.assertEqual(c.cloud_base_url, "https://example.com/v1")
        self.assertEqual(c.cloud_model, "gpt-5")
        self.assertEqual(c.local_base_url, "http://192.168.1.5:11434/v1")
        self.assertEqual(c.local_model, "qwen2.5:7b")
        self.assertEqual(c.runs_dir, Path("/var/lib/sentricoon/runs"))
        self.assertEqual(c.snapshots_dir, Path("/var/lib/sentricoon/snapshots"))
        self.assertEqual(c.failures_dir, Path("/var/lib/sentricoon/failures"))
        self.assertEqual(c.max_attempts_per_step, 5)


# ============================================================================
# build_tool_router — registers every allowlisted native tool
# ============================================================================


class TestBuildToolRouter(unittest.TestCase):
    def test_registers_all_native_tools(self):
        r = build_tool_router()
        names = set(r.registered_actions())
        # All native actions should be wired
        self.assertIn("system.info", names)
        self.assertIn("process.list", names)
        self.assertIn("process.kill", names)
        self.assertIn("file.read", names)
        self.assertIn("file.write", names)
        self.assertIn("file.delete", names)
        self.assertIn("shell.run", names)


# ============================================================================
# build_agent — pluggable injection points work
# ============================================================================


def _test_config(tmp: Path, *, local_enabled: bool = False) -> MyBotConfig:
    env = {"SENTRICOON_CLOUD_API_KEY": "test-key", "SENTRICOON_DATA_ROOT": str(tmp)}
    if local_enabled:
        env["SENTRICOON_LOCAL_ENABLED"] = "1"
    return MyBotConfig.from_env(env=env)


def _injected_router() -> LLMRouter:
    """Mock-only LLM router for plumbing tests."""
    cloud = MockBackend(script=[], name="cloud", model="cloud-1")
    local = MockBackend(script=[], name="local", model="local-1")
    r = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    r.register(cloud)
    r.register(local)
    return r


class TestBuildAgent(unittest.TestCase):
    def test_builds_with_real_components(self):
        """build_agent must wire every subsystem without exploding."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))
            agent = build_agent(
                cfg, run_id="test-run",
                llm_router=_injected_router(),
                confirmation_provider=AlwaysDenyProvider(),
            )
            self.assertIsInstance(agent, Agent)
            self.assertEqual(agent.run_id, "test-run")
            self.assertIsNotNone(agent.snapshotter)
            # All four major LLM-aware subsystems share the injected router
            self.assertIs(agent.planner.router, agent.diagnoser.router)
            self.assertIs(agent.planner.router, agent.repairer.router)

    def test_dry_run_wires_through_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))
            agent = build_agent(
                cfg, run_id="x", dry_run=True,
                llm_router=_injected_router(),
                confirmation_provider=AlwaysDenyProvider(),
            )
            self.assertTrue(agent.executor.dry_run)

    def test_auto_approve_uses_auto_approve_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))
            agent = build_agent(
                cfg, run_id="x", auto_approve=True,
                llm_router=_injected_router(),
            )
            self.assertIsInstance(agent.confirmation_provider, AutoApproveProvider)

    def test_custom_data_root_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))
            agent = build_agent(
                cfg, run_id="run-xyz",
                llm_router=_injected_router(),
                confirmation_provider=AlwaysDenyProvider(),
            )
            # Episodic log lives under the configured runs dir
            self.assertTrue(str(agent.episodic.path).startswith(str(cfg.runs_dir)))

    def test_semantic_verifier_always_constructed(self):
        """Semantic verifier is built regardless of local_enabled —
        the LLMRouter decides whether VERIFIER role lands on local or cloud."""
        for local_enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _test_config(Path(tmp), local_enabled=local_enabled)
                agent = build_agent(
                    cfg, run_id="x",
                    llm_router=_injected_router(),
                    confirmation_provider=AlwaysDenyProvider(),
                )
                self.assertIsNotNone(agent.verifier.semantic,
                                     msg=f"local_enabled={local_enabled}")

    def test_verifier_role_routes_to_cloud_when_local_disabled(self):
        """The cost-bearing assertion: when Ollama isn't configured,
        semantic verification still happens — on cloud, not nowhere."""
        from sentricoon.llm.role import LLMRole
        from sentricoon.main import build_llm_router
        cfg = MyBotConfig.from_env(env={"SENTRICOON_CLOUD_API_KEY": "k"})
        router = build_llm_router(cfg)
        self.assertEqual(router.backend_for(LLMRole.VERIFIER).name, "cloud")


class TestLLMRouterBuild(unittest.TestCase):
    """build_llm_router behavior changes based on local_configured."""

    def test_no_ollama_all_cloud_router(self):
        from sentricoon.llm.role import LLMRole
        from sentricoon.main import build_llm_router
        cfg = MyBotConfig.from_env(env={"SENTRICOON_CLOUD_API_KEY": "k"})
        router = build_llm_router(cfg)
        self.assertEqual(router.registered_backends(), ["cloud"])
        # Every role routes to cloud
        for role in LLMRole:
            self.assertEqual(router.backend_for(role).name, "cloud")

    def test_with_ollama_hybrid_router(self):
        from sentricoon.llm.role import LLMRole
        from sentricoon.main import build_llm_router
        cfg = MyBotConfig.from_env(env={
            "SENTRICOON_CLOUD_API_KEY": "k",
            "SENTRICOON_LOCAL_ENABLED": "1",
        })
        router = build_llm_router(cfg)
        self.assertEqual(set(router.registered_backends()), {"cloud", "local"})
        self.assertEqual(router.backend_for(LLMRole.PLANNER).name, "cloud")
        self.assertEqual(router.backend_for(LLMRole.VERIFIER).name, "local")
        self.assertEqual(router.backend_for(LLMRole.INPUT_REPAIR).name, "local")


# ============================================================================
# run_task — refuses when cloud key is missing
# ============================================================================


class TestRunTask(unittest.TestCase):
    def test_refuses_without_cloud_key(self):
        out_lines: list[str] = []
        cfg = MyBotConfig.from_env(env={})  # no key
        rc = run_task(cfg, "fix speakers", out=out_lines.append)
        self.assertEqual(rc, 2)
        self.assertTrue(any("cloud LLM not configured" in line for line in out_lines))

    def test_returns_zero_on_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))

            valid_plan = json.dumps({"steps": [{
                "action": "system.info", "args": {},
                "expected_state": "platform present", "risk_level": "low",
            }]})

            def builder(config, *, run_id, dry_run=False, auto_approve=False):
                cloud = MockBackend(script=[valid_plan], name="cloud", model="c")
                local = MockBackend(script=[], name="local", model="l")
                r = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
                r.register(cloud)
                r.register(local)
                return build_agent(
                    config, run_id=run_id, dry_run=dry_run, auto_approve=auto_approve,
                    llm_router=r, confirmation_provider=AlwaysDenyProvider(),
                )

            out_lines: list[str] = []
            rc = run_task(cfg, "inspect", out=out_lines.append, builder=builder)
            self.assertEqual(rc, 0)
            self.assertTrue(any("status=done" in line for line in out_lines))

    def test_returns_one_on_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _test_config(Path(tmp))

            def builder(config, *, run_id, dry_run=False, auto_approve=False):
                cloud = MockBackend(script=["bogus"], name="cloud", model="c")
                local = MockBackend(script=[], name="local", model="l")
                r = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
                r.register(cloud)
                r.register(local)
                return build_agent(
                    config, run_id=run_id,
                    llm_router=r, confirmation_provider=AlwaysDenyProvider(),
                )

            out_lines: list[str] = []
            rc = run_task(cfg, "inspect", out=out_lines.append, builder=builder)
            self.assertEqual(rc, 1)
            self.assertTrue(any("status=escalated" in line for line in out_lines))


# ============================================================================
# run_rollback
# ============================================================================


class TestRunRollback(unittest.TestCase):
    def test_missing_snapshot_returns_two(self):
        out: list[str] = []
        rc = run_rollback(Path("/definitely/missing/snap.json"), out=out.append)
        self.assertEqual(rc, 2)
        self.assertTrue(any("not found" in line for line in out))

    def test_malformed_snapshot_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("not json at all")
            out: list[str] = []
            rc = run_rollback(bad, out=out.append)
            self.assertEqual(rc, 2)

    def test_round_trip(self):
        """Capture a file, modify it, rollback restores."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.txt"
            target.write_text("ORIGINAL")
            snapshotter = Snapshotter(Path(tmp) / "snaps")
            snap = snapshotter.capture_files("run-x", [str(target)])
            snap_path = snapshotter.write(snap)

            # Simulate damage
            target.write_text("DAMAGED")

            out: list[str] = []
            rc = run_rollback(snap_path, out=out.append)
            self.assertEqual(rc, 0)
            self.assertEqual(target.read_text(), "ORIGINAL")


# ============================================================================
# argparse + main() dispatch
# ============================================================================


class TestArgumentParser(unittest.TestCase):
    def test_task_only(self):
        args = _build_parser().parse_args(["fix my speakers"])
        self.assertEqual(args.task, "fix my speakers")
        self.assertFalse(args.dry_run)
        self.assertFalse(args.auto_approve)
        self.assertIsNone(args.rollback)

    def test_dry_run_flag(self):
        args = _build_parser().parse_args(["x", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_yes_and_auto_approve_are_aliases(self):
        for flag in ("--yes", "--auto-approve"):
            args = _build_parser().parse_args(["x", flag])
            self.assertTrue(args.auto_approve, msg=flag)

    def test_goal_class(self):
        args = _build_parser().parse_args(["x", "--goal-class", "audio_no_sound"])
        self.assertEqual(args.goal_class, "audio_no_sound")

    def test_rollback_path(self):
        args = _build_parser().parse_args(["--rollback", "/tmp/snap.json"])
        self.assertEqual(args.rollback, Path("/tmp/snap.json"))
        self.assertIsNone(args.task)


class TestMainDispatch(unittest.TestCase):
    def test_no_task_prints_help(self):
        # When called with no args, prints help and returns 0
        rc = main([])
        self.assertEqual(rc, 0)

    def test_rollback_dispatches_to_rollback(self):
        rc = main(["--rollback", "/definitely/missing"])
        self.assertEqual(rc, 2)


class TestPrintOutcome(unittest.TestCase):
    """The terminal-output formatter: surface steps + warnings."""

    def _build_outcome_with(self, steps_history, plan_warnings=None, status="done"):
        from sentricoon.agent import TaskOutcome
        from sentricoon.state import Step, StepRecord, TaskState
        import time
        state = TaskState(task="x")
        state.plan_warnings = list(plan_warnings or [])
        for action, args, output in steps_history:
            now = time.time()
            state.history.append(StepRecord(
                step=Step(action=action, args=args),
                started_at=now, ended_at=now,
                success=True, output=output,
            ))
        return TaskOutcome(run_id="run-x", status=status, state=state)

    def test_done_prints_step_summary(self):
        from sentricoon.main import _print_outcome
        outcome = self._build_outcome_with([
            ("system.info", {}, {"system": "Linux", "release": "6.8"}),
            ("shell.run", {"argv": ["uname", "-r"]}, {"stdout": "6.8.0-111-generic\n", "stderr": ""}),
        ])
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        joined = "\n".join(lines)
        self.assertIn("status=done", joined)
        self.assertIn("Steps:", joined)
        self.assertIn("system.info", joined)
        self.assertIn("uname -r", joined)  # argv formatted brief
        self.assertIn("6.8.0-111-generic", joined)  # actual output surfaces

    def test_plan_warnings_surface(self):
        from sentricoon.main import _print_outcome
        outcome = self._build_outcome_with(
            [("system.info", {}, {"system": "Linux"})],
            plan_warnings=[
                "step 1 (system.info): expected_state 'OS info is retrieved' — passive",
            ],
        )
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        joined = "\n".join(lines)
        self.assertIn("WARNING (plan quality)", joined)
        self.assertIn("passive", joined)

    def test_long_output_truncated(self):
        from sentricoon.main import _print_outcome
        outcome = self._build_outcome_with([
            ("shell.run", {"argv": ["echo"]},
             {"stdout": "A" * 500, "stderr": ""}),
        ])
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        for line in lines:
            self.assertLess(len(line), 250)  # generous bound

    def test_multiline_output_collapsed_to_single_line(self):
        """Multi-line stdout collapses (newlines → spaces) so the data
        lines (often line 2+) are visible. Without this, `free -h` would
        only show the column-header row — useless."""
        from sentricoon.main import _print_outcome
        outcome = self._build_outcome_with([
            ("shell.run", {"argv": ["free"]},
             {"stdout": "Mem: 23Gi total\nSwap: 0B\n", "stderr": ""}),
        ])
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        joined = "\n".join(lines)
        # BOTH lines surface, on the same output line, separated by spaces
        self.assertIn("Mem: 23Gi total", joined)
        self.assertIn("Swap: 0B", joined)
        for line in lines:
            if "Mem:" in line:
                self.assertIn("Swap", line)

    def test_free_h_header_no_longer_dominates(self):
        """Regression: previously the header-only row was all the user saw
        for `free -h`. Now actual numbers must be visible."""
        from sentricoon.main import _print_outcome
        free_h_output = (
            "               total        used        free      shared  buff/cache   available\n"
            "Mem:            23Gi       3.4Gi        19Gi       111Mi       1.5Gi        20Gi\n"
            "Swap:             0B          0B          0B\n"
        )
        outcome = self._build_outcome_with([
            ("shell.run", {"argv": ["free", "-h"]},
             {"stdout": free_h_output, "stderr": ""}),
        ])
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        joined = "\n".join(lines)
        # The actual memory amount appears somewhere
        self.assertIn("23Gi", joined)

    def test_failed_step_marked_FAIL(self):
        from sentricoon.agent import TaskOutcome
        from sentricoon.state import Step, StepRecord, TaskState
        from sentricoon.main import _print_outcome
        import time
        state = TaskState(task="x")
        now = time.time()
        state.history.append(StepRecord(
            step=Step(action="shell.run", args={"argv": ["false"]}),
            started_at=now, ended_at=now,
            success=False, error="exit 1",
        ))
        outcome = TaskOutcome(run_id="x", status="done", state=state)
        lines: list[str] = []
        _print_outcome(outcome, lines.append)
        self.assertTrue(any("[FAIL]" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
