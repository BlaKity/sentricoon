from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from sentricoon.escalation import (
    EscalationReason,
    TaskBudget,
    build_report,
    extract_observations,
    is_repeat_strategy,
    record_strategy,
    register_snapshot,
    should_escalate,
    strategy_fingerprint,
    write_report,
)
from sentricoon.state import Step, StepRecord, TaskState


def _state(task="fix speakers", goal="audio_no_sound") -> TaskState:
    return TaskState(task=task, goal_class=goal)


def _step(action: str, **args) -> Step:
    return Step(action=action, args=args)


def _record(step: Step, success: bool, **kw) -> StepRecord:
    now = time.time()
    return StepRecord(
        step=step,
        started_at=kw.pop("started_at", now),
        ended_at=kw.pop("ended_at", now + 0.01),
        success=success,
        **kw,
    )


class TestStrategyFingerprint(unittest.TestCase):
    def test_fingerprint_stable_across_arg_values(self):
        p1 = [_step("shell.run", command="net stop Audiosrv"), _step("shell.run", command="net start Audiosrv")]
        p2 = [_step("shell.run", command="net stop Builder"), _step("shell.run", command="net start Builder")]
        self.assertEqual(strategy_fingerprint(p1), strategy_fingerprint(p2))

    def test_fingerprint_changes_with_action_sequence(self):
        p1 = [_step("shell.run"), _step("system.info")]
        p2 = [_step("system.info"), _step("shell.run")]
        self.assertNotEqual(strategy_fingerprint(p1), strategy_fingerprint(p2))

    def test_record_strategy_dedupes(self):
        s = _state()
        plan = [_step("a"), _step("b")]
        record_strategy(s, plan)
        record_strategy(s, plan)
        self.assertEqual(len(s.strategies_tried), 1)

    def test_is_repeat_strategy_detects_replay(self):
        s = _state()
        plan = [_step("a"), _step("b")]
        self.assertFalse(is_repeat_strategy(s, plan))
        record_strategy(s, plan)
        self.assertTrue(is_repeat_strategy(s, plan))


class TestShouldEscalate(unittest.TestCase):
    def test_no_trigger_returns_none(self):
        self.assertIsNone(should_escalate(_state(), TaskBudget()))

    def test_abort_requested_wins(self):
        self.assertEqual(
            should_escalate(_state(), TaskBudget(), abort_requested=True),
            EscalationReason.ABORT_REQUESTED,
        )

    def test_budget_exhausted(self):
        s = _state()
        s.history = [_record(_step("noop"), success=False) for _ in range(5)]
        self.assertEqual(
            should_escalate(s, TaskBudget(max_total_attempts=5)),
            EscalationReason.BUDGET_EXHAUSTED,
        )

    def test_time_exhausted(self):
        s = _state()
        s.started_at = time.time() - 10
        self.assertEqual(
            should_escalate(s, TaskBudget(max_wall_time_s=1)),
            EscalationReason.TIME_EXHAUSTED,
        )

    def test_strategy_count_exhausted(self):
        s = _state()
        s.strategies_tried = [f"fp{i}" for i in range(5)]
        self.assertEqual(
            should_escalate(s, TaskBudget(max_distinct_strategies=5)),
            EscalationReason.STRATEGY_EXHAUSTED,
        )

    def test_strategy_loop_detected(self):
        s = _state()
        s.strategies_tried = ["fpA", "fpB", "fpC", "fpA"]
        self.assertEqual(
            should_escalate(s, TaskBudget(max_distinct_strategies=10)),
            EscalationReason.STRATEGY_LOOP,
        )

    def test_quarantine_source_blocks(self):
        s = _state()
        s.plan = [_step("a"), _step("b")]
        fp = strategy_fingerprint(s.plan)
        self.assertEqual(
            should_escalate(s, TaskBudget(), quarantine_source=lambda f: f == fp),
            EscalationReason.STRATEGY_EXHAUSTED,
        )

    def test_priority_abort_beats_budget(self):
        s = _state()
        s.history = [_record(_step("noop"), success=False) for _ in range(99)]
        self.assertEqual(
            should_escalate(s, TaskBudget(), abort_requested=True),
            EscalationReason.ABORT_REQUESTED,
        )


class TestObservations(unittest.TestCase):
    def test_extract_observations_pulls_success_reasons(self):
        history = [
            _record(_step("system.info"), success=True, verification={"success": True, "reason": "Realtek audio detected"}),
            _record(_step("shell.run"), success=False, diagnosis={"ruled_out": "service stopped", "evidence": "service is Running"}),
        ]
        obs, ruled = extract_observations(history)
        self.assertIn("[system.info] Realtek audio detected", obs)
        self.assertIn("service is Running", obs)
        self.assertIn("service stopped", ruled)


class TestFailureReport(unittest.TestCase):
    def test_build_report_includes_history_and_observations(self):
        s = _state()
        s.history = [
            _record(_step("system.info"), success=True, verification={"success": True, "reason": "Realtek detected"}),
            _record(_step("shell.run", command="net start Audiosrv"), success=False, diagnosis={"ruled_out": "audio service stopped"}),
        ]
        s.strategies_tried = ["fp1", "fp2"]
        report = build_report(s, EscalationReason.STRATEGY_EXHAUSTED)
        self.assertEqual(report.attempts, 2)
        self.assertEqual(report.reason, EscalationReason.STRATEGY_EXHAUSTED)
        self.assertTrue(any("Realtek" in o for o in report.observations))
        self.assertIn("audio service stopped", report.ruled_out)
        self.assertTrue(report.next_manual_actions)
        self.assertTrue(any("audio checklist" in s.lower() for s in report.next_manual_actions))

    def test_round_trips_json(self):
        s = _state()
        s.history = [_record(_step("system.info"), success=True)]
        report = build_report(s, EscalationReason.BUDGET_EXHAUSTED)
        parsed = json.loads(report.to_json())
        self.assertEqual(parsed["task"], "fix speakers")
        self.assertEqual(parsed["reason"], "budget_exhausted")
        self.assertEqual(len(parsed["history"]), 1)

    def test_markdown_has_sections(self):
        s = _state()
        s.history = [_record(_step("system.info"), success=True, verification={"success": True, "reason": "ok"})]
        register_snapshot(s, "/tmp/snap.json")
        md = build_report(s, EscalationReason.TIME_EXHAUSTED).to_markdown()
        self.assertIn("# Failure report", md)
        self.assertIn("time_exhausted", md)
        self.assertIn("## Suggested manual next steps", md)
        self.assertIn("## State snapshots", md)

    def test_write_report_creates_files(self):
        s = _state()
        s.history = [_record(_step("system.info"), success=False, error="boom")]
        report = build_report(s, EscalationReason.UNRECOVERABLE_ERROR)
        with tempfile.TemporaryDirectory() as tmp:
            md, js = write_report(report, Path(tmp))
            self.assertTrue(md.exists())
            self.assertEqual(md.suffix, ".md")
            self.assertTrue(js.exists())
            self.assertEqual(js.suffix, ".json")
            self.assertIn("audio_no_sound", md.name)
            parsed = json.loads(js.read_text())
            self.assertEqual(parsed["reason"], "unrecoverable_error")


class TestSnapshotManifest(unittest.TestCase):
    def test_register_snapshot_records_paths(self):
        s = _state()
        register_snapshot(s, "/tmp/audio-pre.json")
        register_snapshot(s, Path("/tmp/services-pre.json"))
        self.assertEqual(s.snapshots, ["/tmp/audio-pre.json", "/tmp/services-pre.json"])


class TestMarkdownAttemptLog(unittest.TestCase):
    def test_markdown_includes_attempt_table_when_history_present(self):
        s = _state()
        s.history = [
            _record(_step("shell.run", argv=["find", "/tmp", "-size", "+50M"]), success=False,
                    verification={"success": False, "reason": "permission denied (4 dirs)",
                                  "needs_retry": False, "layer": "hard"}),
            _record(_step("shell.run", argv=["du", "-sh", "/tmp"]), success=True,
                    verification={"success": True, "reason": "exit code 0",
                                  "needs_retry": False, "layer": "hard"}),
        ]
        md = build_report(s, EscalationReason.ABORT_REQUESTED).to_markdown()
        self.assertIn("## Attempt log", md)
        self.assertIn("| # | Action | Args | Result |", md)
        # action names appear
        self.assertIn("shell.run", md)
        # both args briefs appear
        self.assertIn("find /tmp -size +50M", md)
        self.assertIn("du -sh /tmp", md)
        # results appear
        self.assertIn("FAIL: permission denied", md)
        self.assertIn("OK (exit code 0)", md)

    def test_attempt_log_omitted_when_no_history(self):
        s = _state()
        # no history
        md = build_report(s, EscalationReason.UNRECOVERABLE_ERROR).to_markdown()
        self.assertNotIn("## Attempt log", md)

    def test_pipe_in_args_escaped(self):
        """Pipes would break the markdown table; verify escaping."""
        s = _state()
        s.history = [
            _record(_step("shell.run", argv=["ls", "|", "grep", "x"]), success=True,
                    verification={"success": True, "reason": "exit code 0",
                                  "needs_retry": False, "layer": "hard"}),
        ]
        md = build_report(s, EscalationReason.ABORT_REQUESTED).to_markdown()
        self.assertIn(r"ls \| grep x", md)

    def test_long_args_truncated(self):
        s = _state()
        long_path = "/" + "x" * 200
        s.history = [
            _record(_step("file.read", path=long_path), success=False,
                    error="ENOENT"),
        ]
        md = build_report(s, EscalationReason.ABORT_REQUESTED).to_markdown()
        # Table row should not have a 200-char unbroken cell
        for line in md.split("\n"):
            if line.startswith("| 1 |"):
                self.assertLess(len(line), 250)
                self.assertIn("...", line)

    def test_newlines_in_reason_collapsed(self):
        s = _state()
        s.history = [
            _record(_step("shell.run", argv=["x"]), success=False,
                    verification={"success": False,
                                  "reason": "line1\nline2\nline3",
                                  "needs_retry": False, "layer": "hard"}),
        ]
        md = build_report(s, EscalationReason.ABORT_REQUESTED).to_markdown()
        # No literal newlines inside any table row
        for line in md.split("\n"):
            if line.startswith("| 1 |"):
                # The line is a single line of markdown — content was joined
                self.assertIn("line1 line2 line3", line)


class TestGoalClassSpecificSuggestions(unittest.TestCase):
    def test_audio_routing_suggestions(self):
        s = TaskState(task="fix audio routing", goal_class="audio_routing")
        report = build_report(s, EscalationReason.ABORT_REQUESTED)
        joined = " ".join(report.next_manual_actions)
        self.assertIn("Multi-streaming mode", joined)
        self.assertIn("Realtek", joined)

    def test_disk_audit_suggestions(self):
        s = TaskState(task="disk audit", goal_class="disk_audit")
        report = build_report(s, EscalationReason.ABORT_REQUESTED)
        joined = " ".join(report.next_manual_actions)
        # Key disk-audit-specific suggestions are present
        self.assertIn("accessible-file paths to stdout", joined)
        self.assertTrue(any("2>/dev/null" in s for s in report.next_manual_actions))
        self.assertTrue(any("do not use sudo" in s for s in report.next_manual_actions))

    def test_service_audit_suggestions(self):
        s = TaskState(task="service audit", goal_class="service_audit")
        report = build_report(s, EscalationReason.ABORT_REQUESTED)
        joined = " ".join(report.next_manual_actions)
        self.assertIn("systemctl list-units --failed", joined)
        self.assertIn("Get-Service", joined)

    def test_unknown_goal_class_gets_only_generic(self):
        s = TaskState(task="x", goal_class="totally_made_up_class")
        report = build_report(s, EscalationReason.ABORT_REQUESTED)
        joined = " ".join(report.next_manual_actions).lower()
        self.assertNotIn("realtek", joined)
        self.assertNotIn("systemctl list-units", joined)


if __name__ == "__main__":
    unittest.main()
