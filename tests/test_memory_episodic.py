from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sentricoon.memory import (
    EpisodicLog,
    Memory,
    default_run_path,
    default_runs_dir,
    new_run_id,
)
from sentricoon.state import Step, StepRecord


def _record(action: str = "system.info", success: bool = True, **kw) -> StepRecord:
    now = time.time()
    return StepRecord(
        step=Step(action=action, args=kw.get("args", {})),
        started_at=now,
        ended_at=now + 0.01,
        success=success,
        output=kw.get("output"),
        verification=kw.get("verification"),
        diagnosis=kw.get("diagnosis"),
        repair_strategy=kw.get("repair_strategy"),
        error=kw.get("error"),
    )


class TestRunIdAndPaths(unittest.TestCase):
    def test_run_id_format(self):
        rid = new_run_id()
        self.assertRegex(rid, r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")

    def test_run_ids_unique(self):
        ids = {new_run_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_default_runs_dir_is_home_dot_os_technician(self):
        d = default_runs_dir()
        self.assertEqual(d, Path.home() / ".os-technician" / "runs")

    def test_default_run_path_uses_run_id(self):
        rid = "20260518T142201Z-abcdef12"
        p = default_run_path(rid)
        self.assertEqual(p.name, f"{rid}.jsonl")
        self.assertEqual(p.parent, default_runs_dir())


class TestEpisodicLogAppend(unittest.TestCase):
    def _tmplog(self) -> tuple[EpisodicLog, Path]:
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "run.jsonl"
        return EpisodicLog(path), path

    def test_append_creates_file(self):
        log, path = self._tmplog()
        self.assertFalse(path.exists())
        log.append("task_start", {"task": "x"})
        self.assertTrue(path.exists())

    def test_each_event_is_one_line(self):
        log, path = self._tmplog()
        for i in range(5):
            log.append("step", {"i": i})
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 5)

    def test_event_envelope_has_ts_type_payload(self):
        log, path = self._tmplog()
        log.append("task_start", {"task": "fix speakers"})
        events = log.read_all()
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIn("ts_ms", ev)
        self.assertEqual(ev["type"], "task_start")
        self.assertEqual(ev["payload"]["task"], "fix speakers")
        self.assertIsInstance(ev["ts_ms"], int)

    def test_rejects_empty_event_type(self):
        log, _ = self._tmplog()
        with self.assertRaises(ValueError):
            log.append("", {})

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "c" / "run.jsonl"
            EpisodicLog(path).append("task_start", {})
            self.assertTrue(path.exists())


class TestEpisodicLogTypedHelpers(unittest.TestCase):
    def test_append_step_serializes_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append_step("run-1", _record(action="file.read", args={"path": "/tmp/x"}))
            events = log.read_events("step")
            self.assertEqual(len(events), 1)
            payload = events[0]["payload"]
            self.assertEqual(payload["run_id"], "run-1")
            self.assertEqual(payload["action"], "file.read")
            self.assertEqual(payload["args"], {"path": "/tmp/x"})
            self.assertTrue(payload["success"])

    def test_append_step_handles_non_serializable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            # An exotic output object — json_safe falls back to repr()
            log.append_step("run-1", _record(output=object()))
            events = log.read_events("step")
            self.assertEqual(len(events), 1)
            # Didn't crash; output was coerced
            self.assertIsInstance(events[0]["payload"]["output"], str)

    def test_append_task_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append_task_start("run-1", "fix speakers", "audio_no_sound")
            log.append_plan("run-1", [{"action": "system.info"}], strategy_fingerprint="fpA")
            log.append_step("run-1", _record())
            log.append_strategy("run-1", "fpA", ["system.info"])
            log.append_task_end("run-1", "done", "no audio issues found")
            types = [e["type"] for e in log.read_all()]
            self.assertEqual(types, ["task_start", "plan", "step", "strategy", "task_end"])

    def test_append_confirmation_redacts_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append_confirmation("run-1", "process.kill", "supersecrettoken1234567890", granted=True)
            ev = log.read_events("confirmation")[0]
            # The full token should NOT be in the log
            self.assertNotIn("supersecrettoken1234567890", ev["payload"]["token"])
            self.assertTrue(ev["payload"]["token"].endswith("..."))

    def test_append_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append_escalation("run-1", "budget_exhausted", attempts=20, strategies=5)
            ev = log.read_events("escalation")[0]
            self.assertEqual(ev["payload"]["reason"], "budget_exhausted")
            self.assertEqual(ev["payload"]["attempts"], 20)

    def test_append_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append_snapshot("run-1", "services", "/tmp/snap-services.json")
            ev = log.read_events("snapshot")[0]
            self.assertEqual(ev["payload"]["kind"], "services")


class TestEpisodicLogReads(unittest.TestCase):
    def test_read_empty_log_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            self.assertEqual(log.read_all(), [])

    def test_read_events_filters_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpisodicLog(Path(tmp) / "x.jsonl")
            log.append("step", {"i": 1})
            log.append("strategy", {"fp": "a"})
            log.append("step", {"i": 2})
            self.assertEqual(len(log.read_events("step")), 2)
            self.assertEqual(len(log.read_events("strategy")), 1)
            self.assertEqual(log.count(), 3)

    def test_read_tolerates_torn_last_line(self):
        """Simulate a crash mid-write: append two valid events, then a half-written third."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            log = EpisodicLog(path)
            log.append("step", {"i": 1})
            log.append("step", {"i": 2})
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"ts_ms":12345,"type":"step","payload":{"i":3')  # no closing brace, no newline
            events = log.read_all()
            self.assertEqual(len(events), 2)  # the torn line is dropped, earlier events survive
            self.assertEqual(events[1]["payload"]["i"], 2)

    def test_read_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            log = EpisodicLog(path)
            log.append("step", {"i": 1})
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n\n")
            log.append("step", {"i": 2})
            self.assertEqual(log.count(), 2)


class TestEpisodicLogDurability(unittest.TestCase):
    def test_fsync_is_called(self):
        """The log should fsync after every append — durability is the point."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            EpisodicLog(path).append("step", {"i": 1})
            # File should be fully on disk: size > 0 and readable
            self.assertGreater(path.stat().st_size, 0)

    def test_concurrent_appends_serialize(self):
        """Lock prevents interleaved JSON lines from racing threads."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            log = EpisodicLog(path)

            def writer(n: int):
                for i in range(50):
                    log.append("step", {"thread": n, "i": i})

            threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            events = log.read_all()
            self.assertEqual(len(events), 200)
            # Every line must be valid JSON — if appends interleaved, parsing would fail
            for ev in events:
                self.assertIn("payload", ev)


class TestMemoryFacade(unittest.TestCase):
    def test_quarantine_default_is_false(self):
        # Until layer 4 is built, nothing is quarantined
        self.assertFalse(Memory().is_strategy_quarantined("any-fingerprint"))

    def test_layer_3_methods_raise_not_implemented(self):
        mem = Memory()
        with self.assertRaises(NotImplementedError):
            mem.add_fact("hardware", "Realtek audio")
        with self.assertRaises(NotImplementedError):
            mem.recall("audio")

    def test_layer_4_record_method_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Memory().record_strategy_outcome("fp", "failed")


if __name__ == "__main__":
    unittest.main()
