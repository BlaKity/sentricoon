from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sentricoon.escalation import register_snapshot
from sentricoon.snapshot import (
    RestoreResult,
    Restorer,
    Snapshot,
    SnapshotEntry,
    Snapshotter,
    default_snapshots_dir,
    new_snapshot_id,
)
from sentricoon.state import TaskState


# ============================================================================
# IDs and paths
# ============================================================================


class TestSnapshotIds(unittest.TestCase):
    def test_id_format(self):
        sid = new_snapshot_id()
        self.assertRegex(sid, r"^\d{8}T\d{6}Z-snap-[0-9a-f]{8}$")

    def test_ids_unique(self):
        ids = {new_snapshot_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_default_dir(self):
        self.assertEqual(default_snapshots_dir(), Path.home() / ".os-technician" / "snapshots")


# ============================================================================
# File capture
# ============================================================================


class TestCaptureFile(unittest.TestCase):
    def test_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.txt"
            path.write_text("audio device = Realtek")
            entry = Snapshotter().capture_file(str(path))
            self.assertEqual(entry.kind, "file")
            self.assertTrue(entry.payload["existed"])
            self.assertEqual(entry.payload["content"], "audio device = Realtek")
            self.assertEqual(entry.payload["size_bytes"], 22)
            self.assertIsNone(entry.error)

    def test_missing_file(self):
        entry = Snapshotter().capture_file("/definitely/not/a/path/xyz.txt")
        self.assertEqual(entry.kind, "file")
        self.assertFalse(entry.payload["existed"])
        self.assertIsNone(entry.error)

    def test_too_large_records_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_text("x" * 1000)
            snap = Snapshotter(max_file_bytes=100)
            entry = snap.capture_file(str(big))
            self.assertTrue(entry.payload["existed"])
            self.assertTrue(entry.payload["too_large"])
            self.assertNotIn("content", entry.payload)
            self.assertIsNotNone(entry.error)

    def test_binary_file_recorded_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "binary.bin"
            bin_path.write_bytes(b"\x80\x81\x82\xff invalid utf-8")
            entry = Snapshotter().capture_file(str(bin_path))
            self.assertTrue(entry.payload["existed"])
            self.assertTrue(entry.payload.get("binary"))
            self.assertIsNotNone(entry.error)
            self.assertNotIn("content", entry.payload)

    def test_capture_never_raises(self):
        # Snapshotter is best-effort; should swallow all errors
        try:
            Snapshotter().capture_file("/proc/1/mem")  # permission-protected on Linux
        except Exception as e:  # pragma: no cover
            self.fail(f"capture_file raised: {e}")


# ============================================================================
# Process capture
# ============================================================================


class TestCaptureProcess(unittest.TestCase):
    def test_invalid_pid_records_error(self):
        e = Snapshotter().capture_process(0)
        self.assertEqual(e.kind, "process")
        self.assertIsNotNone(e.error)

    def test_pid_with_name_hint(self):
        e = Snapshotter().capture_process(99999, name_hint="chrome.exe")
        self.assertEqual(e.payload["pid"], 99999)
        # name_hint is recorded even when /proc lookup fails
        self.assertIn("name", e.payload)
        self.assertFalse(e.payload["restorable"])

    def test_pid_marked_non_restorable(self):
        e = Snapshotter().capture_process(1234)
        self.assertFalse(e.payload["restorable"])


# ============================================================================
# Batched capture + serialization
# ============================================================================


class TestBatchAndPersistence(unittest.TestCase):
    def test_capture_files_batched(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.txt"
            b = Path(tmp) / "b.txt"
            a.write_text("alpha")
            b.write_text("beta")
            snap = Snapshotter().capture_files("run-x", [str(a), str(b), "/missing"])
            self.assertEqual(len(snap.entries), 3)
            self.assertTrue(snap.entries[0].payload["existed"])
            self.assertTrue(snap.entries[1].payload["existed"])
            self.assertFalse(snap.entries[2].payload["existed"])

    def test_write_creates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = Path(tmp) / "snaps"
            snapshotter = Snapshotter(snap_dir)
            snap = snapshotter.capture_files("run-1", [])
            path = snapshotter.write(snap)
            self.assertTrue(path.exists())
            self.assertTrue(path.name.endswith(".json"))
            parsed = json.loads(path.read_text())
            self.assertEqual(parsed["run_id"], "run-1")
            self.assertEqual(parsed["snapshot_id"], snap.snapshot_id)

    def test_write_is_atomic(self):
        """Temp file used during write should be gone after success."""
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = Path(tmp) / "snaps"
            snapshotter = Snapshotter(snap_dir)
            snap = snapshotter.capture_files("run-1", [])
            snapshotter.write(snap)
            # No leftover .tmp files
            self.assertEqual(list(snap_dir.glob("*.tmp")), [])

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x.txt"
            f.write_text("hello")
            snap_dir = Path(tmp) / "snaps"
            snapshotter = Snapshotter(snap_dir)
            snap = snapshotter.capture_files("run-1", [str(f)])
            path = snapshotter.write(snap)
            loaded = Restorer(snapshotter).load(path)
            self.assertEqual(loaded.snapshot_id, snap.snapshot_id)
            self.assertEqual(loaded.entries[0].payload["content"], "hello")


# ============================================================================
# Restorer — file
# ============================================================================


class TestRestoreFile(unittest.TestCase):
    def test_restores_modified_file_to_pre_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.txt"
            path.write_text("good")
            snapshotter = Snapshotter()
            snap = snapshotter.capture_files("run", [str(path)])

            # The agent does its damage
            path.write_text("bad")

            results = Restorer(snapshotter).restore(snap)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].action, "wrote")
            self.assertEqual(path.read_text(), "good")

    def test_restores_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "important.txt"
            path.write_text("matters")
            snap = Snapshotter().capture_files("run", [str(path)])

            path.unlink()
            self.assertFalse(path.exists())

            results = Restorer().restore(snap)
            self.assertTrue(results[0].success)
            self.assertEqual(path.read_text(), "matters")

    def test_restores_absent_state_by_deleting_present_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.txt"
            self.assertFalse(path.exists())

            snap = Snapshotter().capture_files("run", [str(path)])  # captures "absent"

            # The agent creates it
            path.write_text("oops")

            results = Restorer().restore(snap)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].action, "deleted")
            self.assertFalse(path.exists())

    def test_noop_when_already_at_pre_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stable.txt"
            snap = Snapshotter().capture_files("run", [str(path)])  # captures absent
            results = Restorer().restore(snap)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].action, "noop")

    def test_too_large_capture_refuses_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_text("x" * 1000)
            snapshotter = Snapshotter(max_file_bytes=100)
            snap = snapshotter.capture_files("run", [str(big)])
            # Agent modifies
            big.write_text("x" * 200)
            results = Restorer().restore(snap)
            self.assertFalse(results[0].success)
            # The capture error path skips with the original error verbatim
            self.assertIn("exceeds snapshot cap", results[0].reason)
            # And the file was NOT overwritten to the (too-large) pre-state
            self.assertEqual(big.read_text(), "x" * 200)

    def test_capture_error_propagates_to_skipped_restore(self):
        # An entry with error=set should be skipped, not crashed on
        entry = SnapshotEntry(
            kind="file", target="/x", captured_at=0, payload={}, error="couldn't read"
        )
        snap = Snapshot(snapshot_id="s", run_id="r", created_at=0, entries=[entry])
        results = Restorer().restore(snap)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].action, "skipped")
        self.assertIn("capture had error", results[0].reason)


# ============================================================================
# Restorer — process
# ============================================================================


class TestRestoreProcess(unittest.TestCase):
    def test_process_restore_is_noop(self):
        entry = SnapshotEntry(
            kind="process", target="1234", captured_at=0,
            payload={"pid": 1234, "name": "x", "restorable": False},
        )
        snap = Snapshot(snapshot_id="s", run_id="r", created_at=0, entries=[entry])
        results = Restorer().restore(snap)
        self.assertEqual(results[0].action, "noop")
        self.assertFalse(results[0].success)
        self.assertIn("cannot be restored", results[0].reason)


# ============================================================================
# Restore order (LIFO)
# ============================================================================


class TestRestoreOrder(unittest.TestCase):
    def test_restored_in_reverse_capture_order(self):
        """If capture order is A,B,C the restore order must be C,B,A."""
        entries = [
            SnapshotEntry(kind="file", target=f"/x/{name}", captured_at=i,
                          payload={"existed": True, "content": name})
            for i, name in enumerate(("alpha", "beta", "gamma"))
        ]
        snap = Snapshot(snapshot_id="s", run_id="r", created_at=0, entries=entries)
        results = Restorer().restore(snap)
        # Results are returned in the order they were applied (reversed)
        targets = [r.target for r in results]
        self.assertEqual(targets, ["/x/gamma", "/x/beta", "/x/alpha"])


# ============================================================================
# Integration with escalation.register_snapshot
# ============================================================================


class TestEscalationIntegration(unittest.TestCase):
    def test_snapshot_path_registered_on_taskstate(self):
        """Snapshotter.write returns a path that escalation can index."""
        with tempfile.TemporaryDirectory() as tmp:
            snapshotter = Snapshotter(Path(tmp) / "snaps")
            snap = snapshotter.capture_files("run-1", [])
            path = snapshotter.write(snap)

            state = TaskState(task="fix speakers", goal_class="audio_no_sound")
            register_snapshot(state, path)
            self.assertEqual(state.snapshots, [str(path)])

            # FailureReport flow already uses state.snapshots — already tested
            # in test_escalation.py. Here we just confirm the join point works.


if __name__ == "__main__":
    unittest.main()
