"""Snapshot / rollback — capture pre-action state, restore on abort.

The agent's executor takes risky actions (file writes, deletes, process
kills). When something goes wrong and escalation kicks in, the user
should be able to undo what was already applied. That requires capturing
*before* state.

This module owns:

  - `Snapshot` + `SnapshotEntry` — typed records of captured state
  - `Snapshotter`               — writes pre-state to disk
  - `Restorer`                  — applies a snapshot in REVERSE order

Cross-platform:
  - File snapshots: cross-platform (text-only for the initial version).
  - Process snapshots: recorded for forensics but NOT restorable
    (you can't unkill a process). RestoreResult flags this explicitly.
  - Service snapshots: not yet implemented; needs systemctl on Linux,
    `sc query` on Windows. Marked as a follow-up.

Integration:
  - The path returned by `Snapshotter.write()` is registered via
    `escalation.register_snapshot(state, path)` (already in escalation.py).
  - That hook means snapshot paths appear in the FailureReport for the
    user to find when reading the run.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_KINDS = frozenset({"file", "process"})


@dataclass
class SnapshotEntry:
    """One captured pre-state record."""

    kind: str            # "file" | "process"
    target: str          # path, pid, or other identifier
    captured_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set if capture itself failed (best-effort)


@dataclass
class Snapshot:
    """A collection of pre-state captures, persistable as one JSON file."""

    snapshot_id: str
    run_id: str
    created_at: float
    entries: list[SnapshotEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Snapshot":
        return cls(
            snapshot_id=raw["snapshot_id"],
            run_id=raw["run_id"],
            created_at=raw["created_at"],
            entries=[SnapshotEntry(**e) for e in raw.get("entries", [])],
        )


@dataclass
class RestoreResult:
    """Per-entry outcome from Restorer.restore()."""

    kind: str
    target: str
    success: bool
    action: str          # "wrote" | "deleted" | "skipped" | "noop"
    reason: str | None = None


def new_snapshot_id() -> str:
    """Timestamp + 8-hex suffix — same shape as run IDs but a separate namespace."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-snap-" + uuid.uuid4().hex[:8]


def default_snapshots_dir() -> Path:
    return Path.home() / ".os-technician" / "snapshots"


# ============================================================================
# Snapshotter — capture pre-state
# ============================================================================


class Snapshotter:
    """Captures pre-state and persists snapshots to disk.

    Capture methods are best-effort: they never raise. A capture failure
    becomes a SnapshotEntry with `error=` set, so the restore path can
    see what wasn't successfully captured.
    """

    # Cap file snapshots to avoid runaway disk use. Files larger than this
    # at capture time are recorded as "too_large" with no content payload,
    # so restore will refuse to write them back (better than silently
    # truncating).
    MAX_FILE_BYTES_DEFAULT = 5 * 1024 * 1024  # 5 MB

    def __init__(
        self,
        snapshots_dir: Path | str | None = None,
        *,
        max_file_bytes: int = MAX_FILE_BYTES_DEFAULT,
    ) -> None:
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()
        self.max_file_bytes = max_file_bytes

    # ----- per-kind capture -----

    def capture_file(self, path: str) -> SnapshotEntry:
        captured_at = time.time()
        p = Path(path)

        if not p.exists():
            return SnapshotEntry(
                kind="file", target=str(path), captured_at=captured_at,
                payload={"existed": False},
            )

        try:
            size = p.stat().st_size
        except OSError as e:
            return SnapshotEntry(
                kind="file", target=str(path), captured_at=captured_at,
                payload={"existed": True}, error=f"stat failed: {e}",
            )

        if size > self.max_file_bytes:
            return SnapshotEntry(
                kind="file", target=str(path), captured_at=captured_at,
                payload={"existed": True, "too_large": True, "size_bytes": size},
                error=f"file exceeds snapshot cap ({size} > {self.max_file_bytes})",
            )

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return SnapshotEntry(
                kind="file", target=str(path), captured_at=captured_at,
                payload={"existed": True, "size_bytes": size, "binary": True},
                error="non-UTF-8 file content — binary snapshot not supported",
            )
        except OSError as e:
            return SnapshotEntry(
                kind="file", target=str(path), captured_at=captured_at,
                payload={"existed": True}, error=f"read failed: {e}",
            )

        return SnapshotEntry(
            kind="file", target=str(path), captured_at=captured_at,
            payload={
                "existed": True,
                "size_bytes": size,
                "content": content,
                "encoding": "utf-8",
            },
        )

    def capture_process(self, pid: int, *, name_hint: str | None = None) -> SnapshotEntry:
        """Record process info for forensic value — NOT restorable.

        Processes can't be reanimated from a snapshot. We capture so the
        FailureReport / Restorer can show *what* was killed, even though
        no automatic undo is possible.
        """
        captured_at = time.time()
        if not isinstance(pid, int) or pid <= 0:
            return SnapshotEntry(
                kind="process", target=str(pid), captured_at=captured_at,
                payload={}, error="pid must be a positive int",
            )

        payload: dict[str, Any] = {"pid": pid, "name": name_hint, "restorable": False}
        try:
            # On Linux read /proc/<pid>/comm; on Windows we'd need tasklist.
            # Best-effort: if /proc is unavailable, we still record the pid.
            comm = Path(f"/proc/{pid}/comm")
            if comm.exists():
                payload["name"] = comm.read_text(encoding="utf-8").strip()
        except OSError:
            pass

        return SnapshotEntry(
            kind="process", target=str(pid), captured_at=captured_at,
            payload=payload,
        )

    # ----- batched capture -----

    def make_snapshot(self, run_id: str) -> Snapshot:
        return Snapshot(
            snapshot_id=new_snapshot_id(),
            run_id=run_id,
            created_at=time.time(),
        )

    def capture_files(self, run_id: str, paths: list[str]) -> Snapshot:
        snap = self.make_snapshot(run_id)
        snap.entries = [self.capture_file(p) for p in paths]
        return snap

    def capture_processes(
        self,
        run_id: str,
        pids: list[int],
        *,
        names: dict[int, str] | None = None,
    ) -> Snapshot:
        snap = self.make_snapshot(run_id)
        names = names or {}
        snap.entries = [self.capture_process(p, name_hint=names.get(p)) for p in pids]
        return snap

    # ----- persistence -----

    def write(self, snapshot: Snapshot) -> Path:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshots_dir / f"{snapshot.snapshot_id}.json"
        body = json.dumps(snapshot.to_dict(), indent=2)
        # Write atomically: write to a temp file, then os.replace.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
        return path


# ============================================================================
# Restorer — apply a snapshot in reverse order
# ============================================================================


class Restorer:
    """Reads a snapshot and applies it back, entry-by-entry.

    Restoration order is REVERSED — last captured, first restored. This
    matches the "undo a sequence" intuition: if action sequence created
    A then modified B then deleted C, we restore C, then B, then A.
    """

    def __init__(self, snapshotter: Snapshotter | None = None) -> None:
        self.snapshotter = snapshotter or Snapshotter()

    def load(self, path: Path | str) -> Snapshot:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return Snapshot.from_dict(raw)

    def restore(self, snapshot: Snapshot) -> list[RestoreResult]:
        results: list[RestoreResult] = []
        for entry in reversed(snapshot.entries):
            results.append(self._restore_entry(entry))
        return results

    def _restore_entry(self, entry: SnapshotEntry) -> RestoreResult:
        if entry.error:
            return RestoreResult(
                kind=entry.kind, target=entry.target, success=False,
                action="skipped", reason=f"capture had error: {entry.error}",
            )

        if entry.kind == "file":
            return self._restore_file(entry)

        if entry.kind == "process":
            return RestoreResult(
                kind="process", target=entry.target, success=False,
                action="noop",
                reason="processes cannot be restored from a snapshot",
            )

        return RestoreResult(
            kind=entry.kind, target=entry.target, success=False,
            action="skipped", reason=f"unsupported snapshot kind: {entry.kind!r}",
        )

    def _restore_file(self, entry: SnapshotEntry) -> RestoreResult:
        path = Path(entry.target)
        payload = entry.payload

        existed_before = bool(payload.get("existed"))

        if not existed_before:
            # Pre-state was "absent" — if it's present now, delete it.
            if path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    return RestoreResult(
                        kind="file", target=str(path), success=False,
                        action="skipped", reason=f"could not delete: {e}",
                    )
                return RestoreResult(
                    kind="file", target=str(path), success=True,
                    action="deleted", reason="restoring pre-state of 'absent'",
                )
            return RestoreResult(
                kind="file", target=str(path), success=True,
                action="noop", reason="already absent",
            )

        # Pre-state existed — we need its content
        if payload.get("too_large") or payload.get("binary"):
            return RestoreResult(
                kind="file", target=str(path), success=False,
                action="skipped",
                reason="snapshot recorded existence but not content (too large or binary)",
            )

        content = payload.get("content")
        if not isinstance(content, str):
            return RestoreResult(
                kind="file", target=str(path), success=False,
                action="skipped", reason="snapshot is missing content payload",
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=payload.get("encoding", "utf-8"))
        except OSError as e:
            return RestoreResult(
                kind="file", target=str(path), success=False,
                action="skipped", reason=f"write failed: {e}",
            )

        return RestoreResult(
            kind="file", target=str(path), success=True,
            action="wrote", reason=f"restored {len(content)} bytes",
        )
