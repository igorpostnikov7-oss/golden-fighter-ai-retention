"""Artifact bookkeeping for the Flow: fingerprints, and "was this file written by THIS run?".

A file left over from an earlier run must never count as a new output. Each stage snapshots its output
files before it starts; afterwards a file only counts if it exists, is not empty and its (mtime, size)
changed or it did not exist before. A deterministic rewrite has the same content but a new mtime, so it
counts; a file nobody touched does not. The run summary records the answer for every file.
"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from golden_fighter import config

Snapshot = dict[Path, tuple[int, int] | None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> Snapshot:
    return {p: (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths}


def not_rewritten(before: Snapshot) -> list[str]:
    """Files that are missing, empty, or untouched since `before` was taken."""
    after = snapshot(list(before))
    return [p.name for p, now in after.items() if now is None or now[1] == 0 or now == before[p]]


def display_path(path: Path) -> str:
    """Path relative to the project root when possible (summaries stay portable), else absolute."""
    try:
        return path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def describe(paths: list[Path], fresh: set[str]) -> list[dict]:
    """One entry per expected file, saying whether this run produced it."""
    entries = []
    for p in paths:
        entry = {"file": p.name, "path": display_path(p), "exists": p.exists(),
                 "produced_by_this_run": p.name in fresh}
        if p.exists():
            stat = p.stat()
            entry |= {"bytes": stat.st_size, "sha256": sha256(p),
                      "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")}
        entries.append(entry)
    return entries
