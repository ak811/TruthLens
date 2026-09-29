"""Small I/O helpers: JSONL streaming (append + flush for resumability) and atomic JSON writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

PathLike = Union[str, os.PathLike]


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # A truncated last line can occur if a run was killed mid-write; skip it
                # (the record will simply be recomputed on resume).
                raise ValueError(f"{path}:{lineno}: invalid JSON line ({exc}). If this is the last line "
                                 f"of an interrupted run, delete it and resume.") from exc


class JsonlWriter:
    """Append-mode JSONL writer that flushes after every record."""

    def __init__(self, path: PathLike, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open(mode, encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> None:
    with JsonlWriter(path, mode="w") as w:
        for r in records:
            w.write(r)


def write_json(path: PathLike, obj: Any) -> None:
    """Write JSON atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
