"""Dataset manifests.

A manifest is a JSONL file with one record per image::

    {"id": "ldm/00012.png", "path": "ldm_fake1000/00012.png", "label": "FAKE", "subset": "ldm"}

* ``id`` is ``<subset>/<path relative to the subset root>`` and is unique, so
  real and fake images with identical file names can never collide (they did
  in the released code, where outputs were keyed by bare file name).
* ``path`` is stored relative to the manifest's directory when possible and is
  resolved against it on load, so a manifest can be moved together with data.
* ``label`` is ``REAL`` or ``FAKE``; ``subset`` names the source (``real``,
  ``ldm``, ``progan``...). Metrics are reported per fake subset against the
  pooled real subset(s), matching Table 2 (one "Real" column shared by LDM and
  ProGAN).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .io_utils import iter_jsonl, write_jsonl

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
LABELS = ("REAL", "FAKE")


@dataclass(frozen=True)
class Sample:
    id: str
    path: str  # absolute, resolved
    label: str
    subset: str

    def to_record(self) -> Dict[str, str]:
        return {"id": self.id, "path": self.path, "label": self.label, "subset": self.subset}


def list_images(root: Path, recursive: bool = False) -> List[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def parse_source(spec: str, default_name: str) -> Tuple[str, str]:
    """Parse ``name=path`` or bare ``path`` (name defaults to ``default_name``)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name or "/" in name:
            raise ValueError(f"Invalid subset name in '{spec}'")
        return name, path
    return default_name, spec


def build_manifest(sources: Sequence[Tuple[str, str, str]], out_path: str, recursive: bool = False,
                   limit_per_subset: int = 0) -> List[Sample]:
    """Build a manifest from ``(subset, label, directory)`` triples and write it to ``out_path``."""
    out_dir = Path(out_path).resolve().parent
    samples: List[Sample] = []
    seen_subsets = set()
    for subset, label, directory in sources:
        label = label.upper()
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label}")
        if subset in seen_subsets:
            raise ValueError(f"Duplicate subset name '{subset}'")
        seen_subsets.add(subset)
        root = Path(directory).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        files = list_images(root, recursive)
        if not files:
            raise ValueError(f"No images with extensions {IMAGE_EXTENSIONS} in {directory}")
        if limit_per_subset:
            files = files[:limit_per_subset]
        for f in files:
            rel_id = f.relative_to(root).as_posix()
            samples.append(Sample(id=f"{subset}/{rel_id}", path=str(f), label=label, subset=subset))
    records = []
    for s in samples:
        try:
            stored = os.path.relpath(s.path, out_dir)
        except ValueError:  # different drive on Windows
            stored = s.path
        records.append({"id": s.id, "path": Path(stored).as_posix(), "label": s.label, "subset": s.subset})
    write_jsonl(out_path, records)
    return samples


def load_manifest(path: str) -> List[Sample]:
    base = Path(path).resolve().parent
    samples: List[Sample] = []
    ids = set()
    for rec in iter_jsonl(path):
        missing = {"id", "path", "label", "subset"} - set(rec)
        if missing:
            raise ValueError(f"Manifest record missing fields {missing}: {rec}")
        label = str(rec["label"]).upper()
        if label not in LABELS:
            raise ValueError(f"Invalid label '{rec['label']}' for {rec['id']}")
        if rec["id"] in ids:
            raise ValueError(f"Duplicate id in manifest: {rec['id']}")
        ids.add(rec["id"])
        p = Path(rec["path"])
        if not p.is_absolute():
            p = (base / p).resolve()
        samples.append(Sample(id=rec["id"], path=str(p), label=label, subset=rec["subset"]))
    if not samples:
        raise ValueError(f"Manifest {path} is empty")
    return samples


def select(samples: Iterable[Sample], subsets: Sequence[str] = (), limit: int = 0) -> List[Sample]:
    """Filter samples by subset names and apply a per-subset ``limit`` (0 = no limit)."""
    out: List[Sample] = []
    counts: Dict[str, int] = {}
    wanted = set(subsets)
    for s in samples:
        if wanted and s.subset not in wanted:
            continue
        if limit and counts.get(s.subset, 0) >= limit:
            continue
        counts[s.subset] = counts.get(s.subset, 0) + 1
        out.append(s)
    return out


def load_image(path: str):
    """Load an image as 3-channel RGB (the released code did not convert, so RGBA/greyscale files
    could reach the vision encoder with the wrong number of channels)."""
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")
