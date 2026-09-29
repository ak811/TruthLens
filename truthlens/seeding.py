"""Deterministic seeding.

LVLM decoding in the released code samples with temperature 0.2 and set no
seed, so probe answers were not reproducible. We derive a per-(image, prompt)
seed from a base seed and a stable hash of the key. This makes each answer
independent of processing order, so interrupted runs can be resumed without
changing results (up to GPU-kernel non-determinism, which seeding cannot
remove).
"""
from __future__ import annotations

import logging
import os
import random
import zlib

_WARNED = False


def stable_seed(base_seed: int, *keys: str) -> int:
    """Derive a 31-bit seed from ``base_seed`` and string keys (stable across runs/platforms)."""
    digest = zlib.crc32("\x1f".join(keys).encode("utf-8"))
    return (int(base_seed) * 1_000_003 + digest) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
    except ImportError:
        return
    except OSError as exc:  # broken/partial torch installation
        global _WARNED
        if not _WARNED:
            logging.getLogger("truthlens").warning("torch could not be imported (%s); torch RNG not seeded", exc)
            _WARNED = True
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
