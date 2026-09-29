"""Step 3 - textual aggregation ``S = g(A)`` (paper Section 2.3).

Two aggregation modes are provided:

``structured`` (default)
    One labelled block per probe category, in prompt-set order::

        [Eyes and Pupils]
        <answer a_i>

        [Texture and Skin Details]
        <answer a_j>

    This matches the paper's description of a *structured* summary that
    organises the per-prompt answers.

``pipe``
    Byte-for-byte reproduction of the released ``concatinate_jsons.py``: the
    raw answers joined with ``" | "`` and no category labels. Use it (together
    with ``categories: released_7``) to reproduce the released pipeline.

The paper does not specify which of the two produced the reported numbers;
see docs/AUDIT.md.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from .prompts import CATEGORY_TITLES

AGGREGATION_MODES = ("structured", "pipe")
PIPE_SEPARATOR = " | "


def aggregate_answers(answers: Mapping[str, str], categories: Sequence[str],
                      mode: str = "structured") -> Tuple[str, List[str]]:
    """Aggregate per-category answers into a summary string.

    Returns ``(summary, used_categories)``; categories without an answer are skipped.
    """
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"Unknown aggregation mode '{mode}'. Choose from {AGGREGATION_MODES}.")
    used = [c for c in categories if c in answers and answers[c] is not None]
    if mode == "pipe":
        return PIPE_SEPARATOR.join(answers[c] for c in used), used
    blocks = [f"[{CATEGORY_TITLES[c]}]\n{answers[c].strip()}" for c in used]
    return "\n\n".join(blocks), used


def collect_answers(probe_records: Sequence[Dict]) -> Dict[str, Dict[str, str]]:
    """Map ``id -> {category: response}`` using the last successful record per (id, category).

    Records carrying an ``error`` are ignored: the released code wrote exception
    messages ("Error: ...") into the description files, where they were then
    passed to the LLM as if they were image descriptions.
    """
    out: Dict[str, Dict[str, str]] = {}
    for rec in probe_records:
        if rec.get("error") or rec.get("response") is None:
            continue
        out.setdefault(rec["id"], {})[rec["category"]] = rec["response"]
    return out
