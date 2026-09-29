"""Evaluation metrics (paper Section 3.2) implemented with NumPy only.

Conventions
-----------
* FAKE is the positive class (label 1), REAL is negative (label 0), matching
  the baselines' ``1_fake`` convention.
* ``real_accuracy`` / ``fake_accuracy`` are the per-class accuracies reported
  in Table 2 ("Real (%)", "Fake (%)").
* ``auc_hard`` is the ROC-AUC of the binary verdict used as a score. For a
  binary score this equals the balanced accuracy, (TPR + TNR) / 2. The paper's
  Table 1 TruthLens AUCs (95.0 / 97.5) equal exactly the balanced accuracies
  implied by Table 2 ((98+92)/2, (98+97)/2), so this is the quantity that
  reproduces Table 1. It is *not* a threshold-free AUC; see docs/AUDIT.md.
* ``auc_confidence`` (optional, not in the paper) uses the judge's
  self-reported confidence to rank predictions; only computed when every
  valid prediction carries a confidence.
* Invalid predictions (unparseable judge reply, failed API call, missing
  probes) are never silently mapped to a class. With ``invalid_policy=
  "incorrect"`` (default) they count as errors; with ``"exclude"`` they are
  dropped. Their number is always reported.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

INVALID_POLICIES = ("incorrect", "exclude")
_CONF_MARGIN = {"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.1}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                   y_score: Optional[Sequence[float]] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0 else float("nan"))
    real_acc = _safe_div(tn, tn + fp)
    fake_acc = _safe_div(tp, tp + fn)
    out = {
        "n": int(len(y_true)), "n_real": tn + fp, "n_fake": tp + fn,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "balanced_accuracy": (real_acc + fake_acc) / 2.0,
        "precision": precision, "recall": recall, "f1": f1,
        "auc_hard": roc_auc(y_true, y_pred),
    }
    if y_score is not None:
        out["auc"] = roc_auc(y_true, y_score)
    return out


def _confidence_score(verdict: str, confidence: Optional[str]) -> Optional[float]:
    if confidence not in _CONF_MARGIN:
        return None
    m = _CONF_MARGIN[confidence]
    return 0.5 + m if verdict == "FAKE" else 0.5 - m


def evaluate_group(labels: Sequence[str], predictions: Sequence[Optional[Mapping]],
                   invalid_policy: str = "incorrect") -> Dict[str, float]:
    """Metrics for one evaluation set.

    ``predictions[i]`` is ``None`` (missing) or a mapping with ``verdict``
    (REAL/FAKE/None) and optional ``confidence``.
    """
    if invalid_policy not in INVALID_POLICIES:
        raise ValueError(f"invalid_policy must be one of {INVALID_POLICIES}")
    y_true: List[int] = []
    y_pred: List[int] = []
    conf_scores: List[Optional[float]] = []
    n_invalid = 0
    for label, pred in zip(labels, predictions):
        truth = 1 if label == "FAKE" else 0
        verdict = (pred or {}).get("verdict")
        if verdict not in ("REAL", "FAKE"):
            n_invalid += 1
            if invalid_policy == "exclude":
                continue
            y_true.append(truth)
            y_pred.append(1 - truth)  # counted as an error
            conf_scores.append(None)
            continue
        y_true.append(truth)
        y_pred.append(1 if verdict == "FAKE" else 0)
        conf_scores.append(_confidence_score(verdict, (pred or {}).get("confidence")))
    metrics = binary_metrics(y_true, y_pred)
    metrics["n_invalid"] = n_invalid
    metrics["invalid_policy"] = invalid_policy
    valid_conf = [s for s in conf_scores if s is not None]
    if y_true and len(valid_conf) == len(conf_scores):
        metrics["auc_confidence"] = roc_auc(y_true, valid_conf)
    else:
        metrics["auc_confidence"] = None
    return metrics


def evaluate_by_subset(samples: Iterable, predictions: Mapping[str, Mapping],
                       invalid_policy: str = "incorrect") -> Dict[str, Dict]:
    """Table-2 style evaluation: each fake subset is evaluated together with all real samples.

    Returns ``{"<fake_subset>": metrics, ..., "all": metrics}``.
    """
    samples = list(samples)
    real = [s for s in samples if s.label == "REAL"]
    fake_subsets = sorted({s.subset for s in samples if s.label == "FAKE"})
    groups = {name: real + [s for s in samples if s.label == "FAKE" and s.subset == name]
              for name in fake_subsets}
    groups["all"] = samples
    results = {}
    for name, members in groups.items():
        results[name] = evaluate_group([s.label for s in members],
                                       [predictions.get(s.id) for s in members], invalid_policy)
        results[name]["subsets"] = sorted({s.subset for s in members})
    return results


def format_table(results: Mapping[str, Mapping]) -> str:
    """Human-readable percentage table."""
    cols = [("real_accuracy", "Real"), ("fake_accuracy", "Fake"), ("balanced_accuracy", "BalAcc"),
            ("auc_hard", "AUC(hard)"), ("precision", "Prec"), ("recall", "Rec"), ("f1", "F1")]
    header = f"{'eval set':<12}" + "".join(f"{t:>11}" for _, t in cols) + f"{'n':>7}{'invalid':>9}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        cells = "".join(f"{100 * m[k]:>11.2f}" if m[k] is not None and not math.isnan(m[k]) else f"{'n/a':>11}"
                        for k, _ in cols)
        lines.append(f"{name:<12}{cells}{m['n']:>7}{m['n_invalid']:>9}")
    return "\n".join(lines)
