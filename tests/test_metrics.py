import math
import random

import pytest

from truthlens.data import Sample
from truthlens.metrics import binary_metrics, evaluate_by_subset, evaluate_group, roc_auc


def test_confusion_and_rates():
    m = binary_metrics([1, 1, 1, 0, 0], [1, 0, 1, 0, 1])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["real_accuracy"] == 0.5 and m["fake_accuracy"] == pytest.approx(2 / 3)


def test_hard_auc_equals_balanced_accuracy():
    rng = random.Random(0)
    y = [rng.randint(0, 1) for _ in range(200)]
    p = [rng.randint(0, 1) for _ in range(200)]
    m = binary_metrics(y, p)
    assert m["auc_hard"] == pytest.approx(m["balanced_accuracy"])


def test_table1_auc_is_reproduced_from_table2_accuracies():
    """Table 2 (ChatUniVi, Prompts+LLM): Real 98%, LDM 92%, ProGAN 97% (n=1000 each)
    -> hard AUC 95.0 / 97.5, the TruthLens values of Table 1."""
    for fake_acc, table1 in ((0.92, 0.95), (0.97, 0.975)):
        y = [0] * 1000 + [1] * 1000
        p = [0] * 980 + [1] * 20 + [1] * int(fake_acc * 1000) + [0] * (1000 - int(fake_acc * 1000))
        assert binary_metrics(y, p)["auc_hard"] == pytest.approx(table1)


def test_against_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(1)
    y = [rng.randint(0, 1) for _ in range(300)]
    s = [round(rng.random(), 1) for _ in range(300)]  # many ties
    pred = [int(v >= 0.5) for v in s]
    m = binary_metrics(y, pred, s)
    assert m["auc"] == pytest.approx(sk.roc_auc_score(y, s))
    assert m["precision"] == pytest.approx(sk.precision_score(y, pred))
    assert m["recall"] == pytest.approx(sk.recall_score(y, pred))
    assert m["f1"] == pytest.approx(sk.f1_score(y, pred))


def test_single_class_gives_nan_auc_not_crash():
    assert math.isnan(roc_auc([1, 1], [0.2, 0.9]))
    m = binary_metrics([1, 1], [1, 0])
    assert math.isnan(m["real_accuracy"]) and m["fake_accuracy"] == 0.5


def test_invalid_policy():
    labels = ["FAKE", "REAL", "FAKE"]
    preds = [{"verdict": "FAKE"}, {"verdict": None}, None]
    inc = evaluate_group(labels, preds, "incorrect")
    exc = evaluate_group(labels, preds, "exclude")
    assert inc["n_invalid"] == exc["n_invalid"] == 2
    assert inc["n"] == 3 and inc["accuracy"] == pytest.approx(1 / 3)
    assert exc["n"] == 1 and exc["accuracy"] == 1.0


def test_confidence_auc_only_when_all_have_confidence():
    labels = ["FAKE", "REAL"]
    assert evaluate_group(labels, [{"verdict": "FAKE"}, {"verdict": "REAL"}])["auc_confidence"] is None
    m = evaluate_group(labels, [{"verdict": "FAKE", "confidence": "LOW"}, {"verdict": "REAL", "confidence": "HIGH"}])
    assert m["auc_confidence"] == 1.0


def test_per_subset_evaluation_pools_real_images():
    samples = [Sample("real/a", "", "REAL", "real"), Sample("ldm/a", "", "FAKE", "ldm"),
               Sample("progan/a", "", "FAKE", "progan")]
    preds = {"real/a": {"verdict": "REAL"}, "ldm/a": {"verdict": "FAKE"}, "progan/a": {"verdict": "REAL"}}
    res = evaluate_by_subset(samples, preds)
    assert res["ldm"]["n"] == 2 and res["ldm"]["accuracy"] == 1.0
    assert res["progan"]["subsets"] == ["progan", "real"] and res["progan"]["fake_accuracy"] == 0.0
    assert res["all"]["n"] == 3
