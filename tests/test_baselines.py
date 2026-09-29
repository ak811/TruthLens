"""Checks on the vendored baselines.

Pure-Python pieces are extracted with ``ast`` so they run without PyTorch;
the end-to-end runs need ``torch``/``torchvision``/``scikit-learn`` and are
skipped otherwise.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CNN = ROOT / "baselines" / "CNNDetection"
DIRE = ROOT / "baselines" / "DIRE"


def load_function(path, name, **globs):
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    ns = dict(globs)
    exec(compile(module, str(path), "exec"), ns)
    return ns[name]


def test_dire_str2bool_accepts_on_and_t():
    import argparse
    f = load_function(DIRE / "utils" / "utils.py", "str2bool", argparse=argparse)
    assert f("on") and f("t") and f("True") and not f("off")
    with pytest.raises(argparse.ArgumentTypeError):
        f("ont")


def test_lfs_pointer_detection():
    f = load_function(CNN / "inference.py", "is_lfs_pointer")
    assert f(CNN / "weights" / "blur_jpg_prob0.5.pth")
    assert not f(CNN / "inference.py")


def test_numpy2_compatible_sources():
    for p in [CNN / "earlystop.py", DIRE / "utils" / "earlystop.py",
              DIRE / "guided-diffusion" / "guided_diffusion" / "resample.py"]:
        src = p.read_text()
        assert "np.Inf" not in src and "dtype=np.int)" not in src, p


def test_cnndetection_metrics_single_class_does_not_crash():
    skm = pytest.importorskip("sklearn.metrics")
    import numpy as np
    f = load_function(CNN / "inference.py", "compute_metrics", np=np,
                      **{k: getattr(skm, k) for k in ("accuracy_score", "auc", "average_precision_score",
                                                      "confusion_matrix", "precision_recall_curve",
                                                      "precision_recall_fscore_support", "roc_auc_score")})
    m = f([1, 1, 1], [0.9, 0.2, 0.7])
    assert m["tp"] == 2 and m["fn"] == 1 and np.isnan(m["roc_auc"])


def _make_folder(root, real="0_real", fake="1_fake"):
    from PIL import Image
    for sub, color in ((real, (10, 10, 10)), (fake, (240, 240, 240))):
        (root / sub).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (40, 40), color).save(root / sub / f"{i}.png")


@pytest.mark.parametrize("which", ["CNNDetection", "DIRE"])
def test_baseline_inference_end_to_end_cpu(tmp_path, which):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("sklearn")
    data = tmp_path / "data"
    _make_folder(data)
    sys.path.insert(0, str(ROOT / "baselines" / which))
    try:
        from networks.resnet import resnet50
        ckpt = tmp_path / "w.pth"
        torch.save({"model": resnet50(num_classes=1).state_dict()}, ckpt)
    finally:
        sys.path.pop(0)
    extra = ["--input_type", "raw"] if which == "DIRE" else ["-j", "0", "-b", "2"]
    out = tmp_path / "m.json"
    cmd = [sys.executable, "inference.py", "-d", str(data), "-m", str(ckpt), "--use_cpu",
           "--output_json", str(out), *extra]
    r = subprocess.run(cmd, cwd=ROOT / "baselines" / which, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import json
    m = json.loads(out.read_text())["metrics"]
    assert m["n_real"] == 3 and m["n_fake"] == 3


def test_cnndetection_rejects_misnamed_class_folders(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("sklearn")
    data = tmp_path / "data"
    _make_folder(data, real="real", fake="fake")  # alphabetical order would invert labels
    r = subprocess.run([sys.executable, "inference.py", "-d", str(data), "--size_only", "-j", "0"],
                       cwd=CNN, capture_output=True, text=True)
    assert r.returncode != 0 and "0_real" in r.stderr
