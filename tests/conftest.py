import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_images(root: Path, n: int = 4):
    """Dark 'real' images and bright 'fake' images with identical file names across subsets."""
    specs = {"real": (20, 20, 20), "ldm": (230, 230, 230), "progan": (220, 200, 210)}
    for subset, color in specs.items():
        d = root / subset
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            mode = "RGBA" if i == 0 else "RGB"
            Image.new(mode, (16, 16), color + ((255,) if mode == "RGBA" else ())).save(d / f"{i:03d}.png")
    return root


@pytest.fixture
def dataset(tmp_path):
    from truthlens.data import build_manifest

    root = make_images(tmp_path / "images")
    manifest = tmp_path / "manifest.jsonl"
    build_manifest([("real", "REAL", str(root / "real")), ("ldm", "FAKE", str(root / "ldm")),
                    ("progan", "FAKE", str(root / "progan"))], str(manifest))
    return manifest


@pytest.fixture
def mock_cfg(tmp_path, dataset):
    from truthlens.config import load_config

    return load_config(None, [f"data.manifest={dataset}", f"output_dir={tmp_path / 'out'}",
                              "probe.backend=mock", "probe.device=cpu", "judge.backend=mock"])
