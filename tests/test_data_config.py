import json
import shutil

import pytest
from PIL import Image

from truthlens.config import DEFAULTS, load_config
from truthlens.data import build_manifest, load_image, load_manifest, select


def test_manifest_ids_are_unique_despite_identical_file_names(dataset):
    samples = load_manifest(str(dataset))
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids)) == 12
    assert {"real/000.png", "ldm/000.png", "progan/000.png"} <= set(ids)
    assert {s.label for s in samples if s.subset == "real"} == {"REAL"}
    assert {s.label for s in samples if s.subset != "real"} == {"FAKE"}


def test_manifest_paths_are_relative_and_portable(tmp_path, dataset):
    rec = json.loads(dataset.read_text().splitlines()[0])
    assert not rec["path"].startswith("/")
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path, moved, ignore=shutil.ignore_patterns("moved"))
    for s in load_manifest(str(moved / "manifest.jsonl")):
        assert s.path.startswith(str(moved)) and Image.open(s.path)


def test_manifest_rejects_bad_input(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        build_manifest([("x", "FAKE", str(tmp_path / "empty"))], str(tmp_path / "m.jsonl"))
    with pytest.raises(FileNotFoundError):
        build_manifest([("x", "FAKE", str(tmp_path / "missing"))], str(tmp_path / "m.jsonl"))
    with pytest.raises(ValueError):
        build_manifest([("x", "MAYBE", str(tmp_path))], str(tmp_path / "m.jsonl"))


def test_select_limit_and_subsets(dataset):
    samples = load_manifest(str(dataset))
    assert len(select(samples, ["ldm"])) == 4
    assert len(select(samples, limit=1)) == 3


def test_load_image_converts_rgba_to_rgb(dataset):
    s = [x for x in load_manifest(str(dataset)) if x.id == "real/000.png"][0]
    assert Image.open(s.path).mode == "RGBA"
    assert load_image(s.path).mode == "RGB"


def test_config_defaults_follow_paper_where_explicit():
    cfg = load_config()
    assert cfg["judge"]["model"] == "gpt-4" and cfg["probe"]["categories"] == "all"
    assert cfg is not DEFAULTS


def test_config_overrides_and_validation(tmp_path):
    cfg = load_config(None, ["probe.generation.do_sample=false", "judge.temperature=0",
                             "probe.options.conv_mode=simple", "aggregate.categories=[eyes, texture]"])
    assert cfg["probe"]["generation"]["do_sample"] is False and cfg["judge"]["temperature"] == 0
    assert cfg["probe"]["options"] == {"conv_mode": "simple"}
    with pytest.raises(KeyError):
        load_config(None, ["judge.modle=gpt-4"])
    with pytest.raises(ValueError):
        load_config(None, ["probe.backend=gpt5v"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("judge:\n  modle: x\n")
    with pytest.raises(KeyError):
        load_config(str(bad))


@pytest.mark.parametrize("name", ["truthlens_chatunivi", "released_pipeline", "table2_blip2",
                                  "table2_llava15", "table2_cogvlm", "smoke_test"])
def test_shipped_configs_are_valid(name):
    from pathlib import Path
    load_config(str(Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"))
