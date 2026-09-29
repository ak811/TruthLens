import json
from pathlib import Path

import pytest

from truthlens import pipeline
from truthlens.cli import main
from truthlens.io_utils import read_json, read_jsonl
from truthlens.lvlm.mock import MockLVLM
from truthlens.seeding import stable_seed


def test_cli_end_to_end_with_mocks(tmp_path, dataset):
    out = tmp_path / "out"
    args = ["--manifest", str(dataset), "--output-dir", str(out), "--set", "probe.backend=mock",
            "--set", "probe.device=cpu", "--set", "judge.backend=mock"]
    assert main(["run", *args]) == 0
    res = read_json(out / "metrics.json")["results"]
    assert res["all"]["accuracy"] == 1.0 and res["all"]["n"] == 12 and res["all"]["n_invalid"] == 0
    assert len(read_jsonl(out / "probes.jsonl")) == 12 * 9
    assert (out / "config.resolved.yaml").exists() and "probe" in read_json(out / "run_info.json")["stages"]
    assert main(["yesno", *args]) == 0
    assert read_json(out / "yesno_metrics.json")["results"]["ldm"]["accuracy"] == 1.0


def test_probe_is_resumable(mock_cfg):
    path = pipeline.run_probe(mock_cfg)
    n = len(read_jsonl(path))
    pipeline.run_probe(mock_cfg)
    assert len(read_jsonl(path)) == n


class FlakyLVLM(MockLVLM):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def generate(self, image, prompt):
        if prompt == self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return super().generate(image, prompt)


def test_failed_probes_are_recorded_then_retried(mock_cfg):
    from truthlens.prompts import PROMPTS
    pipeline.run_probe(mock_cfg, backend=FlakyLVLM(PROMPTS["eyes"]))
    errs = [r for r in read_jsonl(Path(mock_cfg["output_dir"]) / "probes.jsonl") if r["error"]]
    assert len(errs) == 12 and all(r["category"] == "eyes" for r in errs)
    pipeline.run_aggregate(mock_cfg)
    s = read_jsonl(Path(mock_cfg["output_dir"]) / "summaries.jsonl")
    assert all(r["missing_categories"] == ["eyes"] for r in s)
    assert all("CUDA" not in r["summary"] for r in s)  # errors never reach the LLM
    pipeline.run_probe(mock_cfg)  # retry
    pipeline.run_aggregate(mock_cfg)
    assert all(r["missing_categories"] == [] for r in read_jsonl(Path(mock_cfg["output_dir"]) / "summaries.jsonl"))


def test_aggregate_on_missing_error(mock_cfg):
    from truthlens.prompts import PROMPTS
    pipeline.run_probe(mock_cfg, backend=FlakyLVLM(PROMPTS["eyes"]))
    mock_cfg["aggregate"]["on_missing"] = "error"
    with pytest.raises(ValueError):
        pipeline.run_aggregate(mock_cfg)


class BrokenJudge:
    name, model = "broken", "none"

    def complete(self, messages):
        raise TimeoutError("API timeout")


def test_judge_failures_are_invalid_not_real(mock_cfg):
    pipeline.run_probe(mock_cfg)
    pipeline.run_aggregate(mock_cfg)
    pipeline.run_judge(mock_cfg, judge=BrokenJudge())
    report = pipeline.run_evaluate(mock_cfg)
    assert report["results"]["all"]["n_invalid"] == 12
    assert report["results"]["all"]["accuracy"] == 0.0  # invalid_policy=incorrect
    pipeline.run_judge(mock_cfg)  # resume with a working judge retries every failed item
    assert pipeline.run_evaluate(mock_cfg)["results"]["all"]["accuracy"] == 1.0


@pytest.mark.parametrize("text,answer", [("Yes.", "yes"), ("no, it looks real", "no"), ("**Yes**", "yes"),
                                         ("I think the answer is no.", "no"), ("Yes and no.", None),
                                         ("Unclear", None), (None, None)])
def test_parse_yes_no(text, answer):
    assert pipeline.parse_yes_no(text) == answer


def test_stable_seed():
    assert stable_seed(0, "a", "eyes") == stable_seed(0, "a", "eyes")
    assert stable_seed(0, "a", "eyes") != stable_seed(0, "a", "texture")
    assert stable_seed(0, "a", "eyes") != stable_seed(1, "a", "eyes")


def test_import_legacy_files(tmp_path, mock_cfg):
    legacy = tmp_path / "eyes_chat_univi_results_fake1000.json"
    legacy.write_text(json.dumps({f"/home/author/data/fake1000/{i:03d}.png": f"answer {i}" for i in range(4)}
                                 | {"/home/author/data/fake1000/999.png": "orphan",
                                    "/home/author/data/fake1000/001.png": "Error: CUDA OOM"}))
    n = pipeline.import_legacy(mock_cfg, str(legacy), "eyes", "ldm")
    assert n == 4
    recs = read_jsonl(Path(mock_cfg["output_dir"]) / "probes.jsonl")
    assert {r["id"] for r in recs} == {f"ldm/{i:03d}.png" for i in range(4)}
    assert [r for r in recs if r["id"] == "ldm/001.png"][0]["error"] == "Error: CUDA OOM"
