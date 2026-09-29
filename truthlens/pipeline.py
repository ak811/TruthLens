"""TruthLens pipeline stages (paper Section 2, Figure 1).

Each stage reads/writes files in ``output_dir``::

    probes.jsonl         step 1+2: one record per (image, probe category)       [resumable]
    summaries.jsonl      step 3:   one aggregated summary S per image
    verdicts.jsonl       step 4:   one LLM verdict (y, r) per image              [resumable]
    metrics.json         evaluation (Tables 1-3 quantities)
    yesno_*.jsonl/json   "Yes or No Question" baseline of Table 2
    run_info.json        environment + resolved configuration

Resumable stages skip items that already have a successful record, so an
interrupted run can be restarted with the same command.
"""
from __future__ import annotations

import datetime as _dt
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .aggregate import aggregate_answers, collect_answers
from .data import Sample, load_image, load_manifest, select
from .io_utils import JsonlWriter, iter_jsonl, read_json, read_jsonl, write_json, write_jsonl
from .judge import build_messages, make_judge, parse_judge_response
from .metrics import evaluate_by_subset, format_table
from .prompts import CATEGORY_TITLES, PROMPTS, resolve_categories
from .seeding import set_seed, stable_seed

log = logging.getLogger("truthlens")

PROBES_FILE = "probes.jsonl"
SUMMARIES_FILE = "summaries.jsonl"
VERDICTS_FILE = "verdicts.jsonl"
METRICS_FILE = "metrics.json"
YESNO_RESPONSES_FILE = "yesno_responses.jsonl"
YESNO_VERDICTS_FILE = "yesno_verdicts.jsonl"
YESNO_METRICS_FILE = "yesno_metrics.json"


def _progress(iterable, **kw):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kw)
    except ImportError:  # pragma: no cover
        return iterable


def _samples(cfg: Dict[str, Any]) -> List[Sample]:
    d = cfg["data"]
    return select(load_manifest(d["manifest"]), d.get("subsets") or (), int(d.get("limit_per_subset") or 0))


def _out(cfg: Dict[str, Any], name: str) -> Path:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out / name


# --------------------------------------------------------------------------- run info
def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def write_run_info(cfg: Dict[str, Any], stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    path = _out(cfg, "run_info.json")
    info = read_json(path) if path.exists() else {"stages": {}}
    info["stages"][stage] = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": commit,
        "packages": {p: _pkg_version(p) for p in ("truthlens", "torch", "transformers", "openai",
                                                  "numpy", "pillow")},
        "config": cfg,
        **(extra or {}),
    }
    write_json(path, info)


# --------------------------------------------------------------------------- step 1+2: probing
def run_probe(cfg: Dict[str, Any], backend=None) -> Path:
    """Query the LVLM with every prompt ``p_i`` for every image (``a_i = f_MM(I, p_i)``)."""
    from .lvlm import make_lvlm

    samples = _samples(cfg)
    categories = resolve_categories(cfg["probe"]["categories"])
    out_path = _out(cfg, PROBES_FILE)
    done = set()
    if out_path.exists():
        done = {(r["id"], r["category"]) for r in iter_jsonl(out_path) if not r.get("error")}
    todo = [s for s in samples if any((s.id, c) not in done for c in categories)]
    log.info("probe: %d images x %d categories; %d images already complete", len(samples),
             len(categories), len(samples) - len(todo))
    if not todo:
        return out_path

    pc = cfg["probe"]
    if backend is None:
        backend = make_lvlm(pc["backend"], model_path=pc.get("model_path"), device=pc.get("device"),
                            generation=pc.get("generation"), **(pc.get("options") or {}))
    write_run_info(cfg, "probe", {"lvlm": backend.describe()})
    n_err = 0
    with JsonlWriter(out_path) as writer:
        for sample in _progress(todo, desc="probing", unit="img"):
            base = {"id": sample.id, "subset": sample.subset, "label": sample.label,
                    "backend": backend.name, "model_path": backend.model_path}
            try:
                image = load_image(sample.path)
                image_error = None
            except Exception as exc:  # unreadable/corrupt file
                image, image_error = None, f"{type(exc).__name__}: {exc}"
            for cat in categories:
                if (sample.id, cat) in done:
                    continue
                seed = stable_seed(cfg["seed"], sample.id, cat)
                rec = dict(base, category=cat, prompt=PROMPTS[cat], seed=seed, response=None, error=image_error)
                if image is not None:
                    try:
                        set_seed(seed)
                        rec["response"] = backend.generate(image, PROMPTS[cat])
                    except Exception as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                if rec["error"]:
                    n_err += 1
                    log.warning("probe failed for %s / %s: %s", sample.id, cat, rec["error"])
                writer.write(rec)
    if n_err:
        log.warning("probe: %d failed queries recorded (re-run the command to retry them)", n_err)
    return out_path


# --------------------------------------------------------------------------- step 3: aggregation
def run_aggregate(cfg: Dict[str, Any]) -> Path:
    samples = _samples(cfg)
    ac = cfg["aggregate"]
    categories = resolve_categories(ac["categories"])
    probes_path = Path(ac["probes_file"]) if ac.get("probes_file") else _out(cfg, PROBES_FILE)
    if not probes_path.exists():
        raise FileNotFoundError(f"{probes_path} not found - run the probe stage first")
    answers = collect_answers(read_jsonl(probes_path))
    records, n_partial, n_skipped = [], 0, 0
    for s in samples:
        got = answers.get(s.id, {})
        missing = [c for c in categories if c not in got]
        if missing:
            if ac["on_missing"] == "error":
                raise ValueError(f"{s.id}: missing probe answers for {missing}")
            if ac["on_missing"] == "skip" or len(missing) == len(categories):
                n_skipped += 1
                continue
            n_partial += 1
        summary, used = aggregate_answers(got, categories, ac["mode"])
        records.append({"id": s.id, "subset": s.subset, "label": s.label, "mode": ac["mode"],
                        "categories": used, "missing_categories": missing, "summary": summary})
    if n_partial:
        log.warning("aggregate: %d images aggregated with missing categories (see 'missing_categories')",
                    n_partial)
    if n_skipped:
        log.warning("aggregate: %d images skipped (no usable answers or on_missing=skip); they count as "
                    "invalid predictions at evaluation", n_skipped)
    out = _out(cfg, SUMMARIES_FILE)
    write_jsonl(out, records)
    log.info("aggregate: wrote %d summaries to %s", len(records), out)
    return out


# --------------------------------------------------------------------------- step 4: LLM verdict
def run_judge(cfg: Dict[str, Any], judge=None) -> Path:
    jc = cfg["judge"]
    summaries_path = _out(cfg, SUMMARIES_FILE)
    if not summaries_path.exists():
        raise FileNotFoundError(f"{summaries_path} not found - run the aggregate stage first")
    summaries = read_jsonl(summaries_path)
    out_path = _out(cfg, VERDICTS_FILE)
    done = set()
    if out_path.exists():
        done = {r["id"] for r in iter_jsonl(out_path) if not r.get("error")}
    todo = [s for s in summaries if s["id"] not in done]
    log.info("judge: %d summaries, %d already judged", len(summaries), len(summaries) - len(todo))
    if not todo:
        out_path.touch()  # an empty verdict file lets evaluation report all images as invalid
        return out_path
    if judge is None:
        judge = make_judge(jc["backend"], model=jc["model"], temperature=jc["temperature"],
                           max_tokens=jc["max_tokens"], seed=jc["seed"], timeout=jc["timeout"],
                           max_retries=jc["max_retries"])
    write_run_info(cfg, "judge", {"judge_backend": judge.name, "judge_model": judge.model})
    n_err = n_unparsed = 0
    with JsonlWriter(out_path) as writer:
        for s in _progress(todo, desc="judging", unit="img"):
            rec = {"id": s["id"], "subset": s["subset"], "label": s["label"], "judge_backend": judge.name,
                   "judge_model": judge.model, "prompt": jc["prompt"], "verdict": None, "confidence": None,
                   "justification": None, "parse_status": None, "raw_response": None, "error": None}
            try:
                raw = judge.complete(build_messages(s["summary"], jc["prompt"]))
                parsed = parse_judge_response(raw)
                rec.update(raw_response=raw, verdict=parsed.verdict, confidence=parsed.confidence,
                           justification=parsed.justification, parse_status=parsed.parse_status,
                           returned_model=getattr(judge, "last_model", None),
                           system_fingerprint=getattr(judge, "last_system_fingerprint", None))
                n_unparsed += parsed.verdict is None
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                n_err += 1
                log.warning("judge failed for %s: %s", s["id"], rec["error"])
                if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                    writer.write(rec)
                    raise
            writer.write(rec)
    if n_err or n_unparsed:
        log.warning("judge: %d API failures (re-run to retry), %d unparseable replies", n_err, n_unparsed)
    return out_path


# --------------------------------------------------------------------------- "Yes or No Question" baseline
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(text: Optional[str]) -> Optional[str]:
    """Return 'yes'/'no' if the reply contains exactly one of the standalone words; else None.

    Hedged replies containing both words ("yes and no") are treated as unparseable.
    """
    if not text:
        return None
    found = {w.lower() for w in _YES_NO_RE.findall(text)}
    return found.pop() if len(found) == 1 else None


def run_yesno(cfg: Dict[str, Any], backend=None) -> Path:
    """Table 2 "Yes or No Question": ask the LVLM directly, without probes or LLM."""
    from .lvlm import make_lvlm

    samples = _samples(cfg)
    yc = cfg["yesno"]
    yes_means = str(yc["yes_means"]).upper()
    no_means = "REAL" if yes_means == "FAKE" else "FAKE"
    resp_path = _out(cfg, YESNO_RESPONSES_FILE)
    done = set()
    if resp_path.exists():
        done = {r["id"] for r in iter_jsonl(resp_path) if not r.get("error")}
    todo = [s for s in samples if s.id not in done]
    if todo:
        pc = cfg["probe"]
        if backend is None:
            backend = make_lvlm(pc["backend"], model_path=pc.get("model_path"), device=pc.get("device"),
                                generation=pc.get("generation"), **(pc.get("options") or {}))
        write_run_info(cfg, "yesno", {"lvlm": backend.describe()})
        with JsonlWriter(resp_path) as writer:
            for s in _progress(todo, desc="yes/no", unit="img"):
                seed = stable_seed(cfg["seed"], s.id, "yesno")
                rec = {"id": s.id, "subset": s.subset, "label": s.label, "prompt": yc["prompt"],
                       "seed": seed, "response": None, "error": None, "backend": backend.name}
                try:
                    set_seed(seed)
                    rec["response"] = backend.generate(load_image(s.path), yc["prompt"])
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                writer.write(rec)
    latest: Dict[str, Dict] = {}
    for r in iter_jsonl(resp_path):
        if r["id"] not in latest or not r.get("error"):
            latest[r["id"]] = r
    verdicts = []
    for r in latest.values():
        answer = parse_yes_no(r.get("response"))
        verdict = {"yes": yes_means, "no": no_means}.get(answer) if answer else None
        verdicts.append({"id": r["id"], "subset": r["subset"], "label": r["label"], "answer": answer,
                         "verdict": verdict, "confidence": None, "error": r.get("error"),
                         "parse_status": "ok" if answer else "unparseable"})
    out = _out(cfg, YESNO_VERDICTS_FILE)
    write_jsonl(out, verdicts)
    return out


# --------------------------------------------------------------------------- evaluation
def load_predictions(path: Path) -> Dict[str, Dict]:
    preds: Dict[str, Dict] = {}
    for r in iter_jsonl(path):
        if r["id"] in preds and r.get("error"):
            continue  # keep an earlier successful record
        preds[r["id"]] = r
    return preds


def run_evaluate(cfg: Dict[str, Any], verdicts_path: Optional[str] = None,
                 metrics_path: Optional[str] = None) -> Dict[str, Any]:
    vpath = Path(verdicts_path) if verdicts_path else _out(cfg, VERDICTS_FILE)
    if not vpath.exists():
        raise FileNotFoundError(f"{vpath} not found")
    samples = _samples(cfg)
    preds = load_predictions(vpath)
    unknown = set(preds) - {s.id for s in samples}
    if unknown:
        log.warning("evaluate: %d predictions have ids not in the (filtered) manifest; ignored", len(unknown))
    policy = cfg["evaluate"]["invalid_policy"]
    results = evaluate_by_subset(samples, preds, policy)
    report = {"verdicts_file": str(vpath), "invalid_policy": policy, "positive_class": "FAKE",
              "results": results}
    mpath = Path(metrics_path) if metrics_path else vpath.with_name(
        YESNO_METRICS_FILE if vpath.name == YESNO_VERDICTS_FILE else METRICS_FILE)
    write_json(mpath, report)
    print(format_table(results))
    print(f"\nmetrics written to {mpath}")
    return report


# --------------------------------------------------------------------------- legacy import
LEGACY_STEM_TO_CATEGORY = {
    # file stems used by the released concatinate_jsons.py (mapping of
    # 'faceattributes' is inferred from the name - please confirm)
    "eyes": "eyes", "faceattributes": "facial_features", "facialhair": "facial_hair",
    "realism_2": "overall_realism", "realism": "overall_realism", "reflections": "reflections",
    "symmetry_2": "symmetry", "symmetry": "symmetry", "texture": "texture",
    "lighting": "lighting", "background": "background",
}


def import_legacy(cfg: Dict[str, Any], json_path: str, category: str, subset: str) -> int:
    """Convert a released-format probe file ``{image_path: answer}`` into ``probes.jsonl`` records.

    Images are matched by file name within ``subset`` because the released
    files are keyed by absolute paths on the original machine.
    """
    if category not in PROMPTS:
        raise ValueError(f"Unknown category '{category}'. Valid: {list(PROMPTS)}")
    data = read_json(json_path)
    if data and all(isinstance(v, dict) for v in data.values()):  # nested combined file {subfolder: {...}}
        flat = {}
        for v in data.values():
            flat.update(v)
        data = flat
    by_name: Dict[str, Sample] = {}
    for s in load_manifest(cfg["data"]["manifest"]):
        if s.subset != subset:
            continue
        name = Path(s.path).name
        if name in by_name:
            raise ValueError(f"File name '{name}' is not unique in subset '{subset}'; cannot match legacy keys")
        by_name[name] = s
    if not by_name:
        raise ValueError(f"No samples with subset '{subset}' in the manifest")
    n, unmatched = 0, 0
    with JsonlWriter(_out(cfg, PROBES_FILE)) as writer:
        for key, answer in data.items():
            s = by_name.get(Path(str(key).replace("\\", "/")).name)
            if s is None:
                unmatched += 1
                continue
            is_err = isinstance(answer, str) and answer.startswith("Error: ")
            writer.write({"id": s.id, "subset": s.subset, "label": s.label, "category": category,
                          "prompt": PROMPTS[category], "seed": None, "backend": "legacy-import",
                          "model_path": None, "source_file": str(json_path),
                          "response": None if is_err else answer, "error": answer if is_err else None})
            n += 1
    if unmatched:
        log.warning("import-legacy: %d keys in %s did not match any image in subset '%s'",
                    unmatched, json_path, subset)
    log.info("import-legacy: imported %d answers for category '%s' (%s)", n, category,
             CATEGORY_TITLES[category])
    return n
