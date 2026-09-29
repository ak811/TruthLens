"""Experiment configuration (YAML) with defaults, validation and ``--set key=value`` overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .prompts import YES_NO_PROMPT

# Defaults follow the paper where it is explicit (nine prompts, GPT-4 judge)
# and the released code where the paper is silent (LVLM decoding). See
# docs/AUDIT.md for every such choice.
DEFAULTS: Dict[str, Any] = {
    "run_name": "truthlens",
    "output_dir": "outputs/truthlens",
    "seed": 0,
    "data": {
        "manifest": "data/manifest.jsonl",
        "subsets": [],            # empty = all subsets in the manifest
        "limit_per_subset": 0,    # 0 = no limit
    },
    "probe": {
        "backend": "chatunivi",   # chatunivi | llava15 | blip2 | cogvlm | mock
        "model_path": None,       # None = backend default checkpoint
        "device": "cuda",
        "categories": "all",      # "all" | "released_7" | list of category keys
        "options": {},            # backend-specific keyword arguments (e.g. conv_mode, dtype)
        "generation": {
            "do_sample": True,
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024,
        },
    },
    "aggregate": {
        "mode": "structured",     # structured | pipe (released concatinate_jsons.py)
        "categories": "all",
        "on_missing": "warn",     # warn | error | skip
        "probes_file": None,      # None = <output_dir>/probes.jsonl (set to reuse probes, e.g. ablations)
    },
    "judge": {
        "backend": "openai",      # openai | mock
        "model": "gpt-4",         # paper: GPT-4; released code: gpt-3.5-turbo
        "prompt": "truthlens_v1",
        "temperature": None,      # None = API default (released behaviour); paper does not specify
        "max_tokens": None,
        "seed": None,
        "timeout": 120,
        "max_retries": 6,
    },
    "yesno": {
        "prompt": YES_NO_PROMPT,
        "yes_means": "FAKE",
    },
    "evaluate": {
        "invalid_policy": "incorrect",  # incorrect | exclude
    },
}

_FREE_FORM = {("probe", "options")}


def _merge(base: Dict[str, Any], update: Dict[str, Any], path=()) -> Dict[str, Any]:
    for key, value in update.items():
        here = path + (key,)
        if key not in base:
            raise KeyError(f"Unknown configuration key: {'.'.join(here)}")
        if isinstance(base[key], dict) and here not in _FREE_FORM:
            if not isinstance(value, dict):
                raise TypeError(f"Configuration key {'.'.join(here)} must be a mapping")
            _merge(base[key], value, here)
        else:
            base[key] = value
    return base


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Unknown configuration key: {'.'.join(parts[:i + 1])}")
        node = node[part]
    leaf = parts[-1]
    if leaf not in node and tuple(parts[:-1]) not in _FREE_FORM:
        raise KeyError(f"Unknown configuration key: {dotted}")
    node[leaf] = value


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise TypeError(f"{path} must contain a YAML mapping")
        _merge(cfg, user)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    from .aggregate import AGGREGATION_MODES
    from .lvlm import BACKENDS
    from .metrics import INVALID_POLICIES
    from .prompts import resolve_categories

    if cfg["probe"]["backend"] not in BACKENDS:
        raise ValueError(f"probe.backend must be one of {BACKENDS}")
    if cfg["aggregate"]["mode"] not in AGGREGATION_MODES:
        raise ValueError(f"aggregate.mode must be one of {AGGREGATION_MODES}")
    if cfg["aggregate"]["on_missing"] not in ("warn", "error", "skip"):
        raise ValueError("aggregate.on_missing must be warn, error or skip")
    if cfg["judge"]["backend"] not in ("openai", "mock"):
        raise ValueError("judge.backend must be openai or mock")
    if cfg["evaluate"]["invalid_policy"] not in INVALID_POLICIES:
        raise ValueError(f"evaluate.invalid_policy must be one of {INVALID_POLICIES}")
    if str(cfg["yesno"]["yes_means"]).upper() not in ("REAL", "FAKE"):
        raise ValueError("yesno.yes_means must be REAL or FAKE")
    resolve_categories(cfg["probe"]["categories"])
    resolve_categories(cfg["aggregate"]["categories"])


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
