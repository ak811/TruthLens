"""Command-line interface: ``truthlens <command> [options]`` (or ``python -m truthlens``)."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from . import __version__


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="YAML config file (see configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set judge.model=gpt-4o (repeatable)")
    p.add_argument("--output-dir", help="shortcut for --set output_dir=...")
    p.add_argument("--manifest", help="shortcut for --set data.manifest=...")
    p.add_argument("--limit", type=int, help="shortcut for --set data.limit_per_subset=N (debugging)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="truthlens", description=__doc__)
    parser.add_argument("--version", action="version", version=f"truthlens {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="build a dataset manifest from image folders")
    m.add_argument("--real", action="append", default=[], metavar="[NAME=]DIR",
                   help="folder of real images (subset name defaults to 'real'); repeatable")
    m.add_argument("--fake", action="append", default=[], metavar="NAME=DIR",
                   help="folder of fake images with a subset name, e.g. ldm=data/ldm; repeatable")
    m.add_argument("--out", required=True, help="output manifest path (.jsonl)")
    m.add_argument("--recursive", action="store_true", help="search sub-folders")
    m.add_argument("--limit-per-subset", type=int, default=0, help="keep the first N (sorted) images per subset")

    for name, help_ in [("probe", "step 1+2: query the LVLM with the probe prompts"),
                        ("aggregate", "step 3: aggregate probe answers into summaries"),
                        ("judge", "step 4: LLM verdict and justification"),
                        ("evaluate", "compute metrics from a verdict file"),
                        ("run", "probe + aggregate + judge + evaluate"),
                        ("yesno", "Table 2 'Yes or No Question' baseline (LVLM only) + evaluation")]:
        p = sub.add_parser(name, help=help_)
        _common(p)
        if name == "evaluate":
            p.add_argument("--verdicts", help="verdict JSONL (default: <output_dir>/verdicts.jsonl)")
            p.add_argument("--out", help="metrics JSON path")

    li = sub.add_parser("import-legacy", help="import a released-format per-category JSON into probes.jsonl")
    _common(li)
    li.add_argument("json_file")
    li.add_argument("--category", required=True, help="probe category key, e.g. eyes")
    li.add_argument("--subset", required=True, help="manifest subset the file belongs to, e.g. ldm")

    sub.add_parser("prompts", help="print the probe prompt set")
    return parser


def _load_cfg(args):
    from .config import load_config

    overrides = list(args.overrides)
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.manifest:
        overrides.append(f"data.manifest={args.manifest}")
    if args.limit is not None:
        overrides.append(f"data.limit_per_subset={args.limit}")
    return load_config(args.config, overrides)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "prompts":
        from .prompts import CATEGORY_TITLES, PROMPTS

        for key, prompt in PROMPTS.items():
            print(f"{key:<16} [{CATEGORY_TITLES[key]}]\n    {prompt}\n")
        return 0

    if args.command == "manifest":
        from .data import build_manifest, parse_source

        if not args.real and not args.fake:
            print("error: provide at least one --real or --fake folder", file=sys.stderr)
            return 2
        sources = []
        for spec in args.real:
            name, path = parse_source(spec, "real")
            sources.append((name, "REAL", path))
        for spec in args.fake:
            if "=" not in spec:
                print(f"error: --fake requires NAME=DIR (got '{spec}')", file=sys.stderr)
                return 2
            name, path = parse_source(spec, "")
            sources.append((name, "FAKE", path))
        samples = build_manifest(sources, args.out, args.recursive, args.limit_per_subset)
        counts = {}
        for s in samples:
            counts[(s.subset, s.label)] = counts.get((s.subset, s.label), 0) + 1
        for (subset, label), n in counts.items():
            print(f"{subset:<12} {label:<5} {n}")
        print(f"wrote {len(samples)} records to {args.out}")
        return 0

    from . import pipeline
    from .config import dump_config

    cfg = _load_cfg(args)
    dump_config(cfg, f"{cfg['output_dir']}/config.resolved.yaml")

    if args.command == "probe":
        pipeline.run_probe(cfg)
    elif args.command == "aggregate":
        pipeline.run_aggregate(cfg)
    elif args.command == "judge":
        pipeline.run_judge(cfg)
    elif args.command == "evaluate":
        pipeline.run_evaluate(cfg, args.verdicts, args.out)
    elif args.command == "run":
        pipeline.run_probe(cfg)
        pipeline.run_aggregate(cfg)
        pipeline.run_judge(cfg)
        pipeline.run_evaluate(cfg)
    elif args.command == "yesno":
        path = pipeline.run_yesno(cfg)
        pipeline.run_evaluate(cfg, str(path))
    elif args.command == "import-legacy":
        pipeline.import_legacy(cfg, args.json_file, args.category, args.subset)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
