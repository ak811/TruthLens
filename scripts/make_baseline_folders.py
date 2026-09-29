#!/usr/bin/env python
"""Create ``<out>/0_real`` and ``<out>/1_fake`` folders for the baselines from a TruthLens manifest.

Guarantees CNNDetection/DIRE are evaluated on exactly the images used by TruthLens.
Files are symlinked (default) or copied; names are prefixed with the subset to avoid collisions.

    python scripts/make_baseline_folders.py --manifest data/manifest.jsonl --fake-subset ldm --out data/baselines/ldm
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truthlens.data import load_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fake-subset", required=True, help="fake subset to pair with all real images, e.g. ldm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args()
    samples = load_manifest(args.manifest)
    chosen = [s for s in samples if s.label == "REAL" or s.subset == args.fake_subset]
    if not any(s.subset == args.fake_subset for s in chosen):
        sys.exit(f"no fake samples with subset '{args.fake_subset}'")
    counts = {"0_real": 0, "1_fake": 0}
    for s in chosen:
        folder = "0_real" if s.label == "REAL" else "1_fake"
        dst = Path(args.out) / folder / s.id.replace("/", "__")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(s.path, dst)
        else:
            os.symlink(os.path.abspath(s.path), dst)
        counts[folder] += 1
    print(f"{args.out}: {counts['0_real']} real, {counts['1_fake']} fake")


if __name__ == "__main__":
    main()
