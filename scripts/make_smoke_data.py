#!/usr/bin/env python
"""Create a tiny synthetic dataset for the CPU smoke test.

Real images are dark and fake images bright, so the mock LVLM (which answers
from brightness) and mock judge must reach 100% accuracy if the pipeline
plumbing (ids, labels, aggregation, parsing, metrics) is correct. The images
contain no faces and have no scientific meaning.

Real and fake images deliberately share file names to exercise id-collision
handling (a bug in the released code).
"""
import argparse
import random
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/smoke")
    ap.add_argument("--n", type=int, default=6, help="images per subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.root)
    specs = {"real": (10, 90), "ldm": (170, 250), "progan": (170, 250)}
    for subset, (lo, hi) in specs.items():
        d = root / subset
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            size = rng.choice([32, 48, 64])
            color = tuple(rng.randint(lo, hi) for _ in range(3))
            mode = "RGBA" if i % 3 == 0 else "RGB"  # exercise RGBA -> RGB conversion
            img = Image.new(mode, (size, size), color + ((255,) if mode == "RGBA" else ()))
            img.save(d / f"{i:05d}.png")
    print(f"wrote {3 * args.n} images under {root}")


if __name__ == "__main__":
    main()
