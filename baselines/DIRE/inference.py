"""Evaluate a pretrained DIRE classifier on a labelled real/fake folder.

This script was added by the TruthLens authors (it is derived from upstream
``demo.py``) and produces the DIRE rows of Tables 1 and 3.

IMPORTANT - input type. The DIRE classifier is trained on *DIRE maps*
(|x - reconstruction(x)|), not on raw RGB images. The intended protocol is:

1. compute DIRE maps for every image with ``guided-diffusion/compute_dire.py``;
2. run this script on the resulting ``<dire_dir>/0_real`` and ``<dire_dir>/1_fake``.

Running it directly on raw images feeds the classifier out-of-distribution
inputs. Use ``--input_type`` to record which protocol was used; it is written
to the JSON output so results remain traceable.

Expected layout::

    <dir>/0_real/*.png|jpg|jpeg
    <dir>/1_fake/*.png|jpg|jpeg
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from utils.utils import get_network, str2bool

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir", default="examples/realfakedir_real_first1000/", type=str,
                        help="directory containing 0_real/ and 1_fake/ sub-directories")
    parser.add_argument("-m", "--model_path", type=str, default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth")
    parser.add_argument("--input_type", choices=["dire", "raw"], required=True,
                        help="whether --dir holds DIRE maps (intended protocol) or raw RGB images")
    parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for classifying synthetic images")
    parser.add_argument("--verbose", action="store_true", help="print a line per image")
    parser.add_argument("--output_json", type=str, default=None, help="optional path to write metrics as JSON")
    return parser.parse_args(argv)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "real_accuracy": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "fake_accuracy": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "ap": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.input_type == "raw":
        print("WARNING: evaluating the DIRE classifier on raw RGB images. DIRE was trained on DIRE maps "
              "produced by guided-diffusion/compute_dire.py; raw-image results are not the DIRE protocol.",
              file=sys.stderr)

    real_dir = os.path.join(args.dir, "0_real")
    fake_dir = os.path.join(args.dir, "1_fake")
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(f"Both '0_real' and '1_fake' directories must exist under {args.dir}")

    real_files, fake_files = list_images(real_dir), list_images(fake_dir)
    file_list = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    print(f"Loaded {len(real_files)} real images and {len(fake_files)} fake images.")
    if not file_list:
        raise RuntimeError("No images found.")

    model = get_network(args.arch)
    state_dict = torch.load(args.model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    if not args.use_cpu:
        model.cuda()

    trans = transforms.Compose((transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()))

    y_true, y_prob = [], []
    for img_path, label in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
        img = trans(Image.open(img_path).convert("RGB"))
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        if not args.use_cpu:
            in_tens = in_tens.cuda()
        with torch.no_grad():
            prob = model(in_tens).sigmoid().item()
        y_prob.append(prob)
        y_true.append(label)
        if args.verbose:
            print(f"Image: {img_path} | P(synthetic): {prob:.4f} | "
                  f"Pred: {int(prob >= args.threshold)} | GT: {label}")

    metrics = compute_metrics(y_true, y_prob, args.threshold)
    print("*" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    return metrics


if __name__ == "__main__":
    main()
