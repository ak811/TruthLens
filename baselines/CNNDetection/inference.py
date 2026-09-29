"""Evaluate a pretrained CNNDetection model on a labelled real/fake image folder.

This script was added by the TruthLens authors (it is not part of upstream
CNNDetection) and produces the CNNDetection rows of Tables 1 and 3.

Expected input layout (one or more directories passed via ``--dir``)::

    <dir>/0_real/*.png|jpg|...
    <dir>/1_fake/*.png|jpg|...

``torchvision.datasets.ImageFolder`` assigns labels by *alphabetical* order of
the sub-folder names, so the folders MUST be named ``0_real`` and ``1_fake``;
the script now refuses any other layout instead of silently inverting labels.

Preprocessing note: by default every image is resized to 224x224 (this is the
behaviour of the script used for the paper). Upstream CNNDetection recommends
evaluating *without* resizing because resampling suppresses the
high-frequency artifacts the detector relies on. Pass ``--no_resize`` (which
forces ``--batch_size 1``) to evaluate the upstream way.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from networks.resnet import resnet50

EXPECTED_CLASSES = {"0_real": 0, "1_fake": 1}


def is_lfs_pointer(path):
    """Return True if ``path`` is an un-fetched Git LFS pointer file."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir", nargs="+", type=str, default=["examples/realfakedir_Real1000_ProGAN1000"],
                        help="one or more directories, each containing 0_real/ and 1_fake/")
    parser.add_argument("-m", "--model_path", type=str, default="weights/blur_jpg_prob0.5.pth")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-j", "--workers", type=int, default=4, help="number of data-loading workers")
    parser.add_argument("-c", "--crop", type=int, default=None, help="centre-crop size (default: no crop)")
    parser.add_argument("--no_resize", action="store_true",
                        help="do not resize to 224x224 (upstream-recommended protocol; forces batch_size=1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold on sigmoid output")
    parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
    parser.add_argument("--size_only", action="store_true", help="only report image sizes")
    parser.add_argument("--output_json", type=str, default=None, help="optional path to write metrics as JSON")
    return parser.parse_args(argv)


def build_transform(opt):
    trans = []
    if opt.crop is not None:
        trans.append(transforms.CenterCrop(opt.crop))
        print(f"Cropping to [{opt.crop}]")
    else:
        print("Not cropping")
    if not opt.no_resize:
        trans.append(transforms.Resize((224, 224)))
    trans += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(trans)


def check_class_layout(dataset, root):
    if dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(
            f"'{root}' must contain exactly the sub-folders 0_real/ and 1_fake/ "
            f"(found {dataset.class_to_idx}). Other names would silently invert or corrupt labels."
        )


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    both = len(np.unique(y_true)) == 2
    metrics = {
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "real_accuracy": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "fake_accuracy": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        "ap": float(average_precision_score(y_true, y_score)) if both else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if both:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = float(auc(r_curve, p_curve))
    return metrics


def main(argv=None):
    opt = parse_args(argv)
    if opt.no_resize and opt.batch_size != 1:
        print("--no_resize: images may differ in size, setting batch_size=1")
        opt.batch_size = 1

    model = None
    if not opt.size_only:
        if is_lfs_pointer(opt.model_path):
            sys.exit(f"'{opt.model_path}' is a Git LFS pointer, not the weights. "
                     "Run `git lfs pull` or `bash weights/download_weights.sh`.")
        model = resnet50(num_classes=1)
        state_dict = torch.load(opt.model_path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.eval()
        if not opt.use_cpu:
            model.cuda()

    trans = build_transform(opt)
    data_loaders = []
    print(f"Loading [{len(opt.dir)}] datasets")
    for root in opt.dir:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory '{root}' does not exist.")
        dataset = datasets.ImageFolder(root, transform=trans)
        check_class_layout(dataset, root)
        data_loaders.append(torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=False))

    y_true, y_score, heights, widths = [], [], [], []
    with torch.no_grad():
        for loader in data_loaders:
            for data, label in tqdm(loader):
                # Compute the prediction first and only then record labels/sizes, so a
                # failure can never leave y_true and y_score with different lengths.
                if model is not None:
                    inp = data if opt.use_cpu else data.cuda()
                    y_score.extend(model(inp).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
                heights.extend([data.shape[2]] * data.shape[0])
                widths.extend([data.shape[3]] * data.shape[0])

    heights, widths = np.array(heights), np.array(widths)
    y_true = np.array(y_true)
    print("Average sizes: [{:.2f}+/-{:.2f}] x [{:.2f}+/-{:.2f}] = [{:.2f}+/-{:.2f} Mpix]".format(
        heights.mean(), heights.std(), widths.mean(), widths.std(),
        (heights * widths).mean() / 1e6, (heights * widths).std() / 1e6))
    print(f"Num reals: {int(np.sum(1 - y_true))}, Num fakes: {int(np.sum(y_true))}")
    if opt.size_only:
        return None

    metrics = compute_metrics(y_true, y_score, opt.threshold)
    print("*" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    if opt.output_json:
        with open(opt.output_json, "w") as f:
            json.dump({"args": vars(opt), "metrics": metrics}, f, indent=2)
    return metrics


if __name__ == "__main__":
    main()
