import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from utils.utils import get_network, str2bool, to_cuda

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-d",
    "--dir",
    default="examples/realfakedir_real_first1000/",
    type=str,
    help="path to the directory containing 'real' and 'fake' subdirectories",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classifying synthetic images")

args = parser.parse_args()

real_dir = os.path.join(args.dir, "0_real")
fake_dir = os.path.join(args.dir, "1_fake")

if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
    raise FileNotFoundError(f"Both 'real' and 'fake' directories must exist under {args.dir}")

real_files = sorted(glob.glob(os.path.join(real_dir, "*.jpg"))
                    + glob.glob(os.path.join(real_dir, "*.png"))
                    + glob.glob(os.path.join(real_dir, "*.JPEG")))
fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.jpg"))
                    + glob.glob(os.path.join(fake_dir, "*.png"))
                    + glob.glob(os.path.join(fake_dir, "*.JPEG")))

file_list = [(file, 0) for file in real_files] + [(file, 1) for file in fake_files]
print(f"Loaded {len(real_files)} real images and {len(fake_files)} fake images.")

model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()
if not args.use_cpu:
    model.cuda()

print("*" * 50)

trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    )
)

y_true = []
y_pred_prob = []
y_pred_labels = []

for img_path, label in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    if not args.use_cpu:
        in_tens = in_tens.cuda()

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
        y_pred_prob.append(prob)
        pred_label = int(prob >= args.threshold)
        y_pred_labels.append(pred_label)
        y_true.append(label)

        print(f"Image: {img_path} | Prob of being synthetic: {prob:.4f} | Predicted Label: {pred_label} | Ground Truth: {label}")

if len(set(y_true)) > 1:
    roc_auc = roc_auc_score(y_true, y_pred_prob)
else:
    roc_auc = "Undefined (only one class present)"
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average="binary")
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

print("*" * 50)
print(f"ROC AUC: {roc_auc}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
