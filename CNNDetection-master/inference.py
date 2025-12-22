import argparse
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    confusion_matrix,
)
from networks.resnet import resnet50
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dir', nargs='+', type=str, default='examples/realfakedir_Real1000_ProGAN1000')
parser.add_argument('-m', '--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c', '--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

opt = parser.parse_args()

if not opt.size_only:
    model = resnet50(num_classes=1)
    if opt.model_path is not None:
        state_dict = torch.load(opt.model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    model.eval()
    if not opt.use_cpu:
        model.cuda()

trans_init = []
if opt.crop is not None:
    trans_init = [transforms.CenterCrop(opt.crop)]
    print(f"Cropping to [{opt.crop}]")
else:
    print("Not cropping")

trans = transforms.Compose(trans_init + [
    transforms.Resize((224, 224)),  # Ensure uniform size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if type(opt.dir) == str:
    opt.dir = [opt.dir]

print(f'Loading [{len(opt.dir)}] datasets')
data_loaders = []
for dir in opt.dir:
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Dataset directory '{dir}' does not exist.")
    dataset = datasets.ImageFolder(dir, transform=trans)
    data_loaders += [torch.utils.data.DataLoader(dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 num_workers=opt.workers,
                                                 pin_memory=False)]

y_true, y_pred = [], []
Hs, Ws = [], []

with torch.no_grad():
    for data_loader in data_loaders:
        for idx, (data, label) in enumerate(tqdm(data_loader)):
            try:
                Hs.append(data.shape[2])
                Ws.append(data.shape[3])

                y_true.extend(label.flatten().tolist())
                if not opt.size_only:
                    if not opt.use_cpu:
                        data = data.cuda()
                    y_pred.extend(model(data).sigmoid().flatten().tolist())
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                continue

Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(
    np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs * Ws) / 1e6, np.std(Hs * Ws) / 1e6))
print(f'Num reals: {np.sum(1 - y_true)}, Num fakes: {np.sum(y_true)}')

if not opt.size_only:
    y_pred_binary = (y_pred >= 0.5).astype(int)  # Binary predictions based on threshold
    ap = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_binary)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average="binary")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    roc_auc = roc_auc_score(y_true, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall_curve, precision_curve)

    print("*" * 50)
    print(f'AP: {ap * 100:.2f}')
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
