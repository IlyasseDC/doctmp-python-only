import os
import cv2
import lmdb
import torch
import numpy as np
import torch.nn as nn
import gc
import math
import time
import copy
import logging
import torch.optim as optim
import torch.distributed as dist
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from models.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
import albumentations as A
from models.dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataset_cltd import TamperDatasetCLTD
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.utils.data import Subset, DataLoader

from models.dtd import seg_dtd
from dataset_cltd import TamperDatasetCLTD


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = seg_dtd("", 2).to(device)

# Si on est sur CUDA, alors on active DataParallel
if device.type == 'cuda':
    model = nn.DataParallel(model)
def denormalize(tensor, mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def save_visualization(img, gt_mask, pred_mask, index, output_dir="vis_preds_FCD"):
    """
    Sauvegarde une figure avec l'image originale dénormalisée, le masque réel et le masque prédit.
    """
    import matplotlib.pyplot as plt
    import os

    img = denormalize(img.cpu()).numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Mask GT")
    axs[1].axis("off")
    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Mask Pred")
    axs[2].axis("off")

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{index}.png"))
    plt.close()


def eval_net_dtd(model, test_data, plot=False,device='cpu'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=2, num_workers=8, shuffle=False)
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    ckpt = torch.load(args.pth,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    iou=IOUMetric(2)
    precisons = []
    recalls = []
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
            dct_coef = dct_coef.long() 
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(batch['q'].view(-1, 1, 8, 8).to(device))
            pred = model(data,dct_coef,qs)             
            predt = pred.argmax(1)
            predt = pred.argmax(1)
            targt = target.squeeze(1)
            targt = targt.to(device)
            predt = predt.to(device)
            # compute metrics
            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))

            precisons.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())

            # add to IoU using the same shapes as in training
            iou.add_batch(predt.cpu().numpy(), targt.cpu().numpy())

            # Visualiser les 10 premières images du dataset
            max_vis = 50  # nombre total d'images à visualiser
            img_counter = batch_idx * data.shape[0]

            for b in range(data.shape[0]):
                global_index = batch_idx * data.shape[0] + b
                if global_index >= max_vis:
                    break

                img = data[b].cpu()
                gt = target[b, 0].cpu() if target.dim() == 4 else target[b].cpu()
                pd = predt[b].cpu()
                save_visualization(img, gt, pd, index=global_index)

        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))

def denormalize(tensor, mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def save_visualization(img, gt_mask, pred_mask, index, output_dir="vis_preds"):
    """
    Affiche et sauvegarde :
    - Image d'origine avec superposition GT + prédiction
    - Masque GT
    - Masque prédiction
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dénormalisation + conversion image
    img_np = denormalize(img.cpu()).numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    gt = gt_mask.cpu().numpy().astype(np.uint8)
    pred = pred_mask.cpu().numpy().astype(np.uint8)

    # Création de masques colorés pour superposition
    gt_overlay = np.zeros_like(img_np)
    pred_overlay = np.zeros_like(img_np)

    # GT en vert (canal G)
    gt_overlay[:, :, 1] = gt * 255

    # Prédiction en rouge (canal R)
    pred_overlay[:, :, 0] = pred * 255

    # Fusion alpha
    overlaid = cv2.addWeighted(img_np, 0.7, gt_overlay, 0.3, 0)
    overlaid = cv2.addWeighted(overlaid, 0.7, pred_overlay, 0.3, 0)

    # Affichage
    plt.figure(figsize=(10, 4))
    titles = ["Image + GT + Pred", "Mask GT", "Mask Pred"]
    images = [overlaid, gt, pred]

    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        cmap = None if i == 0 else "gray"
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{index}.png"))
    plt.close()

def visualize_first_samples(model, dataset, save_dir):
    """
    Sauvegarde les 5 premières visualisations (image, GT, prédiction) pour un dataset donné.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= 5:
                break

            data = batch['image'].to(device)
            target = batch['label'].to(device)
            dct_coef = batch['rgb'].long().to(device)
            qs = batch['q'].unsqueeze(1).to(device)

            pred = model(data, dct_coef, qs)
            prob = torch.softmax(pred, dim=1)[:, 1, :, :]
            pred_mask = (prob > 0.5).long()
            targt = target.squeeze(1)

            img = data[0].cpu()
            gt = targt[0].cpu()
            pd = pred_mask[0].cpu()

            save_visualization(img, gt, pd, index=idx, output_dir=save_dir)

# ============================== MAIN SCRIPT ==============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = seg_dtd("", 2).to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("Weights/dtd_doctamper.pth", map_location=device)['state_dict'])

datasets_info = {
    "TestingSet":   ("./DocTamperV1-TestingSet", "./pks/DocTamperV1-TestingSet_90.pk", False, range(10)),
    "FCD":           ("./DocTamperV1-FCD", "./pks/DocTamperV1-FCD_90.pk", False, range(10)),
    "SCD":           ("./DocTamperV1-SCD", "./pks/DocTamperV1-SCD_90.pk", False, range(10)),
}
T=8192

qt_path = './pks/qt_table.pk'
save_dir = "results_eval_dtd_org2"
os.makedirs(save_dir, exist_ok=True)
import pandas as pd

qualities = list(range(1, 101, 1))  # 85 à 100 inclus
all_results = []

for name, (lmdb_path, record_path, is_train, subset_range) in datasets_info.items():
    print(f"\n=== Dataset : {name} ===")
    dataset_dir = os.path.join(save_dir, name)
    os.makedirs(dataset_dir, exist_ok=True) 
    for q in qualities:
        print(f"[{name}] Évaluation à JPEG Quality = {q}")
   
        dataset_q = TamperDatasetCLTD(
            lmdb_path, 
            record_path, 
        
            T=T, 
            is_train=False, 
            fixed_quality=q  # qualité fixée ici
        )
        subset = Subset(dataset_q, list(subset_range))

        # === Évaluation simple
        loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=4)
        model.eval()

        iou = IOUMetric(2)
        precisions, recalls = [], []

        with torch.no_grad():
            for batch in loader:
                data = batch['image'].to(device)
                target = batch['label'].to(device)
                dct_coef = batch['rgb'].long().to(device)
                qs = batch['q'].unsqueeze(1).to(device)

                pred = model(data, dct_coef, qs)
                prob = torch.softmax(pred, dim=1)[:, 1, :, :]
                pred_mask = (prob > 0.5).long()
                targt = target.squeeze(1)

                matched = (pred_mask * targt).sum((1, 2))
                pred_sum = pred_mask.sum((1, 2))
                target_sum = targt.sum((1, 2))
                precisions.append((matched / (pred_sum + 1e-8)).mean().item())
                recalls.append((matched / (target_sum + 1e-8)).mean().item())
                iou.add_batch(pred_mask.cpu().numpy(), targt.cpu().numpy())

        acc, acc_cls, iou_values, mean_iou, fwavacc = iou.evaluate()
        pre_mean = np.mean(precisions)
        rec_mean = np.mean(recalls)
        f1 = (2 * pre_mean * rec_mean) / (pre_mean + rec_mean + 1e-8)
        vis_dir = os.path.join(dataset_dir, f"q{q:03d}")
        visualize_first_samples(model, subset, vis_dir)

        # Ajout au tableau global
        all_results.append({
            "Dataset": name,
            "JPEG Quality": q,
            "IoU Class 1": iou_values[1],
            "Precision": pre_mean,
            "Recall": rec_mean,
            "F1 Score": f1
        })

# ---- Création du DataFrame global
df_all = pd.DataFrame(all_results)
df_all.to_csv(os.path.join(save_dir, "all_metrics_vs_quality.csv"), index=False)

# ---- Tracés
metrics = ["IoU Class 1", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for name in df_all["Dataset"].unique():
        df_subset = df_all[df_all["Dataset"] == name]
        plt.plot(df_subset["JPEG Quality"], df_subset[metric], label=name)
    plt.xlabel("JPEG Quality")
    plt.ylabel(metric)
    plt.title(f"{metric} vs JPEG Quality across datasets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric.lower().replace(' ', '_')}_vs_quality.png"))
    plt.close()

for name, (lmdb_path, record_path, is_train, subset_range) in datasets_info.items():
    dataset = TamperDatasetCLTD(lmdb_path, record_path, T=T, is_train=is_train)
    subset = Subset(dataset, list(subset_range))
    eval_net_dtd(model, subset)
