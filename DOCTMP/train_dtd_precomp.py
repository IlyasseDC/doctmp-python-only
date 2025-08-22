# train_dtd_images.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import jpegio

from models.dtd import seg_dtd
from models.losses import LovaszLoss, SoftCrossEntropyLoss
from utils import AverageMeter, get_logger
from metrics import IOUMetric
from dataset_cltd import TamperDatasetCLTD
from models.dtd import *
# train_dtd_images.py
import os
import time
import numpy as np
from PIL import Image
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision
import jpegio

from models.dtd import seg_dtd
from models.losses import LovaszLoss, SoftCrossEntropyLoss
from utils import AverageMeter, get_logger
from metrics import IOUMetric

# -------------------------
# Dataset pré-calculé
# -------------------------
class TamperDatasetPrecomp(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)

        # Aplatir toutes les listes de patches
        self.samples = [p for patches in self.data for p in patches]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image": torch.from_numpy(s["image"]).float(),
            "label": torch.from_numpy(s["label"]).long(),
            "rgb": torch.from_numpy(s["dct"]).long(),
            "q": torch.from_numpy(s["qtb"]).long()
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    rgbs   = torch.stack([b["rgb"] for b in batch])
    qtabs  = torch.stack([b["q"] for b in batch])
    return {"image": images, "label": labels, "rgb": rgbs, "q": qtabs}


# -------------------------
# Entraînement
# -------------------------
def train_dtd_precomp(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(params['save_ckpt_dir'], exist_ok=True)
    os.makedirs(params['save_log_dir'], exist_ok=True)
    logger = get_logger(os.path.join(params['save_log_dir'],
                                     f"train_dtd_precomp_{time.strftime('%Y%m%d_%H%M%S')}.log"))

    # Modèle
    logger.info("Initializing model (via seg_dtd)...")
    model = seg_dtd("", n_class=2).to(device)

    if params.get('load_ckpt') and os.path.exists(params['load_ckpt']):
        logger.info(f"Loading full checkpoint from {params['load_ckpt']}")
        checkpoint = torch.load(params['load_ckpt'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("Checkpoint loaded.")

    # Débloquer dernier stage de Swin
    for name, p in model.named_parameters():
        if "swin.layers.3" in name:  # dernier stage swin
            p.requires_grad = True
        elif "swin" in name:
            p.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    # Dataset
    dataset = TamperDatasetPrecomp(params['pkl_file'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn)

    best_iou = 0
    for epoch in range(params['epochs']):
        model.train()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            dct   = batch['rgb'].to(device)
            qtb   = batch['q'].to(device)

            with autocast():
                output = model(image, dct, qtb)
                loss = 5 * ce_loss(output, label) + lovasz_loss(output.float(), label)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + 1)

            train_loss_meter.update(loss.item())

        logger.info(f"[Epoch {epoch+1}/{params['epochs']}] Train Loss: {train_loss_meter.avg:.4f}")

        # Validation
        model.eval()
        iou = IOUMetric(2)
        precisions, recalls = [], []

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                label = batch['label'].to(device)
                dct   = batch['rgb'].to(device)
                qtb   = batch['q'].to(device)

                with autocast():
                    output = model(image, dct, qtb)

                pred = output.argmax(1)
                targt = label
                matched = (pred * targt).sum((1, 2))
                precisions.append((matched / (pred.sum((1, 2)) + 1e-8)).mean().item())
                recalls.append((matched / (targt.sum((1, 2)) + 1e-8)).mean().item())
                iou.add_batch(pred.cpu().numpy(), label.cpu().numpy())

        acc, acc_cls, iou_vals, mean_iou, fwavacc = iou.evaluate()
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        logger.info(f"[Validation] IoU: {iou_vals}, F1: {f1:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Sauvegarde checkpoints
        ckpt_path = os.path.join(params['save_ckpt_dir'], 'checkpoint-latest.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        if iou_vals[1] > best_iou:
            best_iou = iou_vals[1]
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(params['save_ckpt_dir'], 'checkpoint-best.pth'))
            logger.info(f"Best model saved at epoch {epoch+1} with IoU class 1 = {best_iou:.4f}")


if __name__ == "__main__":
    params = {
        'epochs': 20,
        'batch_size': 4,
        'save_ckpt_dir': './checkpoints_precomp',
        'save_log_dir': './logs_precomp',
        'pkl_file': '/home/ilyassechaouki/DOCTMP/Internal_TrainingSet/precomputed.pkl',
        'load_ckpt': './Weights/dtd_doctamper.pth'
    }
    train_dtd_precomp(params)
