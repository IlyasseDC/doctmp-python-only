import os
import cv2
import torch
import jpegio
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from albumentations.pytorch import ToTensorV2
import torchvision
from PIL import Image
import argparse
import os
import cv2
import lmdb
import torch
import jpegio
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
from torch.cuda.amp import autocast, GradScaler
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
from models.dtd import seg_dtd  # ton modèle DTD
from pathlib import Path
from PIL import ImageOps
import tempfile


# --------------------
# Dataset JPEG
# --------------------
import os
import cv2
import torch
import jpegio
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from PIL import Image, ImageOps
import argparse
import tempfile
from pathlib import Path
from models.dtd import seg_dtd  # ton modèle DTD
from metrics import IOUMetric
from finetune_dtd import TamperDatasetImages, patch_collate


def evaluate(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Charger modèle ---
    model = seg_dtd("", n_class=2).to(device)
    ckpt = torch.load(params['load_ckpt'], map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # --- Dataset & DataLoader ---
    val_dataset = TamperDatasetImages(
        os.path.join(params['data_root'], "Images"),
        os.path.join(params['data_root'], "Labels"),
        quality=params.get("quality", 100)
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=patch_collate
    )

    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    # --- Métriques globales ---
    iou = IOUMetric(2)
    precisions, recalls = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            dct   = batch['rgb'].long().to(device)
            qtb   = batch['q'].long().to(device)

            with autocast():
                output = model(image, dct, qtb)
                loss = 5 * ce_loss(output, label) + lovasz_loss(output, label)

            pred = output.argmax(1)
            targt = label.squeeze(1)

            matched = (pred * targt).sum((1, 2))
            pred_sum = pred.sum((1, 2))
            target_sum = targt.sum((1, 2))

            precisions.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / (target_sum + 1e-8)).mean().item())

            iou.add_batch(pred.cpu().numpy(), label.cpu().numpy())

    # --- Résultats ---
    acc, acc_cls, iou_vals, mean_iou, fwavacc = iou.evaluate()
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision+recall)>0 else 0.0

    print("=== Résultats évaluation (patch-level, identique à train) ===")
    print(f"IoU classes   : {iou_vals}")
    print(f"Mean IoU      : {mean_iou:.4f}")
    print(f"F1-score moy. : {f1:.4f}")
    print(f"Précision moy.: {precision:.4f}")
    print(f"Rappel moy.   : {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Dossier contenant Images/ et Labels/')
    parser.add_argument('--pth', type=str, required=True, help='Checkpoint .pth du modèle seg_dtd')
    parser.add_argument('--quality', type=int, default=100, help='Qualité JPEG pour recompression')
    args = parser.parse_args()

    params = {
        'data_root': args.data_root,
        'load_ckpt': args.pth,
        'quality': args.quality
    }
    evaluate(params)
