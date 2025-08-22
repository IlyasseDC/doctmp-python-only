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
# Dataset Images/Labels + JPEG features
# -------------------------
class TamperDatasetImages(Dataset):
    def __init__(self, img_dir, lbl_dir, quality=75):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".png")])
        assert len(self.img_files) == len(self.lbl_files), "Images ≠ Labels"
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.quality = quality

        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        lbl_path = os.path.join(self.lbl_dir, self.lbl_files[idx])

        img = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("L")

        w, h = img.size
        patches = []

        for top in range(0, h, 512):
            for left in range(0, w, 512):
                right = min(left + 512, w)
                bottom = min(top + 512, h)

                # --- RGB PATCH ---
                patch_img = img.crop((left, top, right, bottom))
                pad_img = Image.new('RGB', (512, 512), (0, 0, 0))
                pad_img.paste(patch_img, (0, 0))

                # --- LABEL PATCH ---
                patch_lbl = label.crop((left, top, right, bottom))
                pad_lbl = Image.new('L', (512, 512), 0)
                pad_lbl.paste(patch_lbl, (0, 0))

                # --- JPEG features ---
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    pad_img.save(tmp.name, "JPEG", quality=self.quality)
                    jpg = jpegio.read(tmp.name)
                    dct = np.ascontiguousarray(jpg.coef_arrays[0])
                    qtb = torch.from_numpy(np.ascontiguousarray(jpg.quant_tables[0])).long().clone().unsqueeze(0)

                dct_t = torch.from_numpy(np.clip(np.abs(dct), 0, 20).astype(np.int64))
                img_t = self.to_tensor(pad_img)
                lbl_np = np.array(pad_lbl, dtype=np.uint8)
                lbl_np = (lbl_np > 127).astype(np.int64)  # binarisation 0/1
                lbl_t = torch.from_numpy(lbl_np)
                patches.append({
                    "image": img_t,
                    "label": lbl_t,
                    "rgb": dct_t,
                    "q": qtb
                })

        return patches



# -------------------------
# Collate avec padding dynamique
# -------------------------
def patch_collate(batch):
    """
    batch = [list(dicts_pour_img1), list(dicts_pour_img2), ...]
    On aplatit tout en une seule liste de patchs
    """
    flat = [p for b in batch for p in b]  # concatène toutes les listes de patchs
    flat = flat[:6]
    images = torch.stack([p["image"] for p in flat])
    labels = torch.stack([p["label"] for p in flat])
    rgbs   = torch.stack([p["rgb"] for p in flat])
    qtabs  = torch.stack([p["q"] for p in flat])

    return {"image": images, "label": labels, "rgb": rgbs, "q": qtabs}



# -------------------------
# Entraînement
# -------------------------
def train_dtd_images(param):
    epochs = param['epochs']
    batch_size = param['batch_size']
    save_ckpt_dir = param['save_ckpt_dir']
    save_log_dir = param['save_log_dir']
    data_root = param['data_root']
    quality = param.get('quality', 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(save_log_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_log_dir, f"train_dtd_img_{time.strftime('%Y%m%d_%H%M%S')}.log"))

    logger.info("Initializing model (via seg_dtd)...")
    model = seg_dtd("", n_class=2).to(device)

    if param.get('load_ckpt') and os.path.exists(param['load_ckpt']):
        logger.info(f"Loading full checkpoint from {param['load_ckpt']}")
        checkpoint = torch.load(param['load_ckpt'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No pre-trained checkpoint loaded.")
    # Freeze Swin backbone
    # Freeze VPH entièrement
    for name, param in model.named_parameters():
        if "vph" in name:
            param.requires_grad = False

    # Freeze Swin sauf le dernier stage + norm
    for name, param in model.named_parameters():
        if "swin" in name:
            param.requires_grad = False
            if "layers.3" in name or "norm" in name:  # stage 4 du swin + normalisation
                param.requires_grad = True


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    # Charger dataset Images
    logger.info("Preparing dataset Images/Labels...")
    full_dataset = TamperDatasetImages(
        os.path.join(data_root, "Images"),
        os.path.join(data_root, "Labels"),
        quality=quality
    )

    # Split fixe : 5500 pour le train, le reste pour le test
    train_size = 5500
    test_size = len(full_dataset) - train_size

    if train_size > len(full_dataset):
        raise ValueError(f"Le dataset ({len(full_dataset)} images) est trop petit pour prendre 5500 échantillons en train.")

    train_set, val_set = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=8, collate_fn=patch_collate)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=patch_collate)


    best_iou = 0
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            image = batch['image'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            dct   = batch['rgb'].long().to(device, non_blocking=True)
            qtb   = batch['q'].long().to(device, non_blocking=True)

            with autocast():
                output = model(image, dct, qtb)
                ce = 5 * ce_loss(output, label)
                lovasz = lovasz_loss(output.float(), label)
                loss = ce + lovasz

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + 1)

            train_loss_meter.update(loss.item())

        logger.info(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss_meter.avg:.4f}")

        # Validation
        model.eval()
        iou = IOUMetric(2)
        precisions, recalls = [], []

        with torch.no_grad():
            for batch in val_loader:
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

        acc, acc_cls, iou_vals, mean_iou, fwavacc = iou.evaluate()
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        logger.info(f"[Validation] IoU: {iou_vals}, F1: {f1:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Sauvegarde checkpoints
        ckpt_path = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        if iou_vals[1] > best_iou:
            best_iou = iou_vals[1]
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(save_ckpt_dir, 'checkpoint-best.pth'))
            logger.info(f"Best model saved at epoch {epoch+1} with IoU class 1 = {best_iou:.4f}")


if __name__ == "__main__":
    params = {
        'epochs': 30,
        'batch_size': 8,
        'save_ckpt_dir': './checkpoints_img',
        'save_log_dir': './logs_img',
        'data_root': './Internal_TrainingSet_',
        'quality': 100,
        'load_ckpt': './Weights/dtd_doctamper.pth'
    }
    train_dtd_images(params)
