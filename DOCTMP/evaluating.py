# eval_dtd_images.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from models.dtd import seg_dtd
from models.losses import LovaszLoss, SoftCrossEntropyLoss
from utils import get_logger
from metrics import IOUMetric

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
        self.quality = int(quality)

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

        # Image & label d'origine
        img_rgb = Image.open(img_path).convert("RGB")
        label   = Image.open(lbl_path).convert("L")

        w, h = img_rgb.size
        patches = []

        for top in range(0, h, 512):
            for left in range(0, w, 512):
                right  = min(left + 512, w)
                bottom = min(top + 512, h)

                # --- PATCH RGB 512x512 (avec padding si bord) ---
                patch_img = img_rgb.crop((left, top, right, bottom))
                pad_img = Image.new('RGB', (512, 512), (0, 0, 0))
                pad_img.paste(patch_img, (0, 0))

                # --- PATCH LABEL 512x512 ---
                patch_lbl = label.crop((left, top, right, bottom))
                pad_lbl = Image.new('L', (512, 512), 0)
                pad_lbl.paste(patch_lbl, (0, 0))

                # --- PIPELINE "papier" : L -> JPEG(quality) -> jpegio(DCT/Q) ---
                # Convertit le patch en niveaux de gris puis ré-encode en JPEG
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    # 1) gris
                    pad_img_gray = pad_img.convert("L")
                    # 2) JPEG contrôlé
                    pad_img_gray.save(tmp.name, "JPEG", quality=self.quality)
                    # 3) DCT & Q-table (canal Y)
                    jpg = jpegio.read(tmp.name)
                    dct = np.ascontiguousarray(jpg.coef_arrays[0])  # Y only
                    qtb = torch.from_numpy(
                        np.ascontiguousarray(jpg.quant_tables[0])
                    ).long().clone().unsqueeze(0)  # (1,8,8)
                    # 4) image réseau = JPEG gris relu et reconverti en RGB
                    comp_rgb = Image.open(tmp.name).convert("RGB").copy()
                # Nettoyage du fichier temp
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

                # --- Tenseurs sortants ---
                dct_t = torch.from_numpy(np.clip(np.abs(dct), 0, 20).astype(np.int64))
                img_t = self.to_tensor(comp_rgb)

                lbl_np = np.array(pad_lbl, dtype=np.uint8)
                lbl_np = (lbl_np > 127).astype(np.int64)  # binarisation 0/1
                lbl_t = torch.from_numpy(lbl_np)

                patches.append({
                    "image": img_t,   # (3,512,512) float32 normalisé
                    "label": lbl_t,   # (512,512)   int64 {0,1}
                    "rgb": dct_t,     # (512,512)   int64 |DCT| clippé à 20
                    "q": qtb          # (1,8,8)     int64 Q-table réelle du JPEG gris
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
    # Freeze VPH 
    for name, param in model.named_parameters():
        if "vph" in name:
            param.requires_grad = False

    for name, param in model.named_parameters():
        if ("fph" in name 
            or "decoder" in name 
            or "fusion" in name 
            or "FU" in name 
            or "segmentation_head" in name):
            param.requires_grad = True

    for name, param in model.named_parameters():
        if "swin" in name:
            param.requires_grad = False
            if "layers.3" in name or "norm" in name:
                param.requires_grad = True

    # LR différenciés
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "swin" in n], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "fph" in n], "lr": 3e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and (
            "decoder" in n or "fusion" in n or "FU" in n or "segmentation_head" in n
        )], "lr": 3e-4},
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler()
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    # Debug print
    print("\n=== Paramètres entraînables ===")
    lr_dict = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        for p in group["params"]:
            lr_dict[id(p)] = lr

    for name, param in model.named_parameters():
        if param.requires_grad:
            lr = lr_dict.get(id(param), None)
            print(f"[TRAINABLE] {name:50s} | lr={lr}")
        else:
            print(f"[FROZEN]   {name:50s}")
    print("================================\n")


    # Charger dataset Images
    # Charger dataset Images/Labels + JPEG features
    logger.info("Preparing datasets FCD+SCD (train/test)...")

    train_dataset = TamperDatasetImages(
        os.path.join(data_root, "train", "Images"),
        os.path.join(data_root, "train", "Labels"),
        quality=quality
    )

    val_dataset = TamperDatasetImages(
        os.path.join(data_root, "test", "Images"),
        os.path.join(data_root, "test", "Labels"),
        quality=quality
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, collate_fn=patch_collate
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=patch_collate
    )


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
            train_loss_meter.update(loss.item())
        scheduler.step()



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
def evaluate(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(params['save_log_dir'], exist_ok=True)
    logger = get_logger(os.path.join(params['save_log_dir'], "eval_dtd_img.log"))

    # -------------------------
    # Chargement du modèle
    # -------------------------
    logger.info("Loading model seg_dtd...")
    model = seg_dtd("", n_class=2).to(device)

    assert os.path.exists(params['load_ckpt']), f"Checkpoint {params['load_ckpt']} not found"
    checkpoint = torch.load(params['load_ckpt'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(f"Checkpoint {params['load_ckpt']} loaded (epoch={checkpoint.get('epoch','?')}).")

    model.eval()

    # -------------------------
    # Dataset test
    # -------------------------
    logger.info("Preparing test dataset...")
    val_dataset = TamperDatasetImages(
        os.path.join(params['data_root'], "test", "Images"),
        os.path.join(params['data_root'], "test", "Labels"),
        quality=params.get('quality', 100)
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=patch_collate
    )

    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    iou_metric = IOUMetric(2)
    precisions, recalls = [], []
    total_loss = 0.0

    # -------------------------
    # Préparation dossier de sauvegarde
    # -------------------------
    save_vis_dir = os.path.join(params['save_log_dir'], "outputs")
    os.makedirs(save_vis_dir, exist_ok=True)

    save_count = 0

    # -------------------------
    # Boucle d'évaluation
    # -------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            dct   = batch['rgb'].long().to(device)
            qtb   = batch['q'].long().to(device)

            with autocast():
                output = model(image, dct, qtb)
                loss = 5 * ce_loss(output, label) + lovasz_loss(output, label)

            total_loss += loss.item()
            pred = output.argmax(1)
            targt = label.squeeze(1)

            # précision / rappel par patch
            matched = (pred * targt).sum((1, 2))
            pred_sum = pred.sum((1, 2))
            target_sum = targt.sum((1, 2))
            precisions.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / (target_sum + 1e-8)).mean().item())

            # IoU global
            iou_metric.add_batch(pred.cpu().numpy(), label.cpu().numpy())

            # -------------------------
            # Sauvegarde des 100 premières sorties
            # -------------------------
            if save_count < 100:
                # On prend seulement le premier patch du batch (car batch=1)
                img_np = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                lbl_np = targt[0].cpu().numpy().astype(np.uint8) * 255
                pred_np = pred[0].cpu().numpy().astype(np.uint8) * 255

                # Sauvegarde images
                Image.fromarray(img_np).save(os.path.join(save_vis_dir, f"img_{save_count:03d}.png"))
                Image.fromarray(lbl_np).save(os.path.join(save_vis_dir, f"gt_{save_count:03d}.png"))
                Image.fromarray(pred_np).save(os.path.join(save_vis_dir, f"pred_{save_count:03d}.png"))

                save_count += 1

    acc, acc_cls, iou_vals, mean_iou, fwavacc = iou_metric.evaluate()
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info("=== Evaluation Results ===")
    logger.info(f"Loss: {total_loss/len(val_loader):.4f}")
    logger.info(f"IoU per class: {iou_vals}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info("==========================")

    return {
        "loss": total_loss/len(val_loader),
        "iou_class": iou_vals,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



if __name__ == "__main__":
    params = {
        'save_log_dir': './logs_eval',
        'data_root': './FCDSCD_exported',
        'quality': 100,
        'load_ckpt': './checkpointsfinetune/checkpoint-best.pth'
    }
    results = evaluate(params)
    print(results)
