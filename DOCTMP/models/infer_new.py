# infer_sroie_from_jpeg_dir.py
# Inférence "mode SROIE" (détection) sur un dossier JPEG externe.
# Arborescence attendue :
#   data_root/
#     ├── Images/*.jpg (ou .jpeg)
#     └── Labels/*.png (masks binaires 0/255)
#
# Sorties :
#   out/
#     ├── heatmaps/   (heatmap 0..255)
#     ├── masks/      (mask binaire après seuillage & morpho)
#     ├── overlays/   (image + heatmap + contours)
#     ├── pred_txts/  (boîtes préd. IC15)
#     └── gt_txts/    (boîtes GT   IC15)
#
# Métriques (détection) : Precision / Recall / H-mean (IoU des boîtes ≥ iou_thr)

import os
import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset
import torchvision
import jpegio

from models.dtd import seg_dtd

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
# =========================
# Utils: boxes, IoU, éval
# =========================
def find_boxes_from_mask(mask_u8, min_area=50):
    """
    mask_u8 : (H,W) uint8 {0,255}
    Retourne une liste de boîtes axis-aligned [x1,y1,x2,y2].
    """
    boxes = []
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
    return boxes


def rect_iou(a, b):
    """
    IoU entre deux boîtes axis-aligned : a=[x1,y1,x2,y2], b idem.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    Appariement glouton par IoU.
    Retourne tp, fp, fn et la liste des matches (g,p,IoU).
    """
    matched_g = set()
    matched_p = set()
    matches = []

    # Pré-calcul des IoU
    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou_mat[i, j] = rect_iou(g, p)

    # Glouton : boucle tant qu'on trouve un pair >= thr
    while True:
        max_iou = -1.0
        max_pair = (-1, -1)
        for i in range(len(gt_boxes)):
            if i in matched_g:
                continue
            for j in range(len(pred_boxes)):
                if j in matched_p:
                    continue
                if iou_mat[i, j] >= iou_thr and iou_mat[i, j] > max_iou:
                    max_iou = iou_mat[i, j]
                    max_pair = (i, j)
        if max_pair[0] == -1:
            break
        i, j = max_pair
        matched_g.add(i)
        matched_p.add(j)
        matches.append((i, j, float(iou_mat[i, j])))

    tp = len(matched_p)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def write_ic15_txt(txt_path, boxes):
    """
    Écrit des boîtes au format IC15 :
    x1,y1,x2,y2,x3,y3,x4,y4,transcription
    On convertit chaque rectangle axis-aligned en 4 points (sens horaire).
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            line = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},###\n"
            f.write(line)


def overlay_prob_heatmap(img_rgb, prob, alpha=0.45):
    """
    img_rgb: (H,W,3) uint8
    prob   : (H,W) float32 in [0,1]
    return : overlay uint8 (H,W,3)
    """
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    mask_u8 = (prob * 255.0 + 0.5).astype(np.uint8)               # [0..255]
    heatmap_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)    # (H,W,3) BGR
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


# =========================
# Dataset JPEG → patches
# =========================
class TamperDatasetJPEG(Dataset):
    def __init__(self, data_root, mode='test', gray_quality=95):
        self.images_dir = Path(data_root) / "Images"
        self.labels_dir = Path(data_root) / "Labels"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self.files = sorted([p for p in self.images_dir.rglob("*")
                             if p.suffix.lower() in (".jpg", ".jpeg")])
        self.max_nums = len(self.files)
        self.mode = mode
        self.gray_quality = int(gray_quality)

        # Normalisation identique à tes autres scripts
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        img_path = self.files[index]
        im_rgb = Image.open(img_path).convert('RGB')
        W, H = im_rgb.size  # dimensions d'origine

        # GT (binaire 0/255 → 0/1)
        gt_path = (self.labels_dir / (img_path.stem + ".png"))
        if not gt_path.exists():
            raise FileNotFoundError(f"GT missing for image: {img_path.name} -> {gt_path}")
        gt_mask = np.array(Image.open(gt_path).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # >>> Pipeline "papier": convertir en L puis (ré)encoder en JPEG, puis lire DCT/qtb <<<
        im_gray = im_rgb.convert("L")

        # Optionnel: padding aux multiples de 8 AVANT encodage (robustesse MCU)
        pad_w8 = (8 - (W % 8)) % 8
        pad_h8 = (8 - (H % 8)) % 8
        if pad_w8 > 0 or pad_h8 > 0:
            im_gray_enc = ImageOps.expand(im_gray, (0, 0, pad_w8, pad_h8), fill=0)
        else:
            im_gray_enc = im_gray

        # Toujours ré-encoder (comme l'original), à qualité contrôlée
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            im_gray_enc.save(tmp.name, "JPEG", quality=self.gray_quality)
            jpg = jpegio.read(tmp.name)

        dct_full = jpg.coef_arrays[0].copy()
        qtb = torch.LongTensor(jpg.quant_tables[0])  # (8,8) long attendu par le modèle

        patches = []
        for top in range(0, H, 512):
            for left in range(0, W, 512):
                right = min(left + 512, W)
                bottom = min(top + 512, H)

                # PATCH RGB (entrée réseau)
                patch = im_rgb.crop((left, top, right, bottom))
                if patch.size != (512, 512):
                    pad_img = Image.new('RGB', (512, 512), (0, 0, 0))
                    pad_img.paste(patch, (0, 0))
                    patch = pad_img

                # PATCH DCT : sectionner dans dct_full (dimension >= image si padding 8 appliqué)
                tile_h = bottom - top
                tile_w = right - left
                dct_tile = np.zeros((tile_h, tile_w), dtype=dct_full.dtype)
                dct_tile[:tile_h, :tile_w] = dct_full[top:bottom, left:right]

                # Forcer multiples de 8
                pad_h8_t = (8 - (dct_tile.shape[0] % 8)) % 8
                pad_w8_t = (8 - (dct_tile.shape[1] % 8)) % 8
                if pad_h8_t > 0 or pad_w8_t > 0:
                    pad_dct = np.zeros((dct_tile.shape[0] + pad_h8_t,
                                        dct_tile.shape[1] + pad_w8_t),
                                       dtype=dct_full.dtype)
                    pad_dct[:dct_tile.shape[0], :dct_tile.shape[1]] = dct_tile
                    dct_tile = pad_dct

                # Forcer taille finale 512×512 côté DCT
                if dct_tile.shape != (512, 512):
                    pad_dct_final = np.zeros((512, 512), dtype=dct_full.dtype)
                    pad_dct_final[:dct_tile.shape[0], :dct_tile.shape[1]] = dct_tile
                    dct_tile = pad_dct_final

                patches.append({
                    'image': self.toctsr(patch),              # (3,512,512) float32
                    'rgb': np.clip(np.abs(dct_tile), 0, 20), # (512,512) int-like
                    'q': qtb,                                 # (8,8) long
                    'pos': (top, left)
                })

        return {
            'patches': patches,
            'path': str(img_path),
            'orig_size': (H, W),
            'gt_mask': gt_mask.astype(np.uint8),
            'orig_img': np.array(im_rgb)
        }



# =========================
# Inférence + post-proc SROIE
# =========================
def run_sroie(
    model, dataset, pth, out_dir,
    device='cuda',
    prob_thresh=0.88,   # ~ 224/255
    kernel=5,
    min_area=50,
    iou_thr=0.5
):
    out_dir = Path(out_dir)
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_txts").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_txts").mkdir(parents=True, exist_ok=True)

    # Poids
    ckpt = torch.load(pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Inférence SROIE"):
            patches = sample['patches']
            H, W = sample['orig_size']
            img_path = sample['path']
            gt_mask = sample['gt_mask']
            img_np  = sample['orig_img']

            # 1) Accumuler probas classe 1
            full_prob = np.zeros((H, W), dtype=np.float32)
            full_cnt  = np.zeros((H, W), dtype=np.float32)

            for p in patches:
                data = p['image'].unsqueeze(0).to(device)                             # (1,3,512,512)
                dct_coef = torch.as_tensor(p['rgb'], dtype=torch.long, device=device) \
                               .unsqueeze(0)                                          # (1,512,512)
                qs = p['q'].to(device=device, dtype=torch.long).unsqueeze(0).unsqueeze(1)  # (1,1,8,8)

                logits = model(data, dct_coef, qs)                                    # (1,2,512,512)
                prob1 = torch.softmax(logits, dim=1)[:, 1].squeeze(0).cpu().numpy()  # (512,512)

                top, left = p['pos']
                h_patch = min(512, H - top)
                w_patch = min(512, W - left)

                full_prob[top:top+h_patch, left:left+w_patch] += prob1[:h_patch, :w_patch]
                full_cnt [top:top+h_patch, left:left+w_patch] += 1.0

            full_cnt[full_cnt == 0] = 1.0
            full_prob /= full_cnt

            # 2) Post-proc : seuillage + morpho
            th = float(prob_thresh)
            mask_bin = (full_prob >= th).astype(np.uint8) * 255

            if kernel > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
                # ouverture puis fermeture (classique pour bruit → régions compactes)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)

            # 3) Boîtes préd & GT
            pred_boxes = find_boxes_from_mask(mask_bin, min_area=min_area)

            gt_mask_u8 = (gt_mask * 255).astype(np.uint8)
            gt_boxes = find_boxes_from_mask(gt_mask_u8, min_area=min_area)

            # 4) Écriture IC15 (txt)
            base = os.path.splitext(os.path.basename(img_path))[0]
            write_ic15_txt(out_dir / "pred_txts" / f"{base}.txt", pred_boxes)
            write_ic15_txt(out_dir / "gt_txts"   / f"gt_{base}.txt", gt_boxes)

            # 5) Overlays & heatmaps
            # heatmap 0..255
            heat = (np.clip(full_prob, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(heat).save(out_dir / "heatmaps" / f"{base}.png")
            Image.fromarray(mask_bin).save(out_dir / "masks" / f"{base}.png")

            overlay = overlay_prob_heatmap(img_np, full_prob, alpha=0.45)
            # Dessiner les boîtes préd (vert) et GT (bleu)
            for x1, y1, x2, y2 in pred_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x1, y1, x2, y2 in gt_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), 2)
            Image.fromarray(overlay).save(out_dir / "overlays" / f"{base}.png")

            # 6) Métriques détection (IoU des boîtes)
            tp, fp, fn, _ = match_boxes(gt_boxes, pred_boxes, iou_thr=iou_thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Résumé global
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    hmean     = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n=== Résultats globaux (détection, IoU boîtes ≥ {:.2f}) ===".format(iou_thr))
    print("Precision : {:.4f}".format(precision))
    print("Recall    : {:.4f}".format(recall))
    print("H-mean    : {:.4f}".format(hmean))


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True,
                    help='Racine du dataset avec sous-dossiers Images/ et Labels/')
    ap.add_argument('--pth', type=str, required=True, help='Checkpoint .pth de seg_dtd')
    ap.add_argument('--out', type=str, default='sroie_outputs', help='Dossier de sortie')
    ap.add_argument('--n_test', type=int, default=0,
                    help='Si >0, limite aux n dernières images (0 = toutes)')
    ap.add_argument('--prob_thresh', type=float, default=0.88,
                    help='Seuil probabilité pour binaire (ex. 0.88≈224/255)')
    ap.add_argument('--kernel', type=int, default=5, help='Taille kernel morpho (0 = pas de morpho)')
    ap.add_argument('--min_area', type=int, default=50, help='Aire minimale d’une région')
    ap.add_argument('--iou_thr', type=float, default=0.5, help='Seuil IoU pour le matching boîtes')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = seg_dtd('', 2).to(device)
    #model = torch.nn.DataParallel(model)
    full_dataset = TamperDatasetJPEG(args.data_root, mode='test')
    if args.n_test and args.n_test > 0:
        n = min(args.n_test, len(full_dataset))
        dataset = Subset(full_dataset, range(len(full_dataset) - n, len(full_dataset)))
    else:
        dataset = full_dataset

    run_sroie(
        model=model,
        dataset=dataset,
        pth=args.pth,
        out_dir=args.out,
        device=device,
        prob_thresh=args.prob_thresh,
        kernel=args.kernel,
        min_area=args.min_area,
        iou_thr=args.iou_thr
    )


if __name__ == "__main__":
    main()
