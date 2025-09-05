
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

# --------------------
# Dataset JPEG découpé en patchs 2048×2048
# --------------------
class TamperDatasetJPEG(Dataset):
    def __init__(self, root, mode):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")])
        self.max_nums = len(self.files)
        self.mode = mode
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        img_path = self.files[index]
        im = Image.open(img_path).convert('RGB')
        w, h = im.size

        # --- Padding global de l'image à un multiple de 8 avant lecture JPEG ---
        pad_w8 = (8 - (w % 8)) % 8
        pad_h8 = (8 - (h % 8)) % 8
        if pad_w8 > 0 or pad_h8 > 0:
            im = ImageOps.expand(im, (0, 0, pad_w8, pad_h8), fill=(0, 0, 0))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                im.save(tmp.name, "JPEG", quality=100)
                jpg = jpegio.read(tmp.name)
        else:
            jpg = jpegio.read(str(img_path))

        # DCT et Q-table
        dct_full = jpg.coef_arrays[0].copy()
        qtb = torch.LongTensor(jpg.quant_tables[0])
        patches = []

        # --- Découpage en patchs 2048×2048 ---
        for top in range(0, h, 2048):
            for left in range(0, w, 2048):
                right = min(left + 2048, w)
                bottom = min(top + 2048, h)

                # ----- PATCH RGB -----
                patch = im.crop((left, top, right, bottom))
                ph, pw = patch.size[1], patch.size[0]
                if pw < 2048 or ph < 2048:
                    pad_img = Image.new('RGB', (2048, 2048), (0, 0, 0))
                    pad_img.paste(patch, (0, 0))
                    patch = pad_img

                # ----- PATCH DCT -----
                tile_h = bottom - top
                tile_w = right - left

                dct_tile = np.zeros((tile_h, tile_w), dtype=dct_full.dtype)
                dct_tile[:tile_h, :tile_w] = dct_full[top:bottom, left:right]

                # Forcer à multiple de 8
                pad_h8 = (8 - (dct_tile.shape[0] % 8)) % 8
                pad_w8 = (8 - (dct_tile.shape[1] % 8)) % 8
                if pad_h8 > 0 or pad_w8 > 0:
                    pad_dct = np.zeros((dct_tile.shape[0] + pad_h8, dct_tile.shape[1] + pad_w8), dtype=dct_full.dtype)
                    pad_dct[:dct_tile.shape[0], :dct_tile.shape[1]] = dct_tile
                    dct_tile = pad_dct

                # Forcer taille finale 2048×2048
                if dct_tile.shape != (2048, 2048):
                    pad_dct_final = np.zeros((2048, 2048), dtype=dct_full.dtype)
                    pad_dct_final[:dct_tile.shape[0], :dct_tile.shape[1]] = dct_tile
                    dct_tile = pad_dct_final

                patches.append({
                    'image': self.toctsr(patch),
                    'rgb': np.clip(np.abs(dct_tile), 0, 20),
                    'q': qtb,
                    'pos': (top, left)
                })

        return {
            'patches': patches,
            'path': str(img_path),
            'orig_size': (h, w)
        }



# --------------------
# Fonction d’évaluation patch par patch
# --------------------
def eval_net_dtd_no_labels(model, test_data, pth, out_dir, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_data):
            patches = sample['patches']
            orig_h, orig_w = sample['orig_size']
            path = sample['path']

            # Masques/Probabilités à la taille originale
            full_prob = np.zeros((orig_h, orig_w), dtype=np.float32)  # prob de classe 1
            full_cnt  = np.zeros((orig_h, orig_w), dtype=np.float32)  # compteur si overlap (ici =1 partout)

            # Traitement patch par patch (probabilités)
            for p in patches:
                data = Variable(p['image'].unsqueeze(0).to(device))
                dct_coef = Variable(torch.tensor(p['rgb']).unsqueeze(0).to(device))
                qs = Variable(p['q'].unsqueeze(0).unsqueeze(1).to(device))

                # logits -> probas
                pred = model(data, dct_coef, qs)                     # (1,2,2048,2048)
                prob1 = torch.softmax(pred, dim=1)[:, 1]             # (1,2048,2048)
                prob1 = prob1.squeeze(0).cpu().numpy()               # (2048,2048) float32

                top, left = p['pos']
                h_patch = min(2048, orig_h - top)
                w_patch = min(2048, orig_w - left)

                # Accumulation (moyenne si overlap)
                full_prob[top:top+h_patch, left:left+w_patch] += prob1[:h_patch, :w_patch]
                full_cnt [top:top+h_patch, left:left+w_patch] += 1.0

            # Normaliser en cas d’overlap (ici = 1, mais robuste)
            full_cnt[full_cnt == 0] = 1.0
            full_prob /= full_cnt

            # Binarisation pour stats + overlay heatmap pour lisibilité
            full_mask = (full_prob >= 0.1).astype(np.uint8)
            proportion_cls1 = float(full_mask.sum()) / float(orig_h * orig_w)

            img_np = np.array(Image.open(path).convert('RGB'))       # image originale (pas recompressée)
            overlay = overlay_prob_heatmap(img_np, full_prob, alpha=0.45)

            # Ajouter contour du masque binaire pour netteté
            mask_bin = (full_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_cnt = overlay.copy()
            cv2.drawContours(overlay_cnt, contours, -1, (255, 0, 0), 2)  # rouge

            # Annoter la proportion de classe 1
            txt = f"Class1 ratio: {proportion_cls1:.4f}"
            cv2.putText(overlay_cnt, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(overlay_cnt, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)

            # Sauvegarde
            filename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{filename}_overlay.png")
            Image.fromarray(overlay_cnt).save(out_path)

    print(f"Overlays sauvegardés dans {out_dir}")


def overlay_prob_heatmap(img_rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.5):
    """
    img_rgb: (H,W,3) uint8
    prob: (H,W) float32 in [0,1]
    return: overlay uint8 (H,W,3)
    """
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    mask_u8 = (prob * 255.0 + 0.5).astype(np.uint8)               # [0..255]
    heatmap_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)    # (H,W,3) BGR
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay



def eval_net_dtd_single_patch(model, test_data, pth, out_dir, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_data):
            patches = sample['patches']
            orig_h, orig_w = sample['orig_size']
            path = sample['path']

            # Ici : on ne prend qu’UN seul patch (par exemple le premier)
            p = patches[0]   # <- si tu veux choisir un autre patch, change l’index
            data = Variable(p['image'].unsqueeze(0).to(device))
            dct_coef = Variable(torch.tensor(p['rgb']).unsqueeze(0).to(device))
            qs = Variable(p['q'].unsqueeze(0).unsqueeze(1).to(device))

            pred = model(data, dct_coef, qs)                     # (1,2,2048,2048)
            prob1 = torch.softmax(pred, dim=1)[:, 1]             # (1,2048,2048)
            prob1 = prob1.squeeze(0).cpu().numpy()               # (2048,2048)

            top, left = p['pos']
            h_patch = min(2048, orig_h - top)
            w_patch = min(2048, orig_w - left)

            # Crée une carte vide, mais on colle UNIQUEMENT la prédiction du patch choisi
            full_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
            full_prob[top:top+h_patch, left:left+w_patch] = prob1[:h_patch, :w_patch]

            # Binarisation + overlay
            full_mask = (full_prob >= 0.5).astype(np.uint8)
            proportion_cls1 = float(full_mask.sum()) / float(orig_h * orig_w)

            img_np = np.array(Image.open(path).convert('RGB'))
            overlay = overlay_prob_heatmap(img_np, full_prob, alpha=0.45)

            # Contours
            mask_bin = (full_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_cnt = overlay.copy()
            cv2.drawContours(overlay_cnt, contours, -1, (255, 0, 0), 2)

            txt = f"Class1 ratio: {proportion_cls1:.4f}"
            cv2.putText(overlay_cnt, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(overlay_cnt, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)

            # Sauvegarde
            filename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{filename}_overlay.png")
            Image.fromarray(overlay_cnt).save(out_path)

    print(f"Overlays sauvegardés dans {out_dir}")
# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Dossier contenant les JPEG')
    parser.add_argument('--pth', type=str, required=True, help='Checkpoint .pth de seg_dtd')
    parser.add_argument('--out', type=str, default='predictions_dtd', help='Dossier de sortie')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = seg_dtd('', 2).to(device)
    model = torch.nn.DataParallel(model)

    test_data = TamperDatasetJPEG(args.data_root, mode='test')

    eval_net_dtd_single_patch(model, test_data, args.pth, args.out, device=device)

if __name__ == "__main__":
    main()