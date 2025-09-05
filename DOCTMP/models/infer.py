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
# Dataset JPEG découpé en patchs 512×512
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
        im = Image.open(img_path).convert('L')
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

        # --- Découpage en patchs 512×512 ---
        for top in range(0, h, 512):
            for left in range(0, w, 512):
                right = min(left + 512, w)
                bottom = min(top + 512, h)

                # ----- PATCH RGB -----
                patch = im.crop((left, top, right, bottom))
                ph, pw = patch.size[1], patch.size[0]
                if pw < 512 or ph < 512:
                    pad_img = Image.new('RGB', (512, 512), (0, 0, 0))
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

                # Forcer taille finale 512×512
                if dct_tile.shape != (512, 512):
                    pad_dct_final = np.zeros((512, 512), dtype=dct_full.dtype)
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
# Dataset JPEG sans padding/re-encodage (dummy)
# --------------------
import lmdb, six, pickle, torch, jpegio
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import cv2, tempfile


class DummyDatasetLMDB(Dataset):
    def __init__(self, roots, max_readers=64, mode="test"):
        # ouverture LMDB
        self.envs = lmdb.open(
            roots,
            max_readers=max_readers,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples
        self.mode = mode

        # normalisation standard
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            # --- image ---
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
            w, h = im.size

            # --- label ---
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)

            # --- DCT + Q-table ---
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                im.save(tmp.name, "JPEG", quality=100)
                jpg = jpegio.read(tmp.name)

            dct = jpg.coef_arrays[0].copy()
            qtb = torch.LongTensor(jpg.quant_tables[0])

        return {
            'patches': [{
                'image': self.toctsr(im),
                'rgb': np.clip(np.abs(dct), 0, 20),
                'q': qtb,
                'pos': (0, 0)
            }],
            'path': img_key,           # clé LMDB
            'orig_size': (h, w),
            'gt_mask': mask,           # GT
            'orig_img': np.array(im)   # image RGB originale
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
            path = sample['path']       # juste pour l’ID
            gt_mask = sample['gt_mask'] # déjà dans le dataset


            # Masques/Probabilités à la taille originale
            full_prob = np.zeros((orig_h, orig_w), dtype=np.float32)  # prob de classe 1
            full_cnt  = np.zeros((orig_h, orig_w), dtype=np.float32)  # compteur si overlap (ici =1 partout)

            # Traitement patch par patch (probabilités)
            for p in patches:
                data = Variable(p['image'].unsqueeze(0).to(device))
                dct_coef = Variable(torch.tensor(p['rgb']).unsqueeze(0).to(device))
                qs = Variable(p['q'].unsqueeze(0).unsqueeze(1).to(device))

                # logits -> probas
                pred = model(data, dct_coef, qs)                     # (1,2,512,512)
                prob1 = torch.softmax(pred, dim=1)[:, 1]             # (1,512,512)
                prob1 = prob1.squeeze(0).cpu().numpy()               # (512,512) float32

                top, left = p['pos']
                h_patch = min(512, orig_h - top)
                w_patch = min(512, orig_w - left)

                # Accumulation (moyenne si overlap)
                full_prob[top:top+h_patch, left:left+w_patch] += prob1[:h_patch, :w_patch]
                full_cnt [top:top+h_patch, left:left+w_patch] += 1.0

            # Normaliser en cas d’overlap (ici = 1, mais robuste)
            full_cnt[full_cnt == 0] = 1.0
            full_prob /= full_cnt

            # Binarisation pour stats + overlay heatmap pour lisibilité
            full_mask = (full_prob >= 0.5).astype(np.uint8)
            proportion_cls1 = float(full_mask.sum()) / float(orig_h * orig_w)

            img_np = sample['orig_img']

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

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
def eval_net_dtd_with_metrics(model, test_data, pth, out_dir, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    all_prec, all_rec, all_f1, all_iou = [], [], [], []

    with torch.no_grad():
        for sample in tqdm(test_data):
            patches = sample['patches']
            orig_h, orig_w = sample['orig_size']
            path = sample['path']

            # chemin masque GT supposé parallèle
            gt_mask = sample['gt_mask']


            full_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
            full_cnt  = np.zeros((orig_h, orig_w), dtype=np.float32)

            for p in patches:
                data = Variable(p['image'].unsqueeze(0).to(device))
                dct_coef = Variable(torch.tensor(p['rgb']).unsqueeze(0).to(device))
                qs = Variable(p['q'].unsqueeze(0).unsqueeze(1).to(device))

                pred = model(data, dct_coef, qs)                     # (1,2,512,512)
                prob1 = torch.softmax(pred, dim=1)[:, 1]             # (1,512,512)
                prob1 = prob1.squeeze(0).cpu().numpy()

                top, left = p['pos']
                h_patch = min(512, orig_h - top)
                w_patch = min(512, orig_w - left)
                full_prob[top:top+h_patch, left:left+w_patch] += prob1[:h_patch, :w_patch]
                full_cnt [top:top+h_patch, left:left+w_patch] += 1.0

            full_cnt[full_cnt == 0] = 1.0
            full_prob /= full_cnt
            full_mask = (full_prob >= 0.5).astype(np.uint8)

            # --- calcul métriques ---
            y_true = gt_mask.flatten()
            y_pred = full_mask.flatten()

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            iou  = jaccard_score(y_true, y_pred, zero_division=0)

            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)
            all_iou.append(iou)

            # Sauvegarde overlay
            img_np = sample['orig_img']   # déjà en numpy dans DummyDatasetLMDB

            overlay = overlay_prob_heatmap(img_np, full_prob, alpha=0.45)
            filename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{filename}_overlay.png")
            Image.fromarray(overlay).save(out_path)

    # --- Résumé global ---
    if all_prec:
        print("=== Résultats globaux ===")
        print(f"Précision moyenne : {np.mean(all_prec):.4f}")
        print(f"Rappel moyen     : {np.mean(all_rec):.4f}")
        print(f"F1-score moyen   : {np.mean(all_f1):.4f}")
        print(f"IoU moyen        : {np.mean(all_iou):.4f}")


# --------------------
# Main
# --------------------
from torch.utils.data import Subset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Dossier contenant les JPEG')
    parser.add_argument('--pth', type=str, required=True, help='Checkpoint .pth de seg_dtd')
    parser.add_argument('--out', type=str, default='predictions_dtd', help='Dossier de sortie')
    parser.add_argument('--n_test', type=int, default=50, help='Nombre d’éléments à tester (par défaut 50)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = seg_dtd('', 2).to(device)
    #model = torch.nn.DataParallel(model)

    full_dataset = DummyDatasetLMDB(args.data_root, mode='test')
    # Limiter à n_test éléments
    test_data = Subset(full_dataset, range(min(args.n_test, len(full_dataset))))

    eval_net_dtd_with_metrics(model, test_data, args.pth, args.out, device=device)


if __name__ == "__main__":
    main()
