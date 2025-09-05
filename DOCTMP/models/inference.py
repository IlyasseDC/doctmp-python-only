# infer_dtd_on_images.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import jpegio
from tqdm import tqdm
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



class TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=95, qtb=90, max_readers=64):
        self.envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums=self.nSamples
        self.minq = minq
        self.mode = mode
        with open('pks/qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        with open('pks/'+roots+'_%d.pk'%minq,'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
            H,W = mask.shape
            record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei>1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei>0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im = im.convert("L")
                if choicei>1:
                    im.save(tmp,"JPEG",quality=q2)
                    im = Image.open(tmp)
                if choicei>0:
                    im.save(tmp,"JPEG",quality=q1)
                    im = Image.open(tmp)
                im.save(tmp,"JPEG",quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct),0,20),
                'q':use_qtb,
                'i':q
            }

# -------- Dataset images --------
class ImageFolderDTD(Dataset):
    def __init__(self, root, exts=(".jpg", ".jpeg"), quality=100):
        self.root = Path(root)
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]

        self.quality = quality
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        im = Image.open(path).convert("RGB")
        orig_w, orig_h = im.size

        # --- Ajustement taille ---
        if orig_w < 512 or orig_h < 512:
            im, padding = pad_to_multiple(im, mult=32)
        else:
            im = im.resize((512, 512), Image.LANCZOS)
            padding = (0, 0, 0, 0)  # pas de padding

        # --- Lire DCT et Q-table ---
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            im.save(tmp, "JPEG", quality=self.quality)
            jpg = jpegio.read(tmp.name)
            dct = np.clip(np.abs(jpg.coef_arrays[0]), 0, 20).astype(np.int64)
            q_table = torch.LongTensor(jpg.quant_tables[0])

        return {
            "image": self.toctsr(im),             # (3,H,W)
            "rgb": torch.from_numpy(dct),         # (H/8,W/8)
            "q": q_table,                         # (8,8)
            "path": str(path),
            "orig_w": orig_w,
            "orig_h": orig_h,
            "padding": torch.tensor(padding, dtype=torch.int)
        }
from PIL import ImageOps

def pad_to_multiple(img, mult=32):
    w, h = img.size
    pad_w = (mult - w % mult) % mult
    pad_h = (mult - h % mult) % mult
    padding = (0, 0, pad_w, pad_h)  # gauche, haut, droite, bas
    return ImageOps.expand(img, padding), padding

from PIL import Image
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2
def overlay_heatmap_with_legend_fixed(
    image_rgb: Image.Image,
    prob_map: np.ndarray,   # float in [0,1]
    alpha: float = 0.5,
    cmap: int = cv2.COLORMAP_JET,
    legend_width: int = 80
) -> Image.Image:
    # 1) Clamp and scale to [0,255]
    prob_map = np.clip(prob_map, 0.0, 1.0).astype(np.float32)
    mask_norm = np.rint(prob_map * 255.0).astype(np.uint8)

    # 2) Colormap
    heatmap_bgr = cv2.applyColorMap(mask_norm, cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 3) Blend with original
    image_np = np.asarray(image_rgb, dtype=np.uint8)
    if heatmap_rgb.shape[:2] != image_np.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb, (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)

    # 4) Build readable legend (top=1.0 -> bottom=0.0)
    h = image_np.shape[0]
    legend_vals = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    legend_color = cv2.applyColorMap(legend_vals, cmap)
    legend_color = cv2.cvtColor(legend_color, cv2.COLOR_BGR2RGB)
    legend_color = np.repeat(legend_color, legend_width, axis=1)

    # 5) Add ticks
    legend_with_text = legend_color.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_ticks = 6
    for i, val in enumerate(np.linspace(1.0, 0.0, n_ticks)):
        y = int(i * (h - 1) / (n_ticks - 1))
        cv2.putText(legend_with_text, f"{val:.2f}", (8, max(12, y)),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend_with_text, "Proba", (8, 16),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 6) Concatenate and return
    final_image = np.hstack((blended, legend_with_text))
    return Image.fromarray(final_image)



@torch.no_grad()
def run_inference(pth, img_root, out_dir, batch_size=1, num_workers=1, device="cuda"):
    img_root_path = Path(img_root)

    ds = ImageFolderDTD(img_root_path)
    dl = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = seg_dtd('', 2).to(dev)
    if dev.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dl, total=len(dl)):
        imgs = batch["image"].to(dev, non_blocking=True)                 # (B,3,H,W)
        dct  = batch["rgb"].to(dev, non_blocking=True)                   # (B,H/8,W/8)
        qs   = batch["q"].unsqueeze(1).to(dev, dtype=torch.long)         # (B,1,8,8)

        pred = model(imgs, dct, qs)                                      # (B,2,H,W)
        pred_prob = torch.softmax(pred, dim=1)[:, 1]                     # (B,H,W)

        # Quick sanity check on distribution
        print(
            f"proba stats - min:{pred_prob.min().item():.4f} "
            f"max:{pred_prob.max().item():.4f} mean:{pred_prob.mean().item():.4f}"
        )

        import torch.nn.functional as F

        B = pred_prob.shape[0]
        for i in range(B):
            src_path = Path(batch["path"][i])

            # Recreate output folder hierarchy
            try:
                rel = src_path.relative_to(img_root_path)
                save_dir = out_dir / rel.parent
            except ValueError:
                save_dir = out_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            # Target original size
            ow = int(batch["orig_w"][i].item() if hasattr(batch["orig_w"][i], "item") else batch["orig_w"][i])
            oh = int(batch["orig_h"][i].item() if hasattr(batch["orig_h"][i], "item") else batch["orig_h"][i])

            # Remove any padding (do it on the tensor)
            pad_left, pad_top, pad_right, pad_bottom = batch["padding"][i]
            pad_left   = int(pad_left)
            pad_top    = int(pad_top)
            pad_right  = int(pad_right)
            pad_bottom = int(pad_bottom)

            prob_t = pred_prob[i]  # (H,W) tensor on dev
            H, W = prob_t.shape[-2], prob_t.shape[-1]
            if pad_left or pad_top or pad_right or pad_bottom:
                prob_t = prob_t[
                    pad_top : H - pad_bottom,
                    pad_left: W - pad_right
                ]

            # Resize to original size using PyTorch, then to NumPy float32 in [0,1]
            prob_resized_t = F.interpolate(
                prob_t.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                size=(oh, ow),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)

            prob_i_resized = prob_resized_t.detach().cpu().numpy().astype(np.float32)
            prob_i_resized = np.clip(prob_i_resized, 0.0, 1.0)

            # Overlay and save
            orig_img = Image.open(src_path).convert("RGB")
            over = overlay_heatmap_with_legend_fixed(
                orig_img, prob_i_resized, alpha=0.5, cmap=cv2.COLORMAP_JET
            )
            out_overlay = save_dir / f"{src_path.stem}_overlay.png"
            over.save(out_overlay)


    print(f"Terminé. Masques et overlays enregistrés dans : {out_dir}")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True, help="Dossier des JPEG (racine)")
    ap.add_argument("--pth", type=str, required=True, help="checkpoint .pth de seg_dtd")
    ap.add_argument("--out", type=str, default="predictions_dtd", help="Dossier de sortie")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run_inference(args.pth, args.img_root, args.out,
                  batch_size=args.bs, num_workers=args.workers)


if __name__ == "__main__":
    main()