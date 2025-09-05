# demonstrator.py
import os
import tempfile
import numpy as np
from PIL import Image

import torch
import torchvision
import jpegio
import cv2
import streamlit as st
from torch.cuda.amp import autocast

from models.dtd import seg_dtd
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
# Prétraitement d'une seule image en batch
# -------------------------
def preprocess_single_image(img: Image.Image, quality=100):
    """Prépare une seule image en batch comme patch_collate."""
    w, h = img.size
    patch_size = 512
    patches = []

    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            right = min(left + patch_size, w)
            bottom = min(top + patch_size, h)

            # Crop + pad
            patch_img = img.crop((left, top, right, bottom))
            pad_img = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
            pad_img.paste(patch_img, (0, 0))

            # Re-encode en JPEG grayscale
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pad_img_gray = pad_img.convert("L")
                pad_img_gray.save(tmp.name, "JPEG", quality=quality)
                jpg = jpegio.read(tmp.name)
                dct = np.ascontiguousarray(jpg.coef_arrays[0])
                qtb = torch.from_numpy(np.ascontiguousarray(jpg.quant_tables[0])).long().unsqueeze(0)
                comp_rgb = Image.open(tmp.name).convert("RGB").copy()
            os.unlink(tmp.name)

            # Torch pipeline
            to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
            img_t = to_tensor(comp_rgb)
            dct_t = torch.from_numpy(np.clip(np.abs(dct), 0, 20).astype(np.int64))

            patches.append({
                "image": img_t,   # (3,512,512)
                "rgb": dct_t,     # (512,512)
                "q": qtb          # (1,8,8)
            })

    # Empile tout en batch
    images = torch.stack([p["image"] for p in patches])
    rgbs   = torch.stack([p["rgb"] for p in patches])
    qtabs  = torch.stack([p["q"] for p in patches])

    return {"image": images, "rgb": rgbs, "q": qtabs}, w, h, patch_size


# -------------------------
# Inférence complète
# -------------------------
def run_inference(model, image: Image.Image, quality=100, device="cuda"):
    batch, w, h, patch_size = preprocess_single_image(image, quality)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for i, (top, left) in enumerate([(y, x) 
            for y in range(0, h, patch_size) 
            for x in range(0, w, patch_size)]):

        right = min(left + patch_size, w)
        bottom = min(top + patch_size, h)

        img_t = batch["image"][i].unsqueeze(0).to(device)
        dct_t = batch["rgb"][i].unsqueeze(0).to(device)
        qtb   = batch["q"][i].unsqueeze(0).to(device)

        with torch.no_grad(), autocast():
            out = model(img_t, dct_t, qtb)
            pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # Retirer le padding
        pred_cropped = pred[:bottom - top, :right - left]
        final_mask[top:bottom, left:right] = pred_cropped

    return final_mask


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha=0.5):
    """Superpose le masque (rouge) à l'image originale."""
    img_np = np.array(image.convert("RGB")).astype(np.uint8)
    mask_rgb = np.zeros_like(img_np)
    mask_rgb[..., 0] = mask * 255  # rouge
    overlay = cv2.addWeighted(img_np, 1 - alpha, mask_rgb, alpha, 0)
    return Image.fromarray(overlay)


# -------------------------
# Streamlit app
# -------------------------
st.title("Détecteur de falsification de documents (DTD)")

# Choix du modèle
model_choice = st.selectbox(
    "Choisissez le modèle à utiliser",
    ["DTD Fine-tuné", "DTD Fine-tuné Pièces d'identité"]
)

MODEL_PATHS = {
    "DTD Fine-tuné": "./checkpointsfinetune/checkpoint-best.pth",
    "DTD Fine-tuné Pièces d'identité": "./checkpoints_img/checkpoint-best.pth"
}

uploaded_file = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image originale", use_container_width=True)

    # Charger modèle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = seg_dtd("", n_class=2).to(device)
    ckpt = torch.load(MODEL_PATHS[model_choice], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Inférence
    with st.spinner("Analyse en cours..."):
        mask = run_inference(model, image, quality=100, device=device)
        overlay = overlay_mask(image, mask)

    st.image(overlay, caption="Superposition masque prédictif", use_container_width=True)

    if mask.sum() > 0:
        st.error("⚠️ Document modifié détecté")
    else:
        st.success("✅ Aucun signe de modification détecté")
