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
# demonstrator.py
# demonstrator.py
import os
import tempfile
import numpy as np
from PIL import Image
import base64

import torch
import torchvision
import jpegio
import cv2
import streamlit as st
from torch.cuda.amp import autocast

from models.dtd import seg_dtd

# ==============================
# Arrière-plan personnalisé
# ==============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Boîte blanche pour le texte
# ==============================
def white_box(text, size="16px"):
    st.markdown(
        f"""
        <div style="background-color:white; padding:20px; border-radius:10px; 
                    color:black; font-size:{size}; margin-bottom:15px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Prétraitement pipeline "fine-tuné" (re-encodage JPEG grayscale)
# -------------------------
def preprocess_single_image(img: Image.Image, quality=100, patch_size=512):
    w, h = img.size
    patches = []
    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            right = min(left + patch_size, w)
            bottom = min(top + patch_size, h)

            patch_img = img.crop((left, top, right, bottom))
            pad_img = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
            pad_img.paste(patch_img, (0, 0))

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pad_img_gray = pad_img.convert("L")
                pad_img_gray.save(tmp.name, "JPEG", quality=quality)
                jpg = jpegio.read(tmp.name)
                dct = np.ascontiguousarray(jpg.coef_arrays[0])
                qtb = torch.from_numpy(
                    np.ascontiguousarray(jpg.quant_tables[0])
                ).long().unsqueeze(0)
                comp_rgb = Image.open(tmp.name).convert("RGB").copy()
            os.unlink(tmp.name)

            to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
            ])
            img_t = to_tensor(comp_rgb)
            dct_t = torch.from_numpy(
                np.clip(np.abs(dct), 0, 20).astype(np.int64)
            )

            patches.append({
                "image": img_t,
                "rgb": dct_t,
                "q": qtb,
                "pos": (top, left, bottom, right)
            })

    return patches, (w, h), patch_size

# -------------------------
# Prétraitement pipeline "DTD original" (JPEG direct)
# -------------------------
def preprocess_jpeg_image(img_path: str, patch_size=512):
    im = Image.open(img_path).convert("L")
    w, h = im.size
    jpg = jpegio.read(img_path)
    dct_full = jpg.coef_arrays[0].copy()
    qtb = torch.LongTensor(jpg.quant_tables[0])

    patches = []
    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            right = min(left + patch_size, w)
            bottom = min(top + patch_size, h)

            patch = im.crop((left, top, right, bottom))
            pad_img = Image.new("L", (patch_size, patch_size), 0)
            pad_img.paste(patch, (0, 0))
            im_rgb = pad_img.convert("RGB")

            dct_tile = np.zeros((patch_size, patch_size), dtype=dct_full.dtype)
            dct_tile[:bottom - top, :right - left] = dct_full[top:bottom, left:right]

            to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
            ])
            img_t = to_tensor(im_rgb)
            dct_t = torch.from_numpy(
                np.clip(np.abs(dct_tile), 0, 20).astype(np.int64)
            )

            patches.append({
                "image": img_t,
                "rgb": dct_t,
                "q": qtb.unsqueeze(0),
                "pos": (top, left, bottom, right)
            })

    return patches, (w, h), patch_size

# -------------------------
# Inférence générique
# -------------------------
def run_inference(model, patches, size, patch_size=512, device="cuda"):
    w, h = size
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for patch in patches:
        top, left, bottom, right = patch["pos"]
        img_t = patch["image"].unsqueeze(0).to(device)
        dct_t = patch["rgb"].unsqueeze(0).to(device)
        qtb = patch["q"].unsqueeze(0).to(device)
        with torch.no_grad(), autocast():
            out = model(img_t, dct_t, qtb)
            pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred_cropped = pred[:bottom - top, :right - left]
        final_mask[top:bottom, left:right] = pred_cropped
    return final_mask

def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0)):
    img_np = np.array(image.convert("RGB")).astype(np.uint8)
    blended = img_np.copy()
    blended[mask == 1] = (0.5 * img_np[mask == 1] + 0.5 * np.array(color)).astype(np.uint8)
    return Image.fromarray(blended)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Démo DTD", layout="wide")

# Masquer menu / footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -- Background
set_background("/home/ilyassechaouki/DOCTMP/Background.png")

if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# ===== ACCUEIL =====
if st.session_state.page == "Accueil":
    white_box("<h2>🎉 Démonstrateur DTD – Détection de falsification</h2>", size="22px")
    white_box(
        """
        👋 Bienvenue dans ce démonstrateur !  

        Vous pouvez tester différents modèles pour détecter les falsifications :  
        - 🧩 <b>DTD</b> : modèle original entraîné avec LMDB  
        - 🔧 <b>DTD Fine-tuné</b> : adapté à nos données  
        - 🪪 <b>DTD Fine-tuné Pièces d'identité</b> : spécialisé sur les documents d'identité  

        👉 Sélectionnez un modèle pour passer à la page d’inférence.
        """,
        size="18px"
    )

    model_choice = st.radio(
        "Choisissez un modèle :",
        ["🧩 DTD", "🔧 DTD Fine-tuné", "🪪 DTD Fine-tuné Pièces d'identité"]
    )

    if st.button("Continuer ➡️"):
        st.session_state.model_choice = model_choice
        st.session_state.page = "Inférence"
        st.rerun()

# ===== INFÉRENCE =====
elif st.session_state.page == "Inférence":
    white_box("<h2>🔍 Page d’inférence</h2>", size="22px")
    
    model_choice = st.session_state.model_choice
    MODEL_PATHS = {
        "🧩 DTD": "./Weights/dtd_doctamper.pth",
        "🔧 DTD Fine-tuné": "./checkpointsfinetune/checkpoint-best.pth",
        "🪪 DTD Fine-tuné Pièces d'identité": "./checkpoints_img/checkpoint-best.pth",
    }
    model_choice = st.selectbox(
        "Choisissez le modèle à utiliser :",
        ["🧩 DTD", "🔧 DTD Fine-tuné", "🪪 DTD Fine-tuné Pièces d'identité"],
        index=["🧩 DTD", "🔧 DTD Fine-tuné", "🪪 DTD Fine-tuné Pièces d'identité"].index(st.session_state.model_choice)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Déposez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            img_path = tmp.name
        image = Image.open(img_path).convert("RGB")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = seg_dtd("", n_class=2).to(device)
        if model_choice == "🧩 DTD":
            model = torch.nn.DataParallel(model)
        ckpt = torch.load(MODEL_PATHS[model_choice], map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with st.spinner("⏳ Analyse en cours..."):
            if model_choice == "🧩 DTD":
                patches, size, patch_size = preprocess_jpeg_image(img_path)
            else:
                patches, size, patch_size = preprocess_single_image(image, quality=100)

            mask = run_inference(model, patches, size, patch_size=patch_size, device=device)
            overlay = overlay_mask(image, mask)

        col1, col2 = st.columns(2)
        with col1:
            white_box("🖼️ **Image originale**")
            st.image(image, use_container_width=True)
        with col2:
            white_box("📌 **Résultat prédictif**")
            st.image(overlay, use_container_width=True)

        if mask.sum() > 0:
            white_box("⚠️ <b>Document modifié détecté</b>", size="18px")
        else:
            white_box("✅ <b>Aucun signe de modification détecté</b>", size="18px")

    if st.button("⬅️ Retour à l’accueil"):
        st.session_state.page = "Accueil"
        st.rerun()
