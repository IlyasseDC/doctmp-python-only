# demonstrator.py
import os
import tempfile
import numpy as np
from PIL import Image
import subprocess, sys
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

# Force downgrade si numpy >= 2
try:
    import numpy
    if int(numpy.__version__.split('.')[0]) >= 2:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==1.20.1"])
        import importlib
        importlib.reload(numpy)
except Exception as e:
    print("⚠️ Auto-downgrade numpy failed:", e)
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
import gdown
WEIGHTS_DIR = "./Weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODEL_DRIVE = {
    "🧩 DTD Original": {
        "id": "1a1qR_t1ZYbUB_XnWrhftWX-QfBeoGdLF",  # dtd_doctamper.pth
        "filename": "dtd_doctamper.pth"
    },
    "📦 Swin ImageNet": {
        "id": "1cz6dnFsI9tpfad7E1Y7uR4LFKiJHgv1U",  # swin_imagenet.pt
        "filename": "swin_imagenet.pt"
    },
    "📦 VPH ImageNet": {
        "id": "1CeI6_dVjD7aN1417SZEy6yL5sQXFtQyM",  # vph_imagenet.pt
        "filename": "vph_imagenet.pt"
    }
}

def get_model_path(model_key):
    info = MODEL_DRIVE[model_key]
    local_path = os.path.join(WEIGHTS_DIR, info["filename"])

    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?export=download&id={info['id']}"
        st.info(f"📥 Downloading {info['filename']} ...")
        gdown.download(url, local_path, quiet=False)

    return local_path

# ==============================
# Page Configuration
# ==============================
ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "Image2.png")
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")
page_icon = Image.open(ICON_PATH)
st.set_page_config(
    page_title="DTD Document Forgery Detection",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS Styles
# ==============================
def load_custom_css():
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Navigation bar */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0rem 0rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .nav-title {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .card-gradient {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
    }
    
    /* Sample image card styling */
    .sample-image-card {
        background: white;
        border-radius: 12px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0.5rem 0;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .sample-image-card:hover {
        border-color: #667eea;
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .sample-image-card.selected {
        border-color: #451DC7;
        box-shadow: 0 8px 25px rgba(69, 29, 199, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #451DC7 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Alert styling */
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Model selection styling */
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0.5rem 0;
    }
    
    .model-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .model-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #e3e8ff 100%);
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 12px;
        border: 2px solid #e1e5e9;
    }
    
    /* Text styling */
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #451DC7 0%, #6B46C1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
        /* ===== Top fixed nav ciblée par un marqueur adjacent ===== */
    /* Le bloc de colonnes immédiatement après #nav-row-marker devient la barre */
    #nav-row-marker + div[data-testid="stHorizontalBlock"]{
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        height: 64px;
        background: #451DC7;                 /* <-- couleur souhaitée */
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-radius: 0 !important;
        margin: 0 !important;
    }
    /* Colonne du titre (à gauche) */
    #nav-row-marker + div[data-testid="stHorizontalBlock"] > div:first-child{
        display: flex; align-items: center;
    }
    /* Titre dans la barre */
    .top-nav-title{
        color: #451DC7;          /* texte blanc pour contraster avec #451DC7 */
        font-weight: 800;
        font-size: 20px;         /* ← augmente la taille (26–28px si tu veux + gros) */
        line-height: 1.1;
        margin: 0;
        white-space: nowrap;
        display: flex; 
        align-items: center; 
        gap: 10px;               /* petit espace entre l'icône et le texte */
        margin-bottom: 20px;
    }

    /* Boutons nav : taille uniforme (sans toucher aux autres boutons de l'app) */
    #nav-row-marker + div[data-testid="stHorizontalBlock"] .stButton > button{
        width: 140px;
        height: 40px;
        padding: 0;
        display: inline-flex; align-items: center; justify-content: center;
        border-radius: 10px;
        box-sizing: border-box;
    }
    /* Décale le contenu pour ne pas passer sous la barre */
    .main .block-container{ padding-top: 0px !important; }

    /* Mobile */
    @media (max-width: 768px){
        #nav-row-marker + div[data-testid="stHorizontalBlock"]{ height: 72px; padding: 0 12px; }
        .top-nav-title{ font-size: 16px; }
        #nav-row-marker + div[data-testid="stHorizontalBlock"] .stButton > button{
            width: 120px; height: 38px;
        }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
        /* Sample image button specific styling */
    .sample-image-button button[data-testid="stBaseButton-secondary"] {
        background: none !important;
        border: 2px solid transparent !important;
        border-radius: 12px !important;
        padding: 0 !important;
        height: 200px !important;
        width: 100% !important;
        color: transparent !important;
        font-size: 0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .sample-image-button button[data-testid="stBaseButton-secondary"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 25px rgba(69,29,199,0.3) !important;
        border-color: #451DC7 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 12px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #451DC7 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# Helper Functions
# ==============================
def create_card(content, gradient=False):
    card_class = "card card-gradient" if gradient else "card"
    return f'<div class="{card_class}">{content}</div>'

def create_alert(message, alert_type="success"):
    icon = "✅" if alert_type == "success" else "⚠️"
    return f'<div class="alert-{alert_type}"><strong>{icon} {message}</strong></div>'

def get_sample_images():
    """Get list of sample images from assets folder"""
    sample_images = []
    
    # Define supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    # Search for images in assets folder
    for extension in extensions:
        files = glob.glob(os.path.join(ASSETS_PATH, extension))
        sample_images.extend(files)
    
    # Remove the icon file from the list
    sample_images = [img for img in sample_images if not img.endswith('Image2.png')]
    
    # Create a dictionary with custom names (you can modify this later)
    image_dict = {}
    for i, img_path in enumerate(sample_images):
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        # You can customize these names later
        display_name = name_without_ext.replace('_', ' ').title()
        image_dict[display_name] = img_path
    
    return image_dict

# ==============================
# Preprocessing Functions
# ==============================
def preprocess_single_image(img: Image.Image, quality=100, patch_size=512):
    """Preprocessing pipeline for fine-tuned models (JPEG grayscale re-encoding)"""
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

def preprocess_jpeg_image(img_path: str, patch_size=512):
    """Preprocessing pipeline for original DTD model (direct JPEG)"""
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

def run_inference(model, patches, size, patch_size=512, device="cuda"):
    """Generic inference function"""
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
    """Create overlay of mask on original image"""
    img_np = np.array(image.convert("RGB")).astype(np.uint8)
    blended = img_np.copy()
    blended[mask == 1] = (0.5 * img_np[mask == 1] + 0.5 * np.array(color)).astype(np.uint8)
    return Image.fromarray(blended)

def analyze_image(image_source, image_data=None, image_path=None):
    """Analyze image for tampering detection"""
    # Map models -> Google Drive
    MODEL_MAP = {
        "🧩 DTD Original": "🧩 DTD Original",
        "🔧 DTD Fine-tuned": "📦 Swin ImageNet",   # ← adapte ici si tu as mis le bon poids
        "🆔 DTD ID Documents": "📦 VPH ImageNet",  # ← adapte aussi
    }
        
    try:
        # Handle different image sources
        if image_source == "upload" and image_data is not None:
            # For uploaded files
            image = Image.open(image_data).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_data.getbuffer())
                temp_img_path = tmp.name
        elif image_source == "sample" and image_path is not None:
            # For sample images
            image = Image.open(image_path).convert("RGB")
            temp_img_path = image_path
        else:
            return None, None, "Invalid image source"
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = seg_dtd("", n_class=2).to(device)
        
        # Handle DataParallel for original DTD model
        if st.session_state.selected_model == "🧩 DTD Original":
            model = torch.nn.DataParallel(model)
        
        # Load checkpoint
        # Récupère le chemin du poids depuis Drive (cache local si déjà téléchargé)
        model_key = st.session_state.selected_model
        if model_key not in MODEL_MAP:
            return None, None, f"Unknown model: {model_key}"

        model_path = get_model_path(MODEL_MAP[model_key])
        
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        # Preprocessing
        if st.session_state.selected_model == "🧩 DTD Original":
            patches, size, patch_size = preprocess_jpeg_image(temp_img_path)
        else:
            patches, size, patch_size = preprocess_single_image(image, quality=100)
        
        # Run inference
        mask = run_inference(model, patches, size, patch_size=patch_size, device=device)
        overlay = overlay_mask(image, mask)
        
        # Cleanup temporary file for uploaded images
        if image_source == "upload":
            os.unlink(temp_img_path)
        
        return mask, overlay, None
        
    except Exception as e:
        # Cleanup temporary file in case of error
        if image_source == "upload" and 'temp_img_path' in locals():
            try:
                os.unlink(temp_img_path)
            except:
                pass
        return None, None, str(e)

# ==============================
# Main Application
# ==============================
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "🧩 DTD Original"
    if "selected_sample_image" not in st.session_state:
        st.session_state.selected_sample_image = None
    
    # Navigation header
    # ===== Barre de navigation (titre à gauche, boutons à droite) =====
    # Marqueur pour cibler la rangée suivante (nav) côté CSS
    st.markdown('<div id="nav-row-marker"></div>', unsafe_allow_html=True)

    # ===== Barre de navigation (titre à gauche, boutons à droite) =====
    c1, c2, c3, c4 = st.columns([6, 3, 3, 3])
    with c1:
        with open(ICON_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .brand-icon {{ height:32px; width:auto; margin-right:8px; vertical-align:middle; }}
        </style>
        <div class="top-nav-title">
        <img class="brand-icon" src="data:image/png;base64,{logo_b64}" alt="logo">
        DTD Document Forgery Detection System
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "home"; st.rerun()
    with c3:
        if st.button("Detection", key="nav_detection"):
            st.session_state.page = "detection"; st.rerun()
    with c4:
        if st.button("About", key="nav_about"):
            st.session_state.page = "about"; st.rerun()

    st.markdown("---")
    
    # Page routing
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "detection":
        show_detection_page()
    elif st.session_state.page == "about":
        show_about_page()

def show_home_page():
    """Home page with model selection"""
    st.markdown('<h1 class="title-text">Welcome to DTD Demonstrator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Advanced Document Tampering Detection using Deep Learning</p>', unsafe_allow_html=True)
    
    # Model selection
    st.markdown(create_card(
        '<h3>🎯 Select Detection Model</h3><p>Choose the model that best fits your document type:</p>'
    ), unsafe_allow_html=True)
    
    # Model options in horizontal layout
    models = {
        "🧩 DTD Original": {
            "description": "Original DTD model trained with LMDB dataset",
            "best_for": "General document forgery detection"
        },
        "🔧 DTD Fine-tuned": {
            "description": "Fine-tuned model adapted to custom datasets",
            "best_for": "Enhanced accuracy on specific document types"
        },
        "🆔 DTD ID Documents": {
            "description": "Specialized model for identity documents",
            "best_for": "Identity cards, passports, and official documents"
        }
    }
    
    # Create three columns for horizontal layout
    col1, col2, col3 = st.columns(3)
    
    columns = [col1, col2, col3]
    model_keys = list(models.keys())
    
    for i, (model_key, model_info) in enumerate(models.items()):
        with columns[i]:
            # Model description card
            st.markdown(create_card(f'''
                <h4 style="text-align: center; margin-bottom: 1rem;">{model_key}</h4>
                <p><strong>Description:</strong> {model_info["description"]}</p>
                <p><strong>Best for:</strong> {model_info["best_for"]}</p>
            '''), unsafe_allow_html=True)
            # Select button under each card
            if st.button(f"Select", key=f"select_{model_key}", use_container_width=True):
                st.session_state.selected_model = model_key
    # Current selection display
    if st.session_state.selected_model:
        st.markdown(create_alert(f"Current model: {st.session_state.selected_model}", "success"), 
                   unsafe_allow_html=True)
    
    # Start detection button
    if st.button("Go to detection page", key="start_detection", use_container_width=True):
        st.session_state.page = "detection"
        st.rerun()

def show_detection_page():
    """Detection page with file upload and analysis"""
    st.markdown('<h1 class="title-text">Document Analysis</h1>', unsafe_allow_html=True)
    
    # Model info
    st.markdown(create_card(f'''
        <h4>Current Model: {st.session_state.selected_model}</h4>
        <p>Upload an image or choose from sample images to analyze for potential document tampering.</p>
    '''), unsafe_allow_html=True)
    
    # Create tabs for different image input methods
    tab1, tab2 = st.tabs(["📂 Upload Image", "🖼️ Sample Images"])
    
    image_to_analyze = None
    image_source = None
    image_path = None
    
    with tab1:
        st.markdown("### Upload Your Document")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert("RGB")
            image_source = "upload"
            st.success("✅ Image uploaded successfully!")
    
    with tab2:
        st.markdown("### Choose from Sample Images")
        sample_images = get_sample_images()

        if sample_images:
            image_paths = list(sample_images.values())
            
            st.markdown("**Click directly on any image to select it:**")
            st.markdown("<br>", unsafe_allow_html=True)
            # Display images in a grid with proper click functionality
            cols_per_row = 3
            for i in range(0, len(image_paths), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_paths):
                        img_path = image_paths[i + j]
                        with col:
                            # Display image with custom styling
                            img = Image.open(img_path)
                            
                            # Check if selected
                            is_selected = st.session_state.get("selected_sample_image") == img_path
                            border_color = "#451DC7" if is_selected else "#e1e5e9"
                            border_width = "3px" if is_selected else "2px"
                            
                            # Custom container with selection indicator
                            st.markdown(f'''
                            <div style="
                                border: {border_width} solid {border_color};
                                border-radius: 12px;
                                padding: 8px;
                                margin-bottom: 15px;
                                background: white;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                                transition: all 0.3s ease;
                                {'box-shadow: 0 8px 25px rgba(69, 29, 199, 0.3);' if is_selected else ''}
                            ''', unsafe_allow_html=True)
                            
                            # Display the actual image
                            st.image(img, use_container_width=True)
                            
                            # Image name and selection button
                            img_name = os.path.basename(img_path)
                            if is_selected:
                                st.success("✅ Selected")
                            else:
                                if st.button(f"Select", key=f"img_btn_{i}_{j}", use_container_width=True):
                                    st.session_state.selected_sample_image = img_path
                                    st.session_state.current_image = img
                                    st.session_state.current_image_source = "sample"
                                    st.rerun()
                            
                            st.markdown("</div>", unsafe_allow_html=True)

            # Display currently selected image info
            if st.session_state.get("selected_sample_image"):
                image_path = st.session_state.selected_sample_image
                image_to_analyze = Image.open(image_path).convert("RGB")
                image_source = "sample"
        else:
            st.warning("⚠️ No sample images found in the assets folder.")

    # Analysis section - moved outside tabs
    st.markdown("---")
    st.markdown("## 🔍 Document Analysis")
    
    if image_to_analyze is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📄 Original Document")
            st.image(image_to_analyze, use_container_width=True, caption="Selected Image")
        
        # Analysis button and results
        if st.button("🔍 Analyze Document", key="analyze_btn", use_container_width=True):
            with st.spinner("Analyzing document for tampering..."):
                if image_source == "upload":
                    mask, overlay, error = analyze_image("upload", uploaded_file)
                elif image_source == "sample":
                    mask, overlay, error = analyze_image("sample", image_path=st.session_state.selected_sample_image)
                else:
                    error = "Invalid image source"
                    mask, overlay = None, None
                
                if error:
                    st.error(f"❌ Error during analysis: {error}")
                elif mask is not None and overlay is not None:
                    with col2:
                        st.markdown("#### 🎯 Detection Results")
                        st.image(overlay, use_container_width=True, caption="Tampering Detection Overlay")
                    
                    # Results analysis
                    tampering_detected = mask.sum() > 0
                    if tampering_detected:
                        st.warning("⚠️ **TAMPERING DETECTED** - Suspicious regions found in the document")
                        tampered_pixels = mask.sum()
                        total_pixels = mask.shape[0] * mask.shape[1]
                        percentage = (tampered_pixels / total_pixels) * 100
                        
                        st.markdown(f'''
                        **📊 Detection Statistics:**
                        - **Tampered pixels:** {tampered_pixels:,}
                        - **Total pixels:** {total_pixels:,}
                        - **Tampering ratio:** {percentage:.2f}%
                        ''')
                    else:
                        st.success("✅ **NO TAMPERING DETECTED** - Document appears authentic")
    else:
        st.info("👆 Please upload an image or select a sample image to start analysis.")


def show_about_page():
    """About page with information about the system"""
    st.markdown('<h1 class="title-text">About DTD System</h1>', unsafe_allow_html=True)
    
    st.markdown(create_card('''
        <h3>🎯 What is DTD?</h3>
        <p>DTD (Document Tampering Detection) is a state-of-the-art deep learning system designed to detect 
        digital forgeries and tampering in document images. The system uses advanced neural networks to 
        analyze JPEG compression artifacts and identify regions that have been modified.</p>
    '''), unsafe_allow_html=True)
    
    st.markdown(create_card('''
        <h3>🔬 How it works</h3>
        <ul>
            <li><strong>DCT Analysis:</strong> Examines JPEG Discrete Cosine Transform coefficients</li>
            <li><strong>Neural Networks:</strong> Uses deep learning to identify tampering patterns</li>
            <li><strong>Patch Processing:</strong> Analyzes images in patches for detailed detection</li>
            <li><strong>Overlay Visualization:</strong> Highlights suspicious regions in red</li>
        </ul>
    '''), unsafe_allow_html=True)
    
    st.markdown(create_card('''
        <h3>📈 Model Variants</h3>
        <ul>
            <li><strong>DTD Original:</strong> Base model trained on LMDB dataset</li>
            <li><strong>DTD Fine-tuned:</strong> Enhanced model with improved accuracy</li>
            <li><strong>DTD ID Documents:</strong> Specialized for identity documents</li>
        </ul>
    '''), unsafe_allow_html=True)
    
    st.markdown(create_card('''
        <h3>⚙️ Technical Specifications</h3>
        <ul>
            <li><strong>Input:</strong> JPEG images (JPG, JPEG, PNG)</li>
            <li><strong>Processing:</strong> Patch-based analysis (512x512 patches)</li>
            <li><strong>Output:</strong> Binary mask highlighting tampered regions</li>
            <li><strong>Framework:</strong> PyTorch with CUDA acceleration</li>
        </ul>
    '''), unsafe_allow_html=True)
    
    st.markdown(create_card('''
        <h3>📚 How to Use Sample Images</h3>
        <p>The system now includes sample images that you can use to test the detection capabilities:</p>
        <ul>
            <li><strong>Sample Images Tab:</strong> Browse through pre-loaded test images</li>
            <li><strong>One-Click Selection:</strong> Simply click "Select" under any image to analyze it</li>
            <li><strong>Variety of Cases:</strong> Images include both authentic and tampered documents</li>
            <li><strong>Custom Names:</strong> Each sample image has a descriptive name for easy identification</li>
        </ul>
        <p><em>Note: Sample images are stored in the assets folder and automatically detected by the system.</em></p>
    '''), unsafe_allow_html=True)

if __name__ == "__main__":
    main()