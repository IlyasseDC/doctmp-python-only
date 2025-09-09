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
# Page Configuration
# ==============================
ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "Image2.png")
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
        font-size: 24px;         /* ← augmente la taille (26–28px si tu veux + gros) */
        line-height: 1.1;
        margin: 0;
        white-space: nowrap;
        display: flex; 
        align-items: center; 
        gap: 10px;               /* petit espace entre l’icône et le texte */
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




    
    # Add custom CSS for hidden navigation buttons

    
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
    if st.button("🚀 Start Detection", key="start_detection", use_container_width=True):
        st.session_state.page = "detection"
        st.rerun()

def show_detection_page():
    """Detection page with file upload and analysis"""
    st.markdown('<h1 class="title-text">Document Analysis</h1>', unsafe_allow_html=True)
    
    # Model info
    st.markdown(create_card(f'''
        <h4>Current Model: {st.session_state.selected_model}</h4>
        <p>Upload an image to analyze for potential document tampering.</p>
    '''), unsafe_allow_html=True)
    
    # Model configuration
    MODEL_PATHS = {
        "🧩 DTD Original": "./Weights/dtd_doctamper.pth",
        "🔧 DTD Fine-tuned": "./checkpointsfinetune/checkpoint-best.pth",
        "🆔 DTD ID Documents": "./checkpoints_img/checkpoint-best.pth",
    }
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload Document Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(create_card('<h4>📄 Original Document</h4>'), unsafe_allow_html=True)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True, caption="Uploaded Image")
        
        # Analysis button
        if st.button("🔍 Analyze Document", key="analyze_btn", use_container_width=True):
            with st.spinner("Analyzing document for tampering..."):
                try:
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        img_path = tmp.name
                    
                    # Load model
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = seg_dtd("", n_class=2).to(device)
                    
                    # Handle DataParallel for original DTD model
                    if st.session_state.selected_model == "🧩 DTD Original":
                        model = torch.nn.DataParallel(model)
                    
                    # Load checkpoint
                    model_path = MODEL_PATHS[st.session_state.selected_model]
                    if os.path.exists(model_path):
                        ckpt = torch.load(model_path, map_location="cpu")
                        model.load_state_dict(ckpt["state_dict"])
                        model.eval()
                        
                        # Preprocessing
                        if st.session_state.selected_model == "🧩 DTD Original":
                            patches, size, patch_size = preprocess_jpeg_image(img_path)
                        else:
                            patches, size, patch_size = preprocess_single_image(image, quality=100)
                        
                        # Run inference
                        mask = run_inference(model, patches, size, patch_size=patch_size, device=device)
                        overlay = overlay_mask(image, mask)
                        
                        # Display results
                        with col2:
                            st.markdown(create_card('<h4>🎯 Detection Results</h4>'), unsafe_allow_html=True)
                            st.image(overlay, use_container_width=True, caption="Tampering Detection Overlay")
                        
                        # Analysis results
                        tampering_detected = mask.sum() > 0
                        if tampering_detected:
                            st.markdown(create_alert("TAMPERING DETECTED - Suspicious regions found in the document", "warning"), 
                                       unsafe_allow_html=True)
                            tampered_pixels = mask.sum()
                            total_pixels = mask.shape[0] * mask.shape[1]
                            percentage = (tampered_pixels / total_pixels) * 100
                            st.markdown(create_card(f'''
                                <h4>📊 Detection Statistics</h4>
                                <p><strong>Tampered pixels:</strong> {tampered_pixels:,}</p>
                                <p><strong>Total pixels:</strong> {total_pixels:,}</p>
                                <p><strong>Tampering ratio:</strong> {percentage:.2f}%</p>
                            '''), unsafe_allow_html=True)
                        else:
                            st.markdown(create_alert("✅ NO TAMPERING DETECTED - Document appears authentic", "success"), 
                                       unsafe_allow_html=True)
                    
                    else:
                        st.error(f"❌ Model file not found: {model_path}")
                    
                    # Cleanup
                    os.unlink(img_path)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

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

if __name__ == "__main__":
    main()