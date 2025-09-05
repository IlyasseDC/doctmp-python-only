# demo_streamlit.py
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import jpegio
import torchvision.transforms.functional as TF

from models.dtd import seg_dtd  # ton modèle

# ----------- CONFIG ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATHS = {
    "DTD Pré-entraîné": "./checkpointsfinetune/checkpoint-best.pth",
    "Pièces d’identité": "./checkpointsfinetune/checkpoint-best.pth",
    "Tickets": "./checkpointsfinetune/checkpoint-best.pth"
}

# ----------- CHARGEMENT MODELES -----------
@st.cache_resource
def load_model(model_path):
    model = seg_dtd("", 2).to(DEVICE)
    #model = torch.nn.DataParallel(model)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# ----------- PIPELINE INFERENCE -----------
def preprocess_image(img: Image.Image, quality=100):
    img_rgb = img.convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_gray = img_rgb.convert("L")
        img_gray.save(tmp.name, "JPEG", quality=quality)
        jpg = jpegio.read(tmp.name)
        dct = np.ascontiguousarray(jpg.coef_arrays[0])
        qtb = torch.from_numpy(np.ascontiguousarray(jpg.quant_tables[0])).long().unsqueeze(0)
        comp_rgb = Image.open(tmp.name).convert("RGB").copy()

    comp_rgb = comp_rgb.resize((512, 512))
    dct_tile = cv2.resize(np.clip(np.abs(dct), 0, 20), (512, 512), interpolation=cv2.INTER_NEAREST)

    img_t = torch.from_numpy(np.array(comp_rgb).transpose(2,0,1)).float()/255.
    img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    dct_t = torch.from_numpy(dct_tile).long()
    return img_t.unsqueeze(0), dct_t.unsqueeze(0), qtb.unsqueeze(0)

def infer(model, img: Image.Image):
    image_t, dct_t, qtb_t = preprocess_image(img)
    image_t, dct_t, qtb_t = image_t.to(DEVICE), dct_t.to(DEVICE), qtb_t.to(DEVICE)
    with torch.no_grad():
        logits = model(image_t, dct_t, qtb_t)
        probs = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
    mask = (probs > 0.5).astype(np.uint8)
    return mask, probs

def overlay_mask(image: Image.Image, mask: np.ndarray):
    img_np = np.array(image.resize((512,512)))
    heatmap = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return overlay

# ----------- ROUTER -----------
def accueil():
    st.title("Démo Détection de Documents Falsifiés")

    st.markdown("### Choisissez un modèle :")
    col1, col2, col3 = st.columns(3)

    if col1.button("📄 DTD Pré-entraîné", use_container_width=True):
        st.session_state["page"] = "DTD Pré-entraîné"

    if col2.button("🪪 Pièces d’identité", use_container_width=True):
        st.session_state["page"] = "Pièces d’identité"

    if col3.button("🎟️ Tickets", use_container_width=True):
        st.session_state["page"] = "Tickets"


def page_modele(nom_modele):
    st.title(f"Détection avec le modèle : {nom_modele}")

    if MODEL_PATHS[nom_modele] is None:
        st.warning("🚧 Ce modèle n’est pas encore implémenté.")
        if st.button("⬅️ Retour"):
            st.session_state["page"] = "Accueil"
        return

    model = load_model(MODEL_PATHS[nom_modele])

    uploaded_file = st.file_uploader("Chargez une image de document", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image d'entrée", use_container_width=True)

        with st.spinner("Analyse en cours..."):
            mask, probs = infer(model, image)
            overlay = overlay_mask(image, mask)

        st.image(overlay, caption="Résultat - Superposition", use_container_width=True)

        if mask.sum() > 0:
            st.error("⚠️ Document modifié détecté")
        else:
            st.success("✅ Aucun signe de modification détecté")

    if st.button("⬅️ Retour"):
        st.session_state["page"] = "Accueil"


# ----------- APP -----------

if "page" not in st.session_state:
    st.session_state["page"] = "Accueil"

if st.session_state["page"] == "Accueil":
    accueil()
else:
    page_modele(st.session_state["page"])
