# convert_weights_safe.py

import torch
import os
import sys

# Ajouter le chemin au dossier contenant models/
sys.path.append(os.path.abspath("."))

# Importer explicitement les bonnes classes
from models.swins import SwinTransformerV2, BasicLayer
from models.swins import (
    SwinTransformerV2,
    BasicLayer,
    SwinTransformerBlock,
    WindowAttention,
    PatchEmbed,
    PatchMerging,
    Mlp
)
import models.swins  # s'assure que tout est dans le namespace pickle



from models.fph import FPH
import models.dtd  # utile pour les dépendances internes à vph

def convert_weights_safe():
    os.makedirs("Converted", exist_ok=True)

    # === Swin ===
    try:
        print(" Trying to load swin_imagenet.pt ...")
        swin_model = torch.load("Weights/swin_imagenet.pt", map_location="cpu")
        swin_state = swin_model.state_dict() if hasattr(swin_model, "state_dict") else swin_model
        torch.save(swin_state, "Converted/swin_imagenet_state.pth")
        print(" Converted: swin_imagenet_state.pth")
    except Exception as e:
        print(" Failed to convert swin_imagenet.pt:", e)

    # === VPH ===
    try:
        print(" Trying to load vph_imagenet.pt ...")
        vph_model = torch.load("./DTD_Weights/Weights/vph_imagenet.pt", map_location="cpu")
        vph_state = vph_model.state_dict() if hasattr(vph_model, "state_dict") else vph_model
        torch.save(vph_state, "Converted/vph_imagenet_state.pth")
        print(" Converted: vph_imagenet_state.pth")
    except Exception as e:
        print(" Failed to convert vph_imagenet.pt:", e)

if __name__ == "__main__":
    convert_weights_safe()
