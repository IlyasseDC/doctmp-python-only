import os
from PIL import Image

input_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/labeled doc tamp dataset clean"
output_dir = "/home/ilyassechaouki/DOCTMP/image doc tamp dataset clean"
os.makedirs(output_dir, exist_ok=True)

# Qualité JPEG (classique 95 ou 100 pour minimiser les pertes)
JPEG_QUALITY = 100  

# Option : désactiver l'optimisation qui peut modifier un peu les tables
EXTRA_PARAMS = {
    "subsampling": 2,      # YCbCr 4:2:0 (standard)
    "quality": JPEG_QUALITY,
    "dpi": (72, 72),
    "optimize": False,     # éviter réoptimisation
    "progressive": False   # éviter multi-pass
}

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".tif") or fname.lower().endswith(".tiff"):
        path = os.path.join(input_dir, fname)
        im = Image.open(path).convert("RGB")  # éviter palette/grayscale incohérente
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".jpg")
        im.save(out_path, "JPEG", **EXTRA_PARAMS)
        print(f"✔ {fname} → {out_path}")
