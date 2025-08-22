from pathlib import Path
from PIL import Image

# Dossier contenant les images
input_dir = Path("/home/ilyassechaouki/DOCTMP/output_jpeg/authentic")

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        with Image.open(img_path) as im:
            w, h = im.size
            print(f"{img_path.name} -> {w}x{h}")
