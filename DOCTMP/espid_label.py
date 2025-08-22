import os
import cv2
import numpy as np

def create_empty_masks(images_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Impossible de lire {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Sauvegarde du masque noir
        mask_path = os.path.join(masks_dir, fname.rsplit(".", 1)[0]+"_" + ".png")
        cv2.imwrite(mask_path, mask)
        print(f"[INFO] Masque noir généré : {mask_path}")


# Exemple d’utilisation
import os
import cv2
import numpy as np

def convert_png_to_jpeg(images_dir, output_dir, quality=85):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".png"):
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Impossible de lire {img_path}")
            continue

        # Nom de sortie en JPEG
        out_name = os.path.splitext(fname)[0]+"_"+".jpg"
        out_path = os.path.join(output_dir, out_name)

        # Sauvegarde avec compression JPEG réaliste
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"[INFO] Image convertie : {out_path}")

# Exemple d’utilisation
images_dir = "/home/ilyassechaouki/DOCTMP/positive"
output_dir = "/home/ilyassechaouki/DOCTMP/ID dataset/Images pos"

#convert_png_to_jpeg(images_dir, output_dir, quality=100)

import os
import cv2
import numpy as np

def create_null_masks(images_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Impossible de lire {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # nom du mask (même base que l’image)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, f"{base}.png")

        cv2.imwrite(mask_path, mask)
        print(f"[INFO] Masque créé : {mask_path}")


# Exemple d’utilisation
images_dir = "/home/ilyassechaouki/DOCTMP/ID dataset/Images pos"
masks_dir  = "/home/ilyassechaouki/DOCTMP/ID dataset/Masks pos"
create_null_masks(images_dir, masks_dir)
import os
import json
import cv2
import numpy as np

def create_masks_from_json(images_dir, json_path, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)

    # Charger le JSON
    with open(json_path, "r") as f:
        annotations = json.load(f)

    for img_name, ann in annotations.items():
        # Forcer extension JPG
        base = os.path.splitext(img_name)[0]
        img_name_jpg = base + ".jpg"

        img_path = os.path.join(images_dir, img_name_jpg)

        if not os.path.exists(img_path):
            print(f"[WARN] Image {img_name_jpg} non trouvée")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Impossible de lire {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # dessiner les bbox des champs modifiés
        for key, value in ann.items():
            if isinstance(value, dict) and "bbox" in value:
                x1, y1, x2, y2 = value["bbox"]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Sauvegarder masque en PNG (binaire, sans toucher aux JPG)
        mask_path = os.path.join(masks_dir, base + ".png")
        cv2.imwrite(mask_path, mask)
        print(f"[INFO] Masque créé : {mask_path}")

# Exemple d’utilisation
images_dir = "/home/ilyassechaouki/DOCTMP/ID dataset/Images"
json_path  = "/home/ilyassechaouki/DOCTMP/esp_textfieldreplacement_annotation.json"
masks_dir  = "/home/ilyassechaouki/DOCTMP/ID dataset/Masks"

#create_masks_from_json(images_dir, json_path, masks_dir)



