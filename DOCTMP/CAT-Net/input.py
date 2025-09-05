import os
import lmdb
import six
import numpy as np
import cv2
from PIL import Image

# === Paramètres ===
lmdb_path = '/home/ilyassechaouki/DOCTMP/DocTamperV1-TrainingSet'  # chemin vers le dossier LMDB DTD
output_dir = './input'
os.makedirs(output_dir, exist_ok=True)

# === Ouvrir LMDB ===
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

# === Extraire les 100 premières images ===
with env.begin(write=False) as txn:
    for index in range(100):
        key = f'image-{index:09d}'.encode('utf-8')
        imgbuf = txn.get(key)

        if imgbuf is None:
            print(f"Image absente pour l'index {index}")
            continue

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        # Charger avec PIL (meilleure compatibilité JPEG)
        im = Image.open(buf).convert("RGB")

        # Sauvegarder avec nom indexé
        im.save(os.path.join(output_dir, f'dtd_train_{index:03d}.jpg'), 'JPEG')
