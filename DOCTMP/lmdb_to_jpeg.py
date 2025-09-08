import lmdb
import os
import six
import cv2
import numpy as np

def export_images_and_masks_from_lmdb(lmdb_path, out_dir, n_export=100):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "Images")
    lbl_dir = os.path.join(out_dir, "Labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Ouvrir LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode('utf-8')))
        n_export = min(n_export, n_samples)

        print(f"LMDB contient {n_samples} images, export des {n_export} premières...")

        for idx in range(n_export):
            # --- Image JPEG brute ---
            img_key = f'image-{idx+1:09d}'.encode('utf-8')
            imgbuf = txn.get(img_key)
            if imgbuf is None:
                print(f"[WARN] Clé {img_key} introuvable")
                continue

            img_path = os.path.join(img_dir, f'img_{idx+1:05d}.jpg')
            with open(img_path, 'wb') as f:
                f.write(imgbuf)

            # --- Masque binaire ---
            lbl_key = f'label-{idx+1:09d}'.encode('utf-8')
            lblbuf = txn.get(lbl_key)
            if lblbuf is not None:
                mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                # Sauvegarde en PNG (0/255)
                lbl_path = os.path.join(lbl_dir, f'img_{idx+1:05d}.png')
                cv2.imwrite(lbl_path, mask)

    print(f"✅ Export terminé : {n_export} images dans {img_dir}, masques dans {lbl_dir}")

# Exemple d’utilisation
if __name__ == "__main__":
    lmdb_path = "DocTamperV1-FCD"   # ton dossier LMDB
    out_dir   = "exported_dataset"  # dossier de sortie
    export_images_and_masks_from_lmdb(lmdb_path, out_dir, n_export=100)
