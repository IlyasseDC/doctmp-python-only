import os
import lmdb
import cv2
import numpy as np

def export_lmdb(lmdb_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Labels"), exist_ok=True)

    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        idx = 0
        for key, value in cursor:
            # Décoder l'entrée (image ou masque)
            arr = np.frombuffer(value, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"[WARN] Impossible de décoder {key}")
                continue

            # Petite logique : supposons que les clés indiquent image vs masque
            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            if "label" in key_str.lower():
                out_path = os.path.join(output_dir, "Labels", f"{idx}.png")
                cv2.imwrite(out_path, img)
            if "image" in key_str.lower() or "images" in key_str.lower():
                out_path = os.path.join(output_dir, "Images", f"{idx}.jpg")
                cv2.imwrite(out_path, img)

            idx += 1
            if idx % 100 == 0:
                print(f"[INFO] Exportés {idx} fichiers...")

    env.close()
    print(f"[DONE] Export terminé. {idx} fichiers écrits.")

# Exemple d’utilisation
lmdb_dir = "DocTamperV1-FCD"
output_dir = "FCD_exported"
export_lmdb(lmdb_dir, output_dir)
