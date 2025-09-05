import os
import shutil

def copy_and_rename_masks(src_dir, dst_dir, offset=2000):
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".png")])

    for f in files:
        old_index = int(os.path.splitext(f)[0])  # ex: 2000
        new_index = old_index - offset           # ex: 2000 - 2000 = 0

        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f"{new_index}.png")

        shutil.copy(src_path, dst_path)
        print(f"[INFO] Copié {f} → {new_index}.png")

    print("[DONE] Copie + renommage terminé")


# Exemple d’utilisation
src_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/SCD_exported/Labels"
dst_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/SCD_exported/RenamedLabels"

#copy_and_rename_masks(src_dir, dst_dir, offset=18000)

import os

def rename_images_and_labels(images_dir, labels_dir, output_images, output_labels):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # on prend le nom de base sans suffixe
    img_bases = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")}
    lbl_bases = {os.path.splitext(f)[0].replace("_mask", ""): f for f in os.listdir(labels_dir) if f.lower().endswith(".png")}

    # intersection (paires valides)
    common = sorted(set(img_bases.keys()) & set(lbl_bases.keys()))

    for idx, base in enumerate(common):
        img_src = os.path.join(images_dir, img_bases[base])
        lbl_src = os.path.join(labels_dir, lbl_bases[base + "_mask"] if base + "_mask" in lbl_bases else lbl_bases[base])

        img_dst = os.path.join(output_images, f"{idx}.jpg")
        lbl_dst = os.path.join(output_labels, f"{idx}.png")

        os.rename(img_src, img_dst)
        os.rename(lbl_src, lbl_dst)

        print(f"[INFO] {img_bases[base]} + {lbl_bases.get(base + '_mask', lbl_bases[base])} → {idx}.jpg / {idx}.png")

    print(f"[DONE] {len(common)} paires renommées")


# Exemple d’utilisation
images_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/KID dataset/Images"
labels_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/KID dataset/Labels"

output_images = "/home/ilyassechaouki/DOCTMP/Internal dataset/KID dataset/ImagesR"
#output_labels = "/home/ilyassechaouki/DOCTMP/Internal dataset/KID dataset/LabelsR"

#rename_images_and_labels(images_dir, labels_dir, output_images, output_labels)

import os
import shutil

def rename_id_dataset(images_dir, masks_dir, output_images, output_masks):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    # dictionnaires base_name → fichier
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if f.lower().endswith(".png")}

    # intersection pour ne garder que les paires valides
    common = sorted(set(img_files.keys()) & set(mask_files.keys()))

    for idx, base in enumerate(common):
        img_src = os.path.join(images_dir, img_files[base])
        mask_src = os.path.join(masks_dir, mask_files[base])

        img_dst = os.path.join(output_images, f"{idx}.jpg")
        mask_dst = os.path.join(output_masks, f"{idx}.png")

        shutil.copy(img_src, img_dst)
        shutil.copy(mask_src, mask_dst)

        print(f"[INFO] {base} → {idx}.jpg / {idx}.png")

    print(f"[DONE] {len(common)} paires copiées et renommées")


# Exemple d’utilisation
images_dir = "/home/ilyassechaouki/DOCTMP/Internal dataset/ID dataset/Images"
masks_dir  = "/home/ilyassechaouki/DOCTMP/Internal dataset/ID dataset/Masks"

output_images = "/home/ilyassechaouki/DOCTMP/Internal dataset/ID dataset/ImagesR"
output_masks  = "/home/ilyassechaouki/DOCTMP/Internal dataset/ID dataset/MasksR"

#rename_id_dataset(images_dir, masks_dir, output_images, output_masks)

import os
import random
import shutil

def merge_datasets(datasets_root, output_dir, n_per_dataset=500, seed=42):
    random.seed(seed)
    images_out = os.path.join(output_dir, "Images")
    labels_out = os.path.join(output_dir, "Labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    dataset_names = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
    counter = 0

    for dname in dataset_names:
        img_dir = os.path.join(datasets_root, dname, "Images")
        lbl_dir = os.path.join(datasets_root, dname, "Labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"[WARN] {dname} ignoré (pas de Images/ ou Labels/)")
            continue

        # fichiers disponibles (les noms sont déjà 0,1,2... donc facile)
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
        lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith(".png")])

        # on garde juste la partie commune
        n_available = min(len(img_files), len(lbl_files))
        sample_indices = random.sample(range(n_available), min(n_per_dataset, n_available))

        for idx in sample_indices:
            img_src = os.path.join(img_dir, f"{idx}.jpg")
            lbl_src = os.path.join(lbl_dir, f"{idx}.png")

            img_dst = os.path.join(images_out, f"{counter}.jpg")
            lbl_dst = os.path.join(labels_out, f"{counter}.png")

            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

            counter += 1

        print(f"[INFO] {dname}: {len(sample_indices)} paires copiées")

    print(f"[DONE] Total {counter} paires fusionnées dans {output_dir}")


import lmdb
import six
import os
import cv2
import numpy as np
from PIL import Image

def export_range_from_lmdb(lmdb_path, img_out_dir, lbl_out_dir, start, end, counter=0):
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode('utf-8')).decode())
        print(f"[INFO] {lmdb_path}: {n_samples} échantillons disponibles")

        for idx in range(start, end):
            if idx >= n_samples:
                print(f"[WARN] Index {idx} hors limite ({n_samples}), stop.")
                break

            # --- image ---
            img_key = f'image-{idx:09d}'.encode('utf-8')
            imgbuf = txn.get(img_key)
            if imgbuf is None:
                print(f"[WARN] Image manquante à l'index {idx}")
                continue
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
            im.save(os.path.join(img_out_dir, f"{counter}.jpg"))

            # --- label ---
            lbl_key = f'label-{idx:09d}'.encode('utf-8')
            lblbuf = txn.get(lbl_key)
            if lblbuf is None:
                print(f"[WARN] Label manquant à l'index {idx}")
                continue
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            cv2.imwrite(os.path.join(lbl_out_dir, f"{counter}.png"), mask)

            counter += 1

    env.close()
    return counter

def export_fcd_scd(fcd_lmdb, scd_lmdb, output_root):
    # dossiers de sortie
    train_img = os.path.join(output_root, "train", "Images")
    train_lbl = os.path.join(output_root, "train", "Labels")
    test_img  = os.path.join(output_root, "test", "Images")
    test_lbl  = os.path.join(output_root, "test", "Labels")

    # === Export TRAIN ===
    counter = 0
    counter = export_range_from_lmdb(fcd_lmdb, train_img, train_lbl, 0, 1000, counter)
    counter = export_range_from_lmdb(scd_lmdb, train_img, train_lbl, 0, 2000, counter)
    print(f"[OK] Train exporté : {counter} paires")

    # === Export TEST ===
    counter = 0
    counter = export_range_from_lmdb(fcd_lmdb, test_img, test_lbl, 1000, 1500, counter)
    counter = export_range_from_lmdb(scd_lmdb, test_img, test_lbl, 2000, 2500, counter)
    print(f"[OK] Test exporté : {counter} paires")

if __name__ == "__main__":
    fcd_lmdb = "/home/ilyassechaouki/DOCTMP/DocTamperV1-FCD"
    scd_lmdb = "/home/ilyassechaouki/DOCTMP/DocTamperV1-SCD"
    output_root = "/home/ilyassechaouki/DOCTMP/FCDSCD_exported"

    export_fcd_scd(fcd_lmdb, scd_lmdb, output_root)
