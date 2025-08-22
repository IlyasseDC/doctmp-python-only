import os
import cv2

def rename_dataset(dataset_path):
    img_dir = os.path.join(dataset_path, "Images")
    lbl_dir = os.path.join(dataset_path, "Labels")

    img_files = sorted(os.listdir(img_dir))
    lbl_files = sorted(os.listdir(lbl_dir))

    if len(img_files) != len(lbl_files):
        print(f"[WARN] Nombre d'images != labels dans {dataset_path} ({len(img_files)} vs {len(lbl_files)})")

    # On suppose que l'ordre est correct
    for idx, (img_name, lbl_name) in enumerate(zip(img_files, lbl_files)):
        img_ext = os.path.splitext(img_name)[1].lower()
        lbl_ext = os.path.splitext(lbl_name)[1].lower()

        new_img_name = f"{idx}{img_ext if img_ext in ['.jpg','.jpeg','.png'] else '.jpg'}"
        new_lbl_name = f"{idx}{lbl_ext if lbl_ext in ['.png','.jpg','.jpeg'] else '.png'}"

        os.rename(os.path.join(img_dir, img_name), os.path.join(img_dir, new_img_name))
        os.rename(os.path.join(lbl_dir, lbl_name), os.path.join(lbl_dir, new_lbl_name))

    print(f"[INFO] {dataset_path} : {len(img_files)} fichiers renommés")


# Exemple d’utilisation
datasets_root = "Internal dataset"
for d in os.listdir(datasets_root):
    path = os.path.join(datasets_root, d)
    if os.path.isdir(path):
        rename_dataset(path)
