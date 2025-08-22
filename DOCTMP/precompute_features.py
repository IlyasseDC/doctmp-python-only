# precompute_features.py
import os
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T
import jpegio

def precompute_dataset(img_dir, lbl_dir, out_file, quality=100):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".png")])
    assert len(img_files) == len(lbl_files), "Images ≠ Labels"

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])

    dataset = []

    for idx, (img_f, lbl_f) in enumerate(zip(img_files, lbl_files)):
        img_path = os.path.join(img_dir, img_f)
        lbl_path = os.path.join(lbl_dir, lbl_f)

        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        w, h = img.size

        patches = []
        for top in range(0, h, 512):
            for left in range(0, w, 512):
                right = min(left + 512, w)
                bottom = min(top + 512, h)

                patch_img = img.crop((left, top, right, bottom))
                pad_img = Image.new('RGB', (512, 512), (0, 0, 0))
                pad_img.paste(patch_img, (0, 0))

                patch_lbl = lbl.crop((left, top, right, bottom))
                pad_lbl = Image.new('L', (512, 512), 0)
                pad_lbl.paste(patch_lbl, (0, 0))

                # --- JPEG features ---
                tmp_name = f"tmp_{idx}_{top}_{left}.jpg"
                pad_img.save(tmp_name, "JPEG", quality=quality)
                jpg = jpegio.read(tmp_name)
                os.remove(tmp_name)

                dct = np.ascontiguousarray(jpg.coef_arrays[0])
                qtb = np.ascontiguousarray(jpg.quant_tables[0].reshape(-1))


                img_t = to_tensor(pad_img).numpy()
                lbl_np = (np.array(pad_lbl, dtype=np.uint8) > 127).astype(np.int64)

                patches.append({
                    "image": img_t,
                    "label": lbl_np,
                    "dct": np.clip(np.abs(dct), 0, 20).astype(np.int64),
                    "qtb": qtb.astype(np.int64)
                })

        dataset.append(patches)
        if idx % 50 == 0:
            print(f"[{idx}/{len(img_files)}] processed")

    with open(out_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {out_file}")


if __name__ == "__main__":
    precompute_dataset(
        img_dir="/home/ilyassechaouki/DOCTMP/Internal_TrainingSet/Images",
        lbl_dir="/home/ilyassechaouki/DOCTMP/Internal_TrainingSet/Labels",
        out_file="./Internal_TrainingSet/precomputed.pkl",
        quality=100
    )
