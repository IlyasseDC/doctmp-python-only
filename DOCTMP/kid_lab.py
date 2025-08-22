import os
import cv2
import numpy as np

def create_null_masks(root_dir, masks_root):
    for subdir, _, files in os.walk(root_dir):
        # Rebuild same folder structure under masks_root
        rel_path = os.path.relpath(subdir, root_dir)
        masks_subdir = os.path.join(masks_root, rel_path)
        os.makedirs(masks_subdir, exist_ok=True)

        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                img_path = os.path.join(subdir, fname)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"[WARN] Could not read {img_path}")
                    continue

                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                mask_name = os.path.splitext(fname)[0] + "_mask.png"
                mask_path = os.path.join(masks_subdir, mask_name)

                cv2.imwrite(mask_path, mask)
                print(f"[INFO] Mask created: {mask_path}")

# Example usage
images_root = "DL_024_U37"
masks_root  = "DL_024_U37_masks"

create_null_masks(images_root, masks_root)
