import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

# Chargement OCR : dictionnaire {img_path: [[x,y,w,h], [x,y,w,h], ...]}
with open('ocr.pk','rb') as f:
    fpk = pickle.load(f)

# Compteurs
img_cnt = 0
max_cnt = 10  # nombre max de blocs remplacés avant de sauver une nouvelle image

# Création des dossiers de sortie
os.makedirs('tamp_imgs', exist_ok=True)
os.makedirs('tamp_masks', exist_ok=True)

for k, box in tqdm(fpk.items()):
    img1 = cv2.imread(k)  # image originale
    if img1 is None:
        print(f"[WARN] Impossible de lire {k}")
        continue

    h, w = img1.shape[:2]
    img2 = img1.copy()  # image source pour copier des blocs
    gt = np.zeros((h, w), dtype=np.uint8)  # masque binaire

    cnt = 0
    for bi1, b1 in enumerate(box):
        # zone cible
        x1, y1, w1, h1 = b1
        tgt = img1[y1:y1+h1, x1:x1+w1]

        for bi2, b2 in enumerate(box):
            if bi1 == bi2:  # on ne copie pas sur la même zone
                continue

            x2, y2, w2, h2 = b2
            src = img2[y2:y2+h2, x2:x2+w2]

            # redimensionner source à la taille de la cible
            if src.size == 0 or tgt.size == 0:
                continue
            src_resized = cv2.resize(src, (w1, h1))

            # coller dans l'image
            img1[y1:y1+h1, x1:x1+w1] = src_resized
            gt[y1:y1+h1, x1:x1+w1] = 255  # maj masque

            cnt += 1
            if cnt >= max_cnt:
                cv2.imwrite(f'tamp_imgs/{img_cnt}.jpg', img1)
                cv2.imwrite(f'tamp_masks/{img_cnt}.png', gt)
                cnt = 0
                img_cnt += 1
                img1 = cv2.imread(k)  # reset image
                gt = np.zeros((h, w), dtype=np.uint8)

    # sauvegarde finale
    cv2.imwrite(f'tamp_imgs/{img_cnt}.jpg', img1)
    cv2.imwrite(f'tamp_masks/{img_cnt}.png', gt)
    img_cnt += 1
