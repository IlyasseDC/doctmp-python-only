import cv2
import numpy as np
import os

def extract_red_mask_filled(image_path, save_path=None):
    # Charger l'image en BGR
    img = cv2.imread(image_path)

    # Convertir en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Plages de rouge
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Masque binaire pour le rouge
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Nettoyage
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Nouvelle étape : remplir les contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)  # -1 => rempli

    if save_path:
        cv2.imwrite(save_path, filled_mask)

    return filled_mask


# Boucle sur le dataset
input_dir = "/home/ilyassechaouki/DOCTMP/labeled doc tamp dataset clean"
output_dir = "masks doc tamp dataset clean/"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg",".tif")):
        img_path = os.path.join(input_dir, fname)
        mask_path = os.path.join(output_dir, fname.rsplit(".",1)[0] + ".png")
        extract_red_mask_filled(img_path, mask_path)
