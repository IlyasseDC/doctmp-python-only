import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# === Paramètres ===
compression_level = 75
input_path = f"/home/ilyassechaouki/DOCTMP/pks/DocTamperV1-TestingSet_{compression_level}.pk"
output_path = f"distribution_qualite_moyenne_{compression_level}.png"

# === Étape 1 : Charger le fichier pickle correspondant au niveau de compression ===
with open(input_path, "rb") as f:
    data = pickle.load(f)

# === Étape 2 : Extraire les qualités minimales ===
min_qualities = []

for key, qlist in data.items():
    if len(qlist) > 0:
        min_qualities.append(np.mean(qlist))

# === Étape 3 : Afficher et sauvegarder la distribution ===
plt.figure(figsize=(10, 6))
plt.hist(min_qualities, bins=range(74, 101), color="cornflowerblue", edgecolor="black", alpha=0.8)
plt.title(f"Distribution des qualités JPEG moyennes (niveau {compression_level})")
plt.xlabel("Qualité JPEG minimale")
plt.ylabel("Nombre d'images")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()
