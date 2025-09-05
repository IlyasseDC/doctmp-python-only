import os
import pickle
import numpy as np

# Dossier contenant les images utilisées dans l'entraînement
input_dir = './input'
output_pkl = 'q_tables/qtable_all_ones.pkl'

# Q-table 8x8 avec tous les indices à 1
fake_qtable = np.ones((8, 8), dtype=np.int64)

# Construire le dictionnaire
qtable_dict = {}
for fname in sorted(os.listdir(input_dir)):
    if fname.endswith('.jpg'):
        qtable_dict[fname] = fake_qtable.copy()

# Sauvegarder
os.makedirs('q_tables', exist_ok=True)
with open(output_pkl, 'wb') as f:
    pickle.dump(qtable_dict, f)

print(f"Fichier généré : {output_pkl} avec {len(qtable_dict)} entrées.")
