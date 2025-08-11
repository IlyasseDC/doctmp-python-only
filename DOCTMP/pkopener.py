import pickle
import torch

# Chemins vers les fichiers
qt_path = './pks/qt_table.pk'
record_path = './pks/DocTamperV1-TestingSet_75.pk'  

# Lire les quantization tables
with open(qt_path, 'rb') as f:
    qt_table = pickle.load(f)

print("Quantization Table (qt_table.pk):")
"""for q_level, qt_tensor in qt_table.items():
    print(f"Qualité {q_level}: Tensor shape {qt_tensor.shape}")
    print(qt_tensor)
    print("-" * 40)"""
qt_96 = qt_table[75]
print("Matrice de quantification pour la qualité 95 :")
print(qt_96)
qt_95 = qt_table[94]
print("Matrice de quantification pour la qualité 95 :")
print(qt_95)

print(len(qt_table))
"""
# Lire les enregistrements associés à un dataset (ex: index vers qualité JPEG)
with open(record_path, 'rb') as f:
    records = pickle.load(f)

print(f"\nNombre total d'entrées dans {record_path}: {len(records)}")
print("Quelques exemples d'enregistrements :")
for i in range(min(1000, len(records))):
    print(f"Index {i} : {records[i]}")"""
