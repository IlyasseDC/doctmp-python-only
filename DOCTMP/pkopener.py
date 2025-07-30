import pickle
import torch

# Chemins vers les fichiers
qt_path = './pks/qt_table.pk'
record_path = './pks/DocTamperV1-TestingSet_90.pk'  # adapte si c'est un autre nom

# Lire les quantization tables
with open(qt_path, 'rb') as f:
    qt_table = pickle.load(f)

print("Quantization Table (qt_table.pk):")
for q_level, qt_tensor in qt_table.items():
    print(f"Qualité {q_level}: Tensor shape {qt_tensor.shape}")
    print(qt_tensor)
    print("-" * 40)

# Lire les enregistrements associés à un dataset (ex: index vers qualité JPEG)
with open(record_path, 'rb') as f:
    records = pickle.load(f)

print(f"\nNombre total d'entrées dans {record_path}: {len(records)}")
print("Quelques exemples d'enregistrements :")
for i in range(min(1000, len(records))):
    print(f"Index {i} : {records[i]}")
