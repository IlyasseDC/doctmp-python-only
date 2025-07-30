import lmdb
import os
import pickle

lmdb_path = './DocTamperV1-TrainingSet'
output_path = f'pks/DocTamperV1-TrainingSet_95.pk'
minq = 90  # ou autre valeur passée en paramètre
num_samples = 170000  # Nombre d’images à inclure

os.makedirs('pks', exist_ok=True)

env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    n_samples_total = int(txn.get('num-samples'.encode('utf-8')))
    print(f"LMDB contains {n_samples_total} images.")

record = {}

for i in range(min(num_samples, n_samples_total)):
    record[i] = [minq]  # On suppose que chaque image est compressée une seule fois

with open(output_path, 'wb') as f:
    pickle.dump(record, f)

print(f"[✓] Pickle saved to: {output_path} with {len(record)} entries.")
