# build_lmdb_from_jpeg_dir.py
# Convertit un dossier d'images JPEG (+ labels PNG) en LMDB compatible avec TamperDataset (eval_dtd_paper.py)
# - Clés:  image-%09d  /  label-%09d
# - Métadonnée: num-samples
# - Fichier record: pks/<pks_name>_<q>.pk  (liste de [q] pour chaque échantillon)

import os
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

import lmdb


def _gather_pairs(images_dir: Path, labels_dir: Path):
    """
    Associe chaque image .jpg/.jpeg avec son label .png de même basename.
    Retourne une liste triée de tuples (img_path, lbl_path) par basename.
    """
    img_paths = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg"):
            img_paths[p.stem] = p

    pairs = []
    missing = []
    for stem, ip in img_paths.items():
        lp = labels_dir / f"{stem}.png"
        if lp.exists():
            pairs.append((ip, lp))
        else:
            missing.append(stem)

    pairs.sort(key=lambda x: x[0].stem)
    if missing:
        print(f"[WARN] {len(missing)} labels manquants (ignorés). Exemple: {missing[:5]}")

    return pairs


def _estimate_mapsize(pairs):
    """
    Estime la taille du LMDB (map_size) à partir de la somme des tailles des fichiers.
    On ajoute une marge.
    """
    total = 0
    for img, lbl in pairs:
        try:
            total += img.stat().st_size
        except Exception:
            pass
        try:
            total += lbl.stat().st_size
        except Exception:
            pass
    # Marge x1.5 + 100MB
    return int(total * 1.5) + 100_000_000


def build_lmdb(images_dir, labels_dir, lmdb_out, pks_name, q):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    lmdb_out   = Path(lmdb_out)
    lmdb_out.mkdir(parents=True, exist_ok=True)

    pairs = _gather_pairs(images_dir, labels_dir)
    n = len(pairs)
    if n == 0:
        raise ValueError("Aucune paire (image, label) trouvée. Vérifie les chemins et extensions.")

    map_size = _estimate_mapsize(pairs)
    print(f"[INFO] {n} paires trouvées. map_size ≈ {map_size/1e6:.1f} MB")

    env = lmdb.open(str(lmdb_out), map_size=map_size, subdir=True, lock=False, readahead=False, meminit=False)
    txn = env.begin(write=True)

    # Écrit le nombre total d'échantillons
    txn.put(b'num-samples', str(n).encode('utf-8'))

    for idx, (img_path, lbl_path) in enumerate(tqdm(pairs, desc="Écriture LMDB")):
        # Clés: image-000000000 / label-000000000 (index 0-based, comme dans ton Dataset)
        k_img = f"image-{idx:09d}".encode("utf-8")
        k_lbl = f"label-{idx:09d}".encode("utf-8")

        # Lire bytes bruts (on stocke les fichiers tels quels)
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        with open(lbl_path, "rb") as f:
            lbl_bytes = f.read()  # PNG binaire

        txn.put(k_img, img_bytes)
        txn.put(k_lbl, lbl_bytes)

        # Commit périodique pour éviter un gros txn en mémoire
        if (idx + 1) % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Commit final
    txn.commit()
    env.sync()
    env.close()
    print(f"[OK] LMDB écrit dans: {lmdb_out}")

    # Écrit le fichier record pks/<pks_name>_<q>.pk
    # TamperDataset lit: 'pks/'+roots+'_%d.pk' % minq  → liste de séquences de qualités
    # Ici on met une séquence d'une seule qualité [q] pour chaque échantillon.
    os.makedirs("pks", exist_ok=True)
    pks_path = Path("pks") / f"{pks_name}_{q}.pk"
    record = [[int(q)] for _ in range(n)]
    with open(pks_path, "wb") as f:
        pickle.dump(record, f)
    print(f"[OK] Record pks écrit: {pks_path}  (longueur={len(record)}, valeur=[{q}])")

    print("\n=== Rappel d'usage avec ton eval_dtd_paper.py ===")
    print(" - --data_root doit pointer vers le dossier parent où se trouve le LMDB (ou './')")
    print(f" - --lmdb_name doit être '{pks_name}' (le nom du dossier LMDB)")
    print(f" - --minq doit être {q} (pour correspondre au fichier pks/{pks_name}_{q}.pk)")
    print("Exemple:")
    print(f"python eval_dtd_paper.py --data_root './' --lmdb_name '{pks_name}' --pth chemin/vers/poids.pth --minq {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Dossier des images JPEG (ex: .../Images)")
    ap.add_argument("--labels_dir", required=True, help="Dossier des masques PNG (ex: .../Labels)")
    ap.add_argument("--lmdb_out",   required=True, help="Dossier de sortie LMDB (ex: ./DocTamperV1-FCD)")
    ap.add_argument("--pks_name",   default=None, help="Nom logique utilisé par eval (lmdb_name). Défaut: basename(lmdb_out)")
    ap.add_argument("--q",          type=int, default=100, help="Qualité 'record' utilisée dans pks (ex: 75/85/95)")
    args = ap.parse_args()

    pks_name = args.pks_name or Path(args.lmdb_out).name
    build_lmdb(args.images_dir, args.labels_dir, args.lmdb_out, pks_name, args.q)


if __name__ == "__main__":
    main()
