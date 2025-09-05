# dataset_cltd.py

import os
import cv2
import lmdb
import torch
import pickle
import numpy as np
import six
import random
import tempfile
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torchvision


class TamperDatasetCLTD(Dataset):
    def __init__(self, lmdb_path, record_path, qt_table_path, T=8192, is_train=True, max_readers=64):
        self.env = lmdb.open(lmdb_path, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples

        self.record = pickle.load(open(record_path, 'rb'))
        raw_qtables = pickle.load(open(qt_table_path, 'rb'))
        self.qtables = {k: torch.LongTensor(v) for k, v in raw_qtables.items()}

        self.T = T
        self.step = 0
        self.is_train = is_train

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                            std=(0.229, 0.224, 0.225))
        ])

        self.totsr = ToTensorV2()

    def __len__(self):
        return self.max_nums

    def __getitem__(self, idx):
        self.step += 1

        with self.env.begin(write=False) as txn:
            img_key = f'image-{idx+1:09d}'
            lbl_key = f'label-{idx+1:09d}'

            imgbuf = txn.get(img_key.encode('utf-8'))
            lblbuf = txn.get(lbl_key.encode('utf-8'))

            if imgbuf is None or lblbuf is None:
                raise IndexError(f"Missing image or label at index {idx}")

            # Lire l'image (niveau de gris au départ)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf).convert("L")

            # Lire le masque
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            mask = (mask != 0).astype(np.uint8)
            mask_tensor = self.totsr(image=mask.copy())['image'].long()

            # Lire les niveaux de qualité
            # Lire les niveaux de qualité de compression
            record = self.record[idx]

            # Supporte record comme numpy array, torch.Tensor, list ou un seul entier
            if isinstance(record, (list, tuple)):
                q_levels = [int(q) for q in record]
            elif isinstance(record, np.ndarray):
                q_levels = [int(q) for q in record.tolist()]
            elif isinstance(record, torch.Tensor):
                q_levels = [int(q.item()) for q in record] if record.numel() > 1 else [int(record.item())]
            else:
                q_levels = [int(record)]


            # Appliquer successivement les compressions JPEG
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                for q in q_levels:
                    im.save(tmp.name, format="JPEG", quality=q)
                    im = Image.open(tmp.name).convert("L")


                jpeg_gray = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                dct = cv2.dct(jpeg_gray)

        # Finaliser : RGB + normalisation + Q-table
        im_rgb = im.convert("RGB")
        img_tensor = self.toctsr(im_rgb)

        last_q = int(q_levels[-1])
        q_tensor = self.qtables.get(last_q, torch.zeros((64,), dtype=torch.long))

        return {
            'image': img_tensor,
            'label': mask_tensor,
            'rgb': torch.from_numpy(np.clip(np.abs(dct), 0, 20)).float(),
            'q': q_tensor,
            'i': last_q
        }

