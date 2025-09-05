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
import jpegio as jio


def dct_to_volume(dct_array, T=20):
    """
    Convertit une matrice DCT 2D (H, W) en un volume one-hot (T+1, H, W)
    Chaque canal correspond à la fréquence absolue DCT clippée dans [0, T]
    """
    dct_array = np.clip(np.abs(dct_array), 0, T).astype(np.int32)
    H, W = dct_array.shape
    vol = np.zeros((T + 1, H, W), dtype=np.float32)
    for t in range(T + 1):
        vol[t] = (dct_array == t).astype(np.float32)
    return vol

class TamperDatasetCLTD(Dataset):
    def __init__(self, lmdb_path, record_path, qt_table_path, T=8192, is_train=True, fixed_quality=95, max_readers=64):
        self.env = lmdb.open(lmdb_path, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples - 1

        self.record = pickle.load(open(record_path, 'rb'))
        self.qtables = pickle.load(open(qt_table_path, 'rb'))
        self.T = T
        self.step = 0
        self.is_train = is_train
        self.fixed_quality = fixed_quality

        self.class_weights = torch.tensor([1.0, 5.0])  # authentic / tampered

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.totsr = ToTensorV2()

    def __len__(self):
        return self.max_nums
    def __getitem__(self, idx):
        self.step += 1

        # Quality factor
        if self.is_train:
            B1 = max(5, 100 - self.step / self.T)
            quality = int(random.uniform(B1, 100))
        else:
            quality = self.fixed_quality

        with self.env.begin(write=False) as txn:
            # Read image
            img_key = f'image-{idx+1:09d}'
            imgbuf = txn.get(img_key.encode('utf-8'))
            if imgbuf is None:
                raise IndexError(f"Image not found for key {img_key}")  
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf).convert('RGB')

            # Read label
            lbl_key = f'label-{idx+1:09d}'
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            if lblbuf is None:
                raise IndexError(f"Label not found for key {lbl_key}")
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            mask = (mask != 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask.copy()).long().unsqueeze(0)

        # Compress grayscale image and compute DCT
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            im_l = im.convert("L")
            im_l.save(tmp.name, format="JPEG", quality=quality)
            jpeg_gray = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            im_compressed_rgb = Image.open(tmp.name).convert("RGB")
            # Lire les coefficients DCT via jpegio

            jpeg_gray = cv2.resize(jpeg_gray, im_compressed_rgb.size[::-1])
            dct_y = cv2.dct(jpeg_gray)

        # Convert DCT matrix to 21-channel volume
        dct_vol = dct_to_volume(dct_y, T=20)

        # Get quantization table
        record = self.record[idx]
        q_used = record[-1] if isinstance(record, (list, tuple)) else 95
        q_tensor = self.qtables.get(q_used, torch.zeros((64,), dtype=torch.long))  

        return {
            'image': self.normalize(im_compressed_rgb),
            'image_raw': transforms.ToTensor()(im_compressed_rgb),  # <-- AJOUT
            'label': mask_tensor,
            'rgb': torch.from_numpy(dct_vol).float(),
            'q': q_tensor,
            'i': q_used
        }

