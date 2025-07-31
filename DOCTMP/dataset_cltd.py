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

class TamperDatasetCLTD(Dataset):
    def __init__(self, lmdb_path, record_path, qt_table_path, T=8192, is_train=True, fixed_quality=95, max_readers=64):
        self.env = lmdb.open(lmdb_path, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples

        self.record = pickle.load(open(record_path, 'rb'))
        self.qtables = pickle.load(open(qt_table_path, 'rb'))
        self.T = T
        self.step = 0
        self.is_train = is_train
        self.fixed_quality = fixed_quality  # ← nouveau

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.totsr = ToTensorV2()

    def __len__(self):
        return self.max_nums

    def __getitem__(self, idx):
        self.step += 1

        # Quality factor for JPEG compression
        if self.is_train:
            B1 = max(5, 100 - self.step / self.T)
            quality = int(random.uniform(B1, 100))
        else:
            quality = self.fixed_quality  # ← maintenant utilisé

        with self.env.begin(write=False) as txn:
            img_key = f'image-{idx+1:09d}'
            imgbuf = txn.get(img_key.encode('utf-8'))
            if imgbuf is None:
                raise IndexError(f"Image not found for key {img_key}")  

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf).convert('RGB')

            lbl_key = f'label-{idx+1:09d}'
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            if lblbuf is None:
                raise IndexError(f"Label not found for key {lbl_key}")

            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            mask = (mask != 0).astype(np.uint8)
            mask_tensor = self.totsr(image=mask.copy())['image'].long()

        # Apply compression to grayscale version and compute DCT
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            im_l = im.convert("L")
            im_l.save(tmp.name, format="JPEG", quality=quality)
            jpeg_gray = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            dct = cv2.dct(jpeg_gray)
            im_compressed_rgb = Image.open(tmp.name).convert("RGB")

        # Get Q-table from final record (could be ignored if fixed quality is used)
        record = self.record[idx]
        q_used = record[-1] if isinstance(record, (list, tuple)) else 95
        q_tensor = self.qtables.get(q_used, torch.zeros((64,), dtype=torch.long))  
        



        return {
            'image': self.normalize(im_compressed_rgb),
            'image_compressed': transforms.ToTensor()(im_compressed_rgb),
            'label': mask_tensor,
            'rgb': torch.from_numpy(np.clip(np.abs(dct), 0, 20)).float(),
            'q': q_tensor,
            'i': q_used  # on peut aussi y mettre q_used si besoin
        }
