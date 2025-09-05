from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
import pickle
import torch
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize
import lmdb

class SplicingFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size=None, grid_crop=False, blocks=('RGB', 'DCTvol', 'qtable'),
                 mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=None):
        self.crop_size = crop_size
        self.grid_crop = grid_crop
        self.blocks = blocks
        self.mode = mode
        self.DCT_channels = DCT_channels
        self.read_from_jpeg = read_from_jpeg
        self.class_weights = torch.tensor(class_weight) if class_weight is not None else None

        self.img_paths = sorted([os.path.join('input', f) for f in os.listdir('input') if f.endswith('.jpg')])

        self.qtable_dict = pickle.load(open('/home/ilyassechaouki/DOCTMP/CAT-Net/q_tables/qtable_all_ones.pkl', 'rb'))

        self.lmdb_env = lmdb.open(
            '/home/ilyassechaouki/DOCTMP/DocTamperV1-TrainingSet',
            readonly=True, lock=False, readahead=False, meminit=False
        )

        self.transform = Compose([
            Resize(320, 320),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        name = os.path.basename(path)
        index = int(name.split('_')[-1].split('.')[0]) 

        # Charger image
        image = Image.open(path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0

        # GT mask
        with self.lmdb_env.begin(write=False) as txn:
            key = f'label-{index:09d}'.encode('utf-8')
            lblbuf = txn.get(key)
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)

        # Appliquer les transformations (resize, to tensor)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']          # [3, H, W]
        mask_tensor = transformed['mask'].long()     # [H, W]

        # Dummy DCTvol (21 canaux)
        _, H, W = image_tensor.shape
        dct_vol = torch.zeros((21, H, W), dtype=torch.float32)

        # Q-table
        qtable = self.qtable_dict.get(name, np.ones((8, 8)))
        qtable_tensor = torch.from_numpy(np.array(qtable, dtype=np.float32))  # [8,8]
        qtable_tensor = qtable_tensor.unsqueeze(0).repeat(21, 1, 1).contiguous()  # [21, 8, 8]
        qtable_tensor = qtable_tensor.float()  # assure le type

        # [21, 8, 8]
        print("qtable shape in batch:", qtable.shape)


        return image_tensor, dct_vol, mask_tensor, qtable_tensor
