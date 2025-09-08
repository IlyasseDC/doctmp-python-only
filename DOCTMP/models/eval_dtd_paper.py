import os
import cv2
import lmdb
import torch
import jpegio
import numpy as np
import torch.nn as nn
import gc
import math
import time
import copy
import logging
import torch.optim as optim
import torch.distributed as dist
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from models.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
import albumentations as A
from models.dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./') # root to the dir of lmdb files
parser.add_argument('--pth', type=str, default='dtd.pth')
parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-FCD')
parser.add_argument('--minq', type=int, default=90)
args = parser.parse_args()
class TamperDataset(Dataset):
    def __init__(self, img_dir, lbl_dir=None):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg"))])
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])

        # --- Lire l'image et convertir en L (comme pipeline LMDB) ---
        im = Image.open(img_path).convert("L")

        # --- Charger le label (facultatif) ---
        mask = None
        if self.lbl_dir is not None:
            lbl_path = os.path.join(self.lbl_dir, os.path.splitext(self.img_files[index])[0] + ".png")
            if os.path.exists(lbl_path):
                mask_arr = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                mask_arr = (mask_arr > 127).astype(np.uint8)
                mask = self.totsr(image=mask_arr.copy())['image']

        # --- Lire directement le JPEG (DCT + Q-table réelles) ---
        jpg = jpegio.read(img_path)
        dct = jpg.coef_arrays[0].copy()
        qtb = torch.LongTensor(jpg.quant_tables[0])

        # --- Convertir en RGB pour le réseau ---
        im_rgb = im.convert("RGB")

        sample = {
            'image': self.toctsr(im_rgb),             # entrée réseau
            'rgb': np.clip(np.abs(dct), 0, 20),       # DCT
            'q': qtb,                                 # Q-table réelle
            'i': 0                                    # placeholder (plus de qualité forcée)
        }
        if mask is not None:
            sample['label'] = mask.long()

        return sample


from torch.utils.data import Subset

# Création du dataset
#test_data_full = TamperDataset(args.data_root + args.lmdb_name, False, minq=args.minq)

# Création du subset 
#subset_indices = list(range(1000, 1200))
#test_data = Subset(test_data_full, subset_indices)




def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

model = seg_dtd('',2).cuda()
#model = torch.nn.DataParallel(model)

def eval_net_dtd(model, test_data, plot=False,device='cuda'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    ckpt = torch.load(args.pth,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    iou=IOUMetric(2)
    precisons = []
    recalls = []
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            pred = model(data,dct_coef,qs)               
            predt = pred.argmax(1)
            pred=pred.cpu().data.numpy()
            targt = target.squeeze(1)
            matched = (predt*targt).sum((1,2))
            pred_sum = predt.sum((1,2))
            target_sum = targt.sum((1,2))
            precisons.append((matched/(pred_sum+1e-8)).mean().item())
            recalls.append((matched/target_sum).mean().item())
            pred = np.argmax(pred,axis=1)
            iou.add_batch(pred,target.cpu().data.numpy())
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))

#eval_net_dtd(model, test_data)
# Dataset basé sur LMDB

# Dataset basé sur JPEG (ex: Images/ et Labels/)
test_data_full_jpeg = TamperDataset("exported_dataset/Images", "exported_dataset/Labels")
test_data_jpeg = Subset(test_data_full_jpeg, list(range(0, 100)))

print("=== Résultats JPEG ===")
eval_net_dtd(model, test_data_jpeg)

