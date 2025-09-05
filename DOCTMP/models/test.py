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
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
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
parser.add_argument('--minq', type=int, default=75)
args = parser.parse_args()

class TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=95, qtb=90, max_readers=64):
        self.envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums=self.nSamples
        self.minq = minq
        self.mode = mode
        with open('pks/qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        with open('pks/'+roots+'_%d.pk'%minq,'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
            H,W = mask.shape
            record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei>1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei>0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im = im.convert("L")
                if choicei>1:
                    im.save(tmp,"JPEG",quality=q2)
                    im = Image.open(tmp)
                if choicei>0:
                    im.save(tmp,"JPEG",quality=q1)
                    im = Image.open(tmp)
                im.save(tmp,"JPEG",quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct),0,20),
                'q':use_qtb,
                'i':q,
                'dct': torch.from_numpy(dct.astype(np.float32)),  # ajout du DCT brut
            }


from torch.utils.data import Subset

# Création du dataset
test_data_full = TamperDataset(args.data_root + args.lmdb_name, False, minq=args.minq)

# Création du subset 
subset_indices = list(range(1000, 1200))
test_data = Subset(test_data_full, subset_indices)




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
model = torch.nn.DataParallel(model)
import matplotlib.pyplot as plt

def eval_net_dtd(model, test_data, plot=False, device='cuda'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)
    LovaszLoss_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    ckpt = torch.load(args.pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    iou = IOUMetric(2)
    precisons = []
    recalls = []

    saved_count = 0  # compteur pour les visualisations

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1)):
            data = batch_samples['image'].to(device)
            target = batch_samples['label'].to(device)
            dct_coef = batch_samples['rgb'].to(device)
            dct_display = batch_samples['dct']  # DCT brut pour affichage
            qs = batch_samples['q'].unsqueeze(1).to(device)

            pred = model(data, dct_coef, qs)
            predt = pred.argmax(1)
            pred = pred.cpu().numpy()
            targt = target.squeeze(1)
            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))
            precisons.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())
            pred = np.argmax(pred, axis=1)
            iou.add_batch(pred, target.cpu().numpy())

            # === Sauvegarde des 5 premières visualisations ===
            for i in range(data.size(0)):
                if saved_count >= 5:
                    break

                input_img = data[i].detach().cpu().permute(1, 2, 0).numpy()
                input_img = (input_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.455, 0.406])
                input_img = np.clip(input_img, 0, 1)

                pred_mask = predt[i].detach().cpu().numpy()
                true_mask = target[i, 0].detach().cpu().numpy()

                # dct_display est un tensor 2D brut (non clippé)
                dct_img = dct_display[i].cpu().numpy()
                dct_img = np.abs(dct_img)

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(input_img)
                axs[0].set_title("Image")
                axs[0].axis("off")

                axs[1].imshow(pred_mask, cmap='gray')
                axs[1].set_title("Masque prédit")
                axs[1].axis("off")

                axs[2].imshow(true_mask, cmap='gray')
                axs[2].set_title("Masque réel")
                axs[2].axis("off")

                axs[3].imshow(np.log1p(dct_img), cmap='gray')
                axs[3].set_title("DCT (canal Y)")
                axs[3].axis("off")

                plt.tight_layout()
                plt.savefig(f"visu_example_{saved_count+1}.png")
                plt.close()

                saved_count += 1

        acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        f1 = 2 * precisons * recalls / (precisons + recalls + 1e-8)
        print('[val] iou:{}\npre:{}\nrec:{}\nf1:{}'.format(iu, precisons, recalls, f1))

eval_net_dtd(model, test_data)

