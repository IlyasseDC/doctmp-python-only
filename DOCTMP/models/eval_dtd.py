import os
import cv2
import lmdb
import torch
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
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataset_cltd import TamperDatasetCLTD


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./') 
parser.add_argument('--pth', type=str, default='dtd.pth')
parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-FCD')
parser.add_argument('--minq', type=int, default=75)
args = parser.parse_args()
"""
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
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

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
                im_array = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                dct = cv2.dct(im_array)
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct),0,20),
                'q':use_qtb,
                'i':q
            }
"""
# Paramètres cohérents
lmdb_path = './DocTamperV1-SCD'
record_path = './pks/DocTamperV1-TrainingSet_90.pk'
qt_path = './pks/qt_table.pk'
T = 8192  

# Charger le dataset complet
full_dataset = TamperDatasetCLTD(lmdb_path, record_path, qt_path, T=T, is_train=True)

# Split identique
train_indices = list(range(0, 1000))
val_indices   = list(range(1000, 1250))
heldout_indices = list(range(1300, 1999))

# Subset test (ex: validation comme dans training)
test_data = torch.utils.data.Subset(full_dataset, heldout_indices)


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = seg_dtd("", 2).to(device)

# Si on est sur CUDA, alors on active DataParallel
if device.type == 'cuda':
    model = nn.DataParallel(model)
def denormalize(tensor, mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def save_visualization(img, gt_mask, pred_mask, index, output_dir="vis_preds_FCD"):
    """
    Sauvegarde une figure avec l'image originale dénormalisée, le masque réel et le masque prédit.
    """
    import matplotlib.pyplot as plt
    import os

    img = denormalize(img.cpu()).numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Mask GT")
    axs[1].axis("off")
    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Mask Pred")
    axs[2].axis("off")

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{index}.png"))
    plt.close()


def eval_net_dtd(model, test_data, plot=False,device='cpu'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=2, num_workers=8, shuffle=False)
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
            dct_coef = dct_coef.long() 
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            pred = model(data,dct_coef,qs)             
            predt = pred.argmax(1)
            predt = pred.argmax(1)
            targt = target.squeeze(1)
            targt = targt.to(device)
            predt = predt.to(device)
            # compute metrics
            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))

            precisons.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())

            # add to IoU using the same shapes as in training
            iou.add_batch(predt.cpu().numpy(), targt.cpu().numpy())

            # Visualiser les 10 premières images du dataset
            max_vis = 50  # nombre total d'images à visualiser
            img_counter = batch_idx * data.shape[0]

            for b in range(data.shape[0]):
                global_index = batch_idx * data.shape[0] + b
                if global_index >= max_vis:
                    break

                img = data[b].cpu()
                gt = target[b, 0].cpu() if target.dim() == 4 else target[b].cpu()
                pd = predt[b].cpu()
                save_visualization(img, gt, pd, index=global_index)

        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))


eval_net_dtd(model, test_data) 



"""
ckpt = torch.load(args.pth, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])

def eval_net_dtd_thresh(model, test_data, plot=False, device='cpu', threshold=0.5, return_raw=False):
    train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)
    LovaszLoss_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)

    model.eval()
    
    iou = IOUMetric(2)
    precisions = []
    recalls = []

    if return_raw:
        all_probs = []
        all_targets = []

    with torch.no_grad():
        for batch_samples in tqdm(train_loader1):
            data = batch_samples['image'].to(device)
            target = batch_samples['label'].to(device)
            dct_coef = batch_samples['rgb'].long().to(device)
            qs = batch_samples['q'].unsqueeze(1).to(device)

            pred = model(data, dct_coef, qs)

            if pred.shape[1] == 2: 
                pred_prob = torch.softmax(pred, dim=1)[:, 1, :, :]
                predt = (pred_prob > threshold).long()

                if return_raw:
                    all_probs.append(pred_prob.cpu().numpy().flatten())
                    all_targets.append(target.cpu().numpy().flatten())
            else:  
                predt = pred.argmax(1)

            targt = target.squeeze(1)
            predt = predt.to(device)
            targt = targt.to(device)

            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))
            precisions.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())

            if pred.shape[1] == 2:
                pred_np = (pred[:, 1, :, :] > threshold).cpu().numpy().astype(np.uint8)
            else:
                pred_np = np.argmax(pred.cpu().numpy(), axis=1)

            iou.add_batch(pred_np, target.cpu().numpy())

    acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
    precision_mean = np.mean(precisions)
    recall_mean = np.mean(recalls)
    f1 = (2 * precision_mean * recall_mean) / (precision_mean + recall_mean + 1e-8)

    print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu, precision_mean, recall_mean, f1))

    if return_raw:
        y_score = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        return iu, precision_mean, recall_mean, f1, y_true, y_score
    else:
        return iu, precision_mean, recall_mean, f1


thresholds = np.linspace(0.1, 0.95, 10)

precisions = []
recalls = []
f1_scores = []
iou0 = []
iou1 = []

for thresh in thresholds:
    print(f"\n Threshold: {thresh:.2f}")
    iu, pre, rec, f1 = eval_net_dtd_thresh(model, test_data, device='cuda', threshold=thresh, return_raw=False)

    if isinstance(iu, np.ndarray) and len(iu) >= 2:
        iou0.append(iu[0])
        iou1.append(iu[1])
    else:
        iou0.append(np.nan)
        iou1.append(np.nan)

    precisions.append(pre)
    recalls.append(rec)
    f1_scores.append(f1)

# Préparer le DataFrame
df = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores,
    'IoU Class 0': iou0,
    'IoU Class 1': iou1
})

df_melted = df.melt(id_vars='Threshold', var_name='Metric', value_name='Score')

# Tracer avec Seaborn
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='o')
plt.plot(thresholds, f1_scores, label='F1 Score', marker='o')
plt.plot(thresholds, iou0, label='IoU Class 0', marker='o')
plt.plot(thresholds, iou1, label='IoU Class 1', marker='o')

plt.title('Evolution of Metrics vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('threshold_metrics_iou_curve_matplotlib_test.png')
plt.show()
"""
"""
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Sample a subset of the test data (for speed)
test_data = TamperDataset(args.data_root + args.lmdb_name, False, minq=args.minq)
from torch.utils.data import Subset

# Charger l'ensemble complet
full_test_data = TamperDataset(args.data_root + args.lmdb_name, False, minq=args.minq)

# Réduire à 1 000 échantillons
test_data = Subset(full_test_data, list(range(min(1000, len(full_test_data)))))

# Run evaluation once, collecting raw scores
iu, pre, rec, f1, y_true, y_score = eval_net_dtd_thresh(model, test_data, device='cpu', return_raw=True)


# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_Train.png")
plt.show()

# Precision-Recall Curve
prec, recs, _ = precision_recall_curve(y_true, y_score)
pr_auc = auc(recs, prec)

plt.figure()
plt.plot(recs, prec, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_Train.png")
plt.show()"""
"""
iou=IOUMetric(2)
precisons = []
recalls = []
with torch.no_grad():
    for batch_idx, batch_samples in enumerate(tqdm(test_loader)):
        pred = model(datas) # pred of shape (Batchsize, 2, img_Height, img_Width)
        pred_tamper = pred.argmax(1)
        target_ = target.squeeze(1)
        match = (pred_tamper*target_).sum((1,2))
        preds = pred_tamper.sum((1,2))
        target_sum = target_.sum((1,2))
        precisons.append((match/(preds+1e-8)).mean().item())
        recalls.append((match/target_sum).mean().item())
        pred=pred.cpu().data.numpy()
        pred= np.argmax(pred,axis=1)
        iou.add_batch(pred,target.cpu().data.numpy())
    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
    precisons = np.array(precisons).mean()
    recalls = np.array(recalls).mean()
    print('[val] iou:{} p:{} r:{} f:{}'.format(iu[1],precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))"""