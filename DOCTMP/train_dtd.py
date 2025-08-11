import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from models.dtd import seg_dtd
from models.losses import LovaszLoss, SoftCrossEntropyLoss
from utils import AverageMeter, get_logger
from metrics import IOUMetric
from dataset_cltd import TamperDatasetCLTD
from models.dtd import *

def train_dtd(param):

    # Paramètres
    epochs = param['epochs']
    batch_size = param['batch_size']
    save_ckpt_dir = param['save_ckpt_dir']
    save_log_dir = param['save_log_dir']
    lmdb_path = param['lmdb_path']
    record_path = param['record_path']
    qt_path = param['qt_path']
    T = param.get('T')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(save_log_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_log_dir, f"train_dtd_{time.strftime('%Y%m%d_%H%M%S')}.log"))

    logger.info("Initializing model (via seg_dtd)...")
    model = seg_dtd("", n_class=2).to(device)

    if device.type == "cuda":
        model = nn.DataParallel(model)

    if param.get('load_ckpt') and os.path.exists(param['load_ckpt']):
        logger.info(f"Loading full checkpoint from {param['load_ckpt']}")
        checkpoint = torch.load(param['load_ckpt'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No pre-trained checkpoint loaded.")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    # Chargement dynamique du dataset par split
    logger.info("Preparing streaming datasets...")
    # 1. Charger tout le dataset
    full_dataset = TamperDatasetCLTD(lmdb_path, record_path, qt_path, T=T, is_train=True)

    # 2. Définir les indices
    train_indices = list(range(0, 1000))
    val_indices   = list(range(1000, 1500))

    # 3. Créer les sous-datasets
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set   = torch.utils.data.Subset(full_dataset, val_indices)
    # heldout_set = torch.utils.data.Subset(full_dataset, heldout_indices)  # pour plus tard si besoin

    # 4. Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    best_iou = 0
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss_meter = AverageMeter()
        for batch in train_loader:
            image = batch['image'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            dct = batch['rgb'].long().to(device, non_blocking=True)
            qtb = batch['q'].long().unsqueeze(1).to(device, non_blocking=True)

            with autocast():
                output = model(image, dct, qtb)
                ce = 5 * ce_loss(output, label)
            lovasz = lovasz_loss(output.float(), label)
            loss = ce + lovasz

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + 1)

            train_loss_meter.update(loss.item())

        logger.info(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss_meter.avg:.4f}")
        logger.info(f"[Epoch {epoch+1}] GPU Mem Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        model.eval()
        iou = IOUMetric(2)
        val_loss_meter = AverageMeter()
        precisions, recalls = [], []

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device, non_blocking=True)
                label = batch['label'].to(device, non_blocking=True)
                dct = batch['rgb'].long().to(device, non_blocking=True)
                qtb = batch['q'].long().unsqueeze(1).to(device, non_blocking=True)
                # Qualité JPEG moyenne du batch
                jpeg_qualities = batch['i']  
                avg_quality = sum(jpeg_qualities) / len(jpeg_qualities)
                print(jpeg_qualities)
                logger.info(f"[Epoch {epoch+1}] Batch avg JPEG quality: {avg_quality:.2f}")


                with autocast():
                    output = model(image, dct, qtb)
                    loss = 5 * ce_loss(output, label) + lovasz_loss(output, label)

                pred = output.argmax(1)
                targt = label.squeeze(1)
                matched = (pred * targt).sum((1, 2))
                pred_sum = pred.sum((1, 2))
                target_sum = targt.sum((1, 2))
                precisions.append((matched / (pred_sum + 1e-8)).mean().item())
                recalls.append((matched / target_sum).mean().item())

                iou.add_batch(pred.cpu().numpy(), label.cpu().numpy())
                val_loss_meter.update(loss.item())

        acc, acc_cls, iou_vals, mean_iou, fwavacc = iou.evaluate()
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        logger.info(f"[Validation] IoU: {iou_vals}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        ckpt_path = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)

        if iou_vals[1] > best_iou:
            best_iou = iou_vals[1]
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(save_ckpt_dir, 'checkpoint-best.pth'))
            logger.info(f"Best model saved at epoch {epoch+1} with IoU class 1 = {best_iou:.4f}")

if __name__ == "__main__":
    params = {
        'epochs': 20,
        'batch_size': 2,
        'save_ckpt_dir': './checkpoints',
        'save_log_dir': './logs',
        'lmdb_path': './DocTamperV1-FCD',
        'record_path': './pks/DocTamperV1-FCD_75.pk',
        'qt_path': './pks/qt_table.pk',
        'T': 40,
        'load_ckpt': './Weights/dtd_doctamper.pth'
    }
    train_dtd(params)