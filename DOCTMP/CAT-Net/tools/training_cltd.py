import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from lib.config import config, update_config
from lib.utils.utils import create_logger
from losses import LovaszLoss, SoftCrossEntropyLoss
from lib.models.network_CAT import get_seg_model


from dataset_cltd import TamperDatasetCLTD
from metrics import IOUMetric
from utils import AverageMeter
import torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


def train_catnet_cltd(args):
    os.makedirs(args.save_ckpt_dir, exist_ok=True)

    if not hasattr(args, "opts"):
        args.opts = None
    update_config(config, args)


    logger, final_output_dir, _ = create_logger(config, args.cfg, 'train')
    logger.info(f"Training CAT-Net on DocTamper using config: {args.cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. DATASETS ===
    # 1. Charger le dataset complet (sans is_train / fixed_quality ici)
    full_dataset = TamperDatasetCLTD(args.lmdb_path, args.record_path, args.qt_path, T=args.T)
    # Ne conserver que les 12 000 premiers échantillons
    max_subset = 14000
    if len(full_dataset) > max_subset:
        from torch.utils.data import Subset
        full_dataset = Subset(full_dataset, list(range(max_subset)))


    # 2. Déterminer la taille 80/20
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # 3. Séparer les indices aléatoirement
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )

    # 4. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)


    # === 2. MODEL ===
    model = get_seg_model(config).to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)


    # === 3. LOSSES + OPTIMIZER ===
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    lovasz_loss = LovaszLoss(mode="multiclass")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    # === 4. TRAINING LOOP ===
    best_iou = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = AverageMeter()
        for batch in train_loader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            dct = batch['rgb'].to(device)
            qtb = batch['q'].unsqueeze(1).to(device)

            with autocast():
                x = torch.cat([image, dct], dim=1)
                output = model(x, qtb)
                # Adapter la taille du label à celle de la prédiction
                if label.shape[-2:] != output.shape[-2:]:
                    label = F.interpolate(label.float(), size=output.shape[-2:], mode="nearest").long()

                ce = 5 * ce_loss(output, label)
                lv = lovasz_loss(output.float(), label)
                loss = ce + lv

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + 1)

            train_loss_meter.update(loss.item())

        logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss_meter.avg:.4f}")

        # === 5. VALIDATION ===
        model.eval()
        iou = IOUMetric(2)
        val_loss_meter = AverageMeter()
        precisions, recalls = [], []

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                label = batch['label'].to(device)
                dct = batch['rgb'].to(device)
                qtb = batch['q'].unsqueeze(1).to(device)

                with autocast():
                    x = torch.cat([image, dct], dim=1)
                    output = model(x, qtb)
                    if label.shape[-2:] != output.shape[-2:]:
                        label = F.interpolate(label.float(), size=output.shape[-2:], mode="nearest").long()

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

        logger.info(f"[Val Epoch {epoch+1}] Loss={val_loss_meter.avg:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, mIoU={mean_iou:.4f}")
        logger.info(f"IoU details: {iou_vals}")

        # === 6. SAVE ===
        os.makedirs(args.save_ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(args.save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, ckpt_path)

        if iou_vals[1] > best_iou:
            best_iou = iou_vals[1]
            best_path = os.path.join(args.save_ckpt_dir, 'checkpoint-best.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, best_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/CAT_finetune.yaml')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_ckpt_dir', type=str, default='./checkpoints_catnet')
    parser.add_argument('--lmdb_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)
    parser.add_argument('--qt_path', type=str, required=True)
    parser.add_argument('--T', type=int, default=1000)
    args = parser.parse_args()

    train_catnet_cltd(args)
