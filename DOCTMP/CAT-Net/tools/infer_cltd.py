import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataset_cltd import TamperDatasetCLTD
from lib import models
from lib.config import config, update_config
from lib.core.criterion import CrossEntropy
from lib.utils.utils import FullModel
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with CAT-Net on CLTD')
    parser.add_argument('--cfg', required=True, help='Path to config YAML')
    parser.add_argument('--model', required=True, help='Path to trained model .pth.tar')
    parser.add_argument('--lmdb', required=True, help='Path to LMDB folder')
    parser.add_argument('--record', required=True, help='Path to record.pkl')
    parser.add_argument('--qtables', required=True, help='Path to qtables.pkl')
    parser.add_argument('--save-dir', default='output/infer_CLTD_3', help='Where to save results')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality factor to simulate')
    args, rest = parser.parse_known_args()
    args.opts = rest 
    update_config(config, args)

    return args

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = TamperDatasetCLTD(
        lmdb_path=args.lmdb,
        record_path=args.record,
        qt_table_path=args.qtables,
        is_train=False,
        fixed_quality=args.quality
    )

    from torch.utils.data import Subset

    subset_indices = list(range(min(100, len(dataset))))  # max 1000 éléments
    subset_dataset = Subset(dataset, subset_indices)

    dataloader = torch.utils.data.DataLoader(
        subset_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )


    # Loss
    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                             weight=dataset.class_weights).to(device)

    # Model
    from lib.models.network_CAT import get_seg_model
    model = get_seg_model(config)

    model = FullModel(model, criterion)
    checkpoint = torch.load(args.model, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[new_k] = v
    model.model.load_state_dict(new_state_dict)

    model = nn.DataParallel(model).to(device)
    model.eval()

    # Accumulate predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            image = batch['image'].to(device)       # RGB (normalized)
            dctvol = batch['rgb'].to(device)        # DCT volume
            qtable = batch['q'].to(device).unsqueeze(0)  # batch=1
            label = batch['label'].long().to(device).squeeze(1)  # [1, H, W]

            # Input for the model
            x = torch.cat([image, dctvol], dim=1)

            # Dummy forward to match shapes
            _, dummy_output = model(x, label, qtable)
            if label.shape[-2:] != dummy_output.shape[-2:]:
                label = F.interpolate(label.unsqueeze(1).float(), size=dummy_output.shape[-2:], mode="nearest").long().squeeze(1)

            # Real inference
            _, pred = model(x, label, qtable)  # [1, C, H, W]
            pred = torch.squeeze(pred, 0)      # [C, H, W]
            pred = F.softmax(pred, dim=0)[1]   # Class 1 prob
            pred_np = pred.cpu().numpy()

            # Resize to original size
            image_vis = batch['image_raw'][0].permute(1, 2, 0).numpy()
            image_vis = (image_vis * 255).astype(np.uint8)
            pred_resized = cv2.resize(pred_np, (image_vis.shape[1], image_vis.shape[0]))

            # Binarize prediction
            pred_bin = (pred_resized > 0.2).astype(np.uint8)

            # Ground truth mask
            gt_mask = batch['label'][0, 0].cpu().numpy().astype(np.uint8)
            if pred_bin.shape != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_bin.shape[1], pred_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Flatten for metrics
            all_preds.extend(pred_bin.flatten().tolist())
            all_labels.extend(gt_mask.flatten().tolist())

            # Visualisation
            heatmap = np.uint8(255 * pred_resized)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_vis, 0.6, heatmap_color, 0.4, 0)

            # Save images
            filename = f"image_{i:04d}"
            cv2.imwrite(os.path.join(args.save_dir, f"{filename}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            np.save(os.path.join(args.save_dir, f"{filename}_pred.npy"), pred_resized)

            # Plot
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(image_vis)
            axs[0].set_title("Image")
            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("GT Mask")
            axs[2].imshow(pred_resized, cmap='jet', vmin=0, vmax=1)
            axs[2].set_title("Pred CAT-Net")
            axs[3].imshow(overlay)
            axs[3].set_title("Overlay")

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f"{filename}_vis.png"))
            plt.close()

    # Compute global metrics
    iou = jaccard_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n=== Evaluation Metrics ===")
    print(f"IoU:       {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([{
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }])
    df_metrics.to_csv(os.path.join(args.save_dir, 'metrics.csv'), index=False)

if __name__ == '__main__':
    main()
