import sys, os
import argparse
import shutil
import logging
import time
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, precision_score, recall_score

# Ajout du chemin du repo
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from lib import models
from lib.config import config, update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.utils.utils import FullModel
from Splicing.data.data_core import SplicingDataset as splicing_dataset
from project_config import dataset_paths

def main():
    # === CONFIG ===
    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=[
        'TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar',
        'TEST.FLIP_TEST', 'False',
        'TEST.NUM_SAMPLES', '0'
    ])
    update_config(config, args)

    # === CUDNN SETTINGS ===
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # === DATA ===
    test_dataset = splicing_dataset(
        crop_size=None, grid_crop=True,
        blocks=('RGB', 'DCTvol', 'qtable'),
        DCT_channels=1, mode='arbitrary',
        read_from_jpeg=True
    )
    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=1, pin_memory=False
    )

    # === CRITERION ===
    criterion = CrossEntropy(
        ignore_label=config.TRAIN.IGNORE_LABEL,
        weight=test_dataset.class_weights
    ).cuda()

    # === MODEL ===
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    model_file = config.TEST.MODEL_FILE
    assert os.path.exists(model_file), "Model file not found"

    print(f"=> Loading model from {model_file}")
    model = FullModel(model, criterion)
    checkpoint = torch.load(model_file)
    model.model.load_state_dict(checkpoint['state_dict'])
    print(f"Epoch loaded: {checkpoint['epoch']}")

    model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()
    model.eval()

    # === OUTPUT DIR ===
    save_dir = dataset_paths['SAVE_PRED']
    save_dir.mkdir(parents=True, exist_ok=True)

    # === EVALUATION STORAGE ===
    all_preds, all_labels = [], []

    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            return os.path.split(name)[-1]

    # === INFERENCE LOOP ===
    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            image = image.cuda()
            label = label.long().cuda()

            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            prob = F.softmax(pred, dim=0)[1]
            pred_np = prob.cpu().numpy()
            label_np = label.cpu().numpy().squeeze()

            all_preds.append(pred_np.flatten())
            all_labels.append(label_np.flatten())

            # === SAVE HEATMAP ===
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filepath = save_dir / filename
            try:
                width = pred_np.shape[1]
                fig = plt.figure(frameon=False)
                dpi = 40
                fig.set_size_inches(width / dpi, ((width * pred_np.shape[0])/pred_np.shape[1]) / dpi)
                sns.heatmap(pred_np, vmin=0, vmax=1, cbar=False, cmap='jet')
                plt.axis('off')
                plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig)
            except Exception as e:
                print(f"Error while saving heatmap for {filename}: {e}")

    # === EVALUATION ===
    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_preds)
    y_pred_bin = (y_scores >= 0.5).astype(np.uint8)

    iou = np.sum((y_pred_bin & y_true)) / np.sum((y_pred_bin | y_true))
    f1 = f1_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    print("\n====== BENCHMARK RESULTS - CAT FULL ======")
    print(f"IoU       : {iou:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    print(f"PR AUC    : {pr_auc:.4f}")

    # === SAVE RESULTS JSON ===
    results = {
        "IoU": float(iou),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "ROC AUC": float(roc_auc),
        "PR AUC": float(pr_auc)
    }
    with open("benchmark_CAT_full.json", "w") as f:
        json.dump(results, f, indent=4)

    # === SAVE PLOTS ===
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_CAT_full.png")

    plt.figure()
    plt.plot(recall_curve, precision_curve, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("pr_CAT_full.png")

if __name__ == '__main__':
    main()
