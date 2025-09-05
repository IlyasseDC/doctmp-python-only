"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from skimage.transform import resize
from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

#from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from dataset_cltd import TamperDatasetCLTD as splicing_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    # args = parse_args()
    # Instead of using argparse, force these args:

    ## CHOOSE ##
    #args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    #args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    args = argparse.Namespace(cfg='experiments/CAT_light.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    #test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream
    test_dataset = splicing_dataset(
        lmdb_path = '/home/ilyassechaouki/DOCTMP/DocTamperV1-SCD',
        record_path = '/home/ilyassechaouki/DOCTMP/pks/DocTamperV1-SCD_90.pk',
        qt_table_path = '/home/ilyassechaouki/DOCTMP/pks/qt_table.pk',
        is_train = False,
        fixed_quality = 100
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda()

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))
    model = FullModel(model, criterion)
    checkpoint = torch.load(model_state_file)
    model.model.load_state_dict(checkpoint['state_dict'])
    print("Epoch: {}".format(checkpoint['epoch']))
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)


    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            name = os.path.split(name)[-1]
            return name

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image = batch['image'].cuda()
            dctvol = batch['rgb'].cuda()
            label = batch['label'].long().cuda().squeeze(1)
            qtable = batch['q'].cuda()

            
            #input_tensor = torch.cat([image, dctvol], dim=1)
           

            model.eval()
            _, pred = model(dctvol, label, qtable)

            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()
            

           


            filename = f"cltd_{index:05d}.png"
            filepath = dataset_paths['SAVE_PRED'] / filename

            try:
                # --- Reconstruct input image (de-normalized) ---
                img_rgb = batch['image_raw'][0].cpu().permute(1, 2, 0).numpy()
                img_rgb = np.clip(img_rgb, 0, 1)
                # Resize pred to match img_rgb
                pred_resized = resize(pred, img_rgb.shape[:2], order=1, preserve_range=True, anti_aliasing=True)

                # --- Display image + prediction overlay ---
                dpi = 100
                h, w = img_rgb.shape[:2]
                fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
                ax.imshow(img_rgb)  # Image d'entrée
                ax.imshow(pred_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)  # Overlay heatmap
                ax.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                plt.imsave(filepath.with_name(filename.replace('.png', '_check_input.png')), img_rgb)


                # --- Save raw prediction ---
                np.save(filepath.with_suffix('.npy'), pred)

            except Exception as e:
                print(f"Error occurred while saving output '{filename}': {e}")
        # === EVALUATION MÉTRIQUES ===
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc

    all_preds, all_labels = [], []

    print("\n=> Calcul des métriques globales sur toutes les images...")

    for index in range(len(test_dataset)):
        # Chargement prédiction binaire (float entre 0 et 1)
        pred_path = dataset_paths['SAVE_PRED'] / f"cltd_{index:05d}.npy"
        if not pred_path.exists():
            print(f"Fichier de prédiction manquant: {pred_path}")
            continue
        pred = np.load(pred_path)  # shape: (H, W)

        # Ground truth
        label = test_dataset[index]['label'].squeeze().numpy()  # (H, W)

        # Resize prediction à la taille du label si besoin
        if pred.shape != label.shape:
            pred = resize(pred, label.shape, order=1, preserve_range=True, anti_aliasing=True)

        # Aplatir
        all_preds.append(pred.flatten())
        all_labels.append(label.flatten())

    y_scores = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_pred_bin = (y_scores >= 0.5).astype(np.uint8)

    # Métriques
    iou = np.sum((y_pred_bin & y_true)) / np.sum((y_pred_bin | y_true))
    f1 = f1_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(pr_recall, pr_precision)

    # Résumé console
    print("\n====== Résultats - CAT-Net sur DocTamper (CLTD) ======")
    print(f"IoU       : {iou:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    print(f"PR AUC    : {pr_auc:.4f}")

    # Sauvegarde JSON
    result_dict = {
        "IoU": float(iou),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "ROC AUC": float(roc_auc),
        "PR AUC": float(pr_auc)
    }

    with open("benchmark_CAT_CLTD.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    # Courbes
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("roc_curve_CLTD.png")

    plt.figure()
    plt.plot(pr_recall, pr_precision, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.savefig("pr_curve_CLTD.png")

if __name__ == '__main__':
    main()

