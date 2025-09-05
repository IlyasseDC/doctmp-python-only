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

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import lmdb
import six
import pickle
import cv2
import numpy as np

# Charger l'environnement LMDB
lmdb_path = '/home/ilyassechaouki/DOCTMP/DocTamperV1-TrainingSet'  # chemin vers ton dossier LMDB
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

def load_gt_mask(index):
    with env.begin(write=False) as txn:
        key = f'label-{index:09d}'.encode('utf-8')
        lblbuf = txn.get(key)
        if lblbuf is None:
            print(f"Aucun masque trouvé pour index {index}")
            return None
        mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
        return mask

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
    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    # args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream

    print(test_dataset.get_info())

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
    model = nn.DataParallel(model, device_ids=gpus).cuda()

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
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filepath = dataset_paths['SAVE_PRED'] / filename

            # plot
            import cv2
            from PIL import Image

            # Fichiers
            image_filename = get_next_filename(index)
            filename_no_ext = os.path.splitext(image_filename)[0]
            input_path = dataset_paths['IMAGE_FOLDER'] / image_filename
            gt_path = dataset_paths['GT_FOLDER'] / f"{filename_no_ext}.png"
            output_overlay_path = dataset_paths['SAVE_PRED'] / f"{filename_no_ext}_overlay.png"
            output_vis_path = dataset_paths['SAVE_VIS'] / f"{filename_no_ext}_vis.png"
            dataset_paths['SAVE_VIS'].mkdir(parents=True, exist_ok=True)
            lmdb_index = int(filename_no_ext.split('_')[-1])

            # Charger image originale
            original_image = cv2.imread(str(input_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            height, width, _ = original_image.shape

            # Resize prédiction
            pred_resized = cv2.resize(pred, (width, height))
            heatmap = np.uint8(255 * pred_resized)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

            # Sauvegarder l'overlay seul
            cv2.imwrite(str(output_overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Charger masque GT si disponible
            gt_mask = load_gt_mask(lmdb_index)
            if gt_mask is None:
                gt_mask = np.zeros((height, width), dtype=np.uint8)
            else:
                gt_mask = cv2.resize(gt_mask, (width, height))


            # Visualisation finale
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(original_image)
            axs[0].set_title("Image")
            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("Mask GT")
            axs[2].imshow(pred_resized, cmap='jet', vmin=0, vmax=1)
            axs[2].set_title("Pred CAT-Net")
            axs[3].imshow(overlay)
            axs[3].set_title("Overlay")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_vis_path, dpi=150)
            plt.close()



if __name__ == '__main__':
    main()
