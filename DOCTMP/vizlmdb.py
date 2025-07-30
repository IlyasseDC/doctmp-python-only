import os
import cv2
import six
import lmdb
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='DocTamperV1-FCD')
parser.add_argument('--output', type=str, default='images')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

env = lmdb.open(args.input, readonly=True, lock=False, readahead=False, meminit=False)

with env.begin(write=False) as txn:
    n_samples = int(txn.get('num-samples'.encode('utf-8')))
    print(f"Found {n_samples} samples in LMDB.")

    for index in range(n_samples):
        img_key = 'image-%09d' % index
        imgbuf = txn.get(img_key.encode('utf-8'))
        if imgbuf is None:
            continue

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf)
        im.save(os.path.join(args.output, f"image-{index:09d}.jpg"))

        lbl_key = 'label-%09d' % index
        lblbuf = txn.get(lbl_key.encode('utf-8'))
        if lblbuf is not None:
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            if mask.max() == 1:
                mask = mask * 255
            cv2.imwrite(os.path.join(args.output, f"mask-{index:09d}.png"), mask)

        if index % 100 == 0:
            print(f"Saved {index}/{n_samples}")

print("Extraction completed.")
