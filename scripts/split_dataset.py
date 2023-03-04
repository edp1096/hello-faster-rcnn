import os
import random
import numpy as np
import shutil


SRC_ROOT = "archive"
SRC_IMAGE = f"{SRC_ROOT}/images"
SRC_ANNOT = f"{SRC_ROOT}/annotations"

DST_ROOT = "data/mask"
DST_TRAIN = f"{DST_ROOT}/train"
DST_TRAIN_IMAGE = f"{DST_TRAIN}/images"
DST_TRAIN_ANNOT = f"{DST_TRAIN}/annotations"
DST_TEST = f"{DST_ROOT}/test"
DST_TEST_IMAGE = f"{DST_TEST}/images"
DST_TEST_ANNOT = f"{DST_TEST}/annotations"


if os.path.exists(SRC_ROOT) == False:
    print("No 'archive' folder found.\nPlease download data from https://www.kaggle.com/andrewmvd/face-mask-detection and extract it to 'archive' folder.")
    exit(1)


os.makedirs(DST_ROOT, exist_ok=True)
os.makedirs(DST_TRAIN, exist_ok=True)
os.makedirs(DST_TRAIN_IMAGE, exist_ok=True)
os.makedirs(DST_TRAIN_ANNOT, exist_ok=True)
os.makedirs(DST_TEST, exist_ok=True)
os.makedirs(DST_TEST_IMAGE, exist_ok=True)
os.makedirs(DST_TEST_ANNOT, exist_ok=True)


random.seed(1234)
idx = random.sample(range(853), 170)

for img in np.array(sorted(os.listdir(SRC_IMAGE)))[idx]:
    shutil.move(f"{SRC_IMAGE}/{img}", f"{DST_TEST_IMAGE}/{img}")

for annot in np.array(sorted(os.listdir(SRC_ANNOT)))[idx]:
    shutil.move(f"{SRC_ANNOT}/{annot}", f"{DST_TEST_ANNOT}/{annot}")

for img in np.array(sorted(os.listdir(SRC_IMAGE))):
    shutil.move(f"{SRC_IMAGE}/{img}", f"{DST_TRAIN_IMAGE}/{img}")

for annot in np.array(sorted(os.listdir(SRC_ANNOT))):
    shutil.move(f"{SRC_ANNOT}/{annot}", f"{DST_TRAIN_ANNOT}/{annot}")


shutil.rmtree(SRC_ROOT)