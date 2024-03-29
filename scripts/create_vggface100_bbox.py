import os, random
import pandas as pd
from PIL import Image
from shutil import copyfile
import numpy as np


DATA_SRC_ROOT = "D:/dev/datasets/VGG-Face2"
# DATA_SRC_ROOT = "data/vggface_src"

IMAGE_SRC_PATH = f"{DATA_SRC_ROOT}/data/train"
DATA_DST_ROOT = "data/vggface_bbox_dst"
IMAGE_TRAIN_PATH = f"{DATA_DST_ROOT}/train/images"
ANNOTATION_TRAIN_PATH = f"{DATA_DST_ROOT}/train/annotations"
IMAGE_TEST_PATH = f"{DATA_DST_ROOT}/test/images"
ANNOTATION_TEST_PATH = f"{DATA_DST_ROOT}/test/annotations"


person_limit = 100
image_limit_per_person = 20

selected_ids = []

df_bbox = pd.read_csv(f"{DATA_SRC_ROOT}/meta/bb_landmark/loose_bb_train.csv")
df_landmark = pd.read_csv(f"{DATA_SRC_ROOT}/meta/bb_landmark/loose_landmark_train.csv")

selected_ids = []

i = 0
while True:
    selected_item = random.choice(os.listdir(IMAGE_SRC_PATH))
    if selected_item in selected_ids:
        continue

    selected_ids.append(selected_item)
    i += 1

    if i >= person_limit:
        break

os.makedirs(DATA_DST_ROOT, exist_ok=True)
os.makedirs(IMAGE_TRAIN_PATH, exist_ok=True)
os.makedirs(ANNOTATION_TRAIN_PATH, exist_ok=True)
os.makedirs(IMAGE_TEST_PATH, exist_ok=True)
os.makedirs(ANNOTATION_TEST_PATH, exist_ok=True)

image_count = 0
image_count_per_person = 0
person_count = 0
random_image_values = np.random.randint(0, image_limit_per_person, person_limit)
prev_person_id = ""
mode = "train"
for j, row in df_bbox.iterrows():
    person_id = row["NAME_ID"].split("/")[0]
    image_name = row["NAME_ID"].split("/")[1]

    if person_id not in selected_ids:
        continue

    if person_id != prev_person_id:
        image_count_per_person = 0
        person_count += 1
        prev_person_id = person_id
        mode = "train"

    if mode != "test" and image_count_per_person >= image_limit_per_person * 0.8:
        mode = "test"

    if image_count_per_person >= image_limit_per_person:
        continue

    # Prepare data
    target_image_path = IMAGE_TRAIN_PATH
    target_annotation_path = ANNOTATION_TRAIN_PATH
    if mode == "test":
        target_image_path = IMAGE_TEST_PATH
        target_annotation_path = ANNOTATION_TEST_PATH

    # copyfile(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg", f"{target_image_path}/{image_count}.jpg")
    # resize image to 1333 if width or height > 1333
    max_size = 1333
    img_scale_ratio = 1
    img = Image.open(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg")
    if img.width > img.height:
        if img.width > max_size:
            print("width over max_size:", img.width, img.height)
            img_scale_ratio = max_size / img.width
            img = img.resize((max_size, int(img.height * img_scale_ratio)))
    else:
        if img.height > max_size:
            print("height over max_size:", img.width, img.height)
            img_scale_ratio = max_size / img.height
            img = img.resize((int(img.width * img_scale_ratio), max_size))

    img.save(f"{target_image_path}/{image_count}.jpg")

    # Make square bounding box
    if row["W"] > row["H"]:
        row["Y"] -= (row["W"] - row["H"]) / 2
        row["H"] = row["W"]
    elif row["W"] < row["H"]:
        row["X"] -= (row["H"] - row["W"]) / 2
        row["W"] = row["H"]

    bbox = [row["X"], row["Y"], row["X"] + row["W"], row["Y"] + row["H"]]  # top-left and bottom-right
    bbox = [int(point * img_scale_ratio) for point in bbox]  # scale to resized image

    with open(f"{target_annotation_path}/{image_count}.csv", "w") as f:
        f.write(",".join([str(point) for point in bbox]))

    image_count += 1
    image_count_per_person += 1

    if image_count % 100 == 0:
        print("image_count:", image_count)

print("image_count:", image_count)
