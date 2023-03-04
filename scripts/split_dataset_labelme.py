import os
import numpy as np
import shutil

SOURCE_DIR = "data/src"
TARGET_DIR = "src/dst"

TOTAL_COUNT = 100
TRAIN_COUNT = int(TOTAL_COUNT * 0.6)
TEST_COUNT = int(TOTAL_COUNT * 0.2)

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train/annotations", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test/annotations", exist_ok=True)

for dog_class in os.listdir(SOURCE_DIR):
    print(f"{dog_class}: ", end="")
    dir = SOURCE_DIR + "/" + dog_class
    class_name = dog_class.replace("new_", "")

    cnt = 0
    for file in np.array(sorted(os.listdir(dir))):
        fname = file.split(".")[0]
        ext = file.split(".")[-1]

        if ext != "jpg" or ext != "png":
            continue

        if cnt <= TRAIN_COUNT:
            shutil.copy(f"{dir}/{fname}.jpg", f"{TARGET_DIR}/train/images/{dog_class}_{cnt}.{ext}")
            shutil.copy(f"{dir}/{fname}.json", f"{TARGET_DIR}/train/annotations/{dog_class}_{cnt}.json")
        else:
            shutil.copy(f"{dir}/{fname}.jpg", f"{TARGET_DIR}/test/images/{dog_class}_{cnt}.{ext}")
            shutil.copy(f"{dir}/{fname}.json", f"{TARGET_DIR}/test/annotations/{dog_class}_{cnt}.json")

        if cnt >= TRAIN_COUNT + TEST_COUNT:
            break

        cnt += 1

    print("done")

print("All done. train count, test count:", TRAIN_COUNT, TEST_COUNT)
