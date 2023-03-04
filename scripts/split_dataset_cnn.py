import os
import shutil
import glob

SRC_ROOT = "data/src"
DST_ROOT = "data/dst"

DST_ROOT_TRAIN = f"{DST_ROOT}/train"
DST_ROOT_VALID = f"{DST_ROOT}/valid"
DST_ROOT_TEST = f"{DST_ROOT}/test"

os.makedirs(DST_ROOT, exist_ok=True)
os.makedirs(DST_ROOT_TRAIN, exist_ok=True)
os.makedirs(DST_ROOT_VALID, exist_ok=True)
# os.makedirs(DST_ROOT_TEST, exist_ok=True)

extTYPEs = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]

limit_total = 180
limit_train = 150
limit_valid = 30

f"""
Required images >= {limit_total}
* train: {limit_train}
* valid: {limit_valid}
* test: no idea...
"""

LACK_LOG_CREATED = False
cnt = 0
for dir in os.listdir(SRC_ROOT):
    target_dir_name = dir.replace("new_", "").replace("_6point", "").lower()

    src_root = os.path.join(SRC_ROOT, dir)
    dst_train = os.path.join(DST_ROOT_TRAIN, target_dir_name)
    dst_valid = os.path.join(DST_ROOT_VALID, target_dir_name)

    if os.path.isdir(src_root) == False:
        continue

    print(f"{dir:>34s} -> {target_dir_name:>34s}", end=" ")

    image_files = []
    for ext in extTYPEs:
        image_files += glob.glob(os.path.join(src_root, ext))

    if len(image_files) < limit_total:
        with open(f"{DST_ROOT}/not_enough_images.log", "a") as f:
            f.write(f"{dir} {len(image_files)}\n")

        LACK_LOG_CREATED = True
        print(" /  LACK")
        continue

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_valid, exist_ok=True)

    i, j = 1, 1
    # for file in os.listdir(src_root):
    for file in image_files:
        if i > limit_total:
            break

        fname = os.path.basename(file)
        fext = os.path.splitext(fname)[1]

        # target_fname = fname
        target_fname = f"{i}.{fext}"

        dst_path = dst_train
        if i >= limit_train:
            dst_path = dst_valid
            target_fname = f"{j}.{fext}"
            j += 1

        file_src = file
        file_dst = os.path.join(dst_path, target_fname)

        shutil.copyfile(file_src, file_dst)
        i += 1

    cnt += 1
    print(" /  OK")

print(f"Total {cnt} directories processed")
