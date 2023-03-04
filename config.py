# DATASET_NAME = "mask_detection"
# DATA_ROOT = "data/mask"
# PARSE_MODE = "xml"  # VOC format xml
# CLASS_COUNT = 4  # background, no_mask, mask, wrong mask

DATASET_NAME = "face"
DATA_ROOT = "data/face_nemo"
PARSE_MODE = "json"  # LabelMe format json
CLASS_COUNT = 2  # background, face


# EPOCHS = 10
EPOCHS = 4
BATCH_SIZE = 3  # VRAM 4GB : 1, 8GB : 3
