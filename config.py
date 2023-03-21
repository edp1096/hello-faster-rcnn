DATASET_NAME = "mask_detection"
DATA_ROOT = "data/mask/train"
DATA_TEST_ROOT = "data/mask/test"
PARSE_MODE = "xml"  # VOC xml format
CLASS_COUNT = 4  # background, no_mask, mask, wrong mask

# DATASET_NAME = "humanface_bbox"
# DATA_ROOT = "data/vggface_dst/train"
# DATA_TEST_ROOT = "data/vggface_dst/test"
# PARSE_MODE = "csv"  # single line csv. p1x, p1y, p2x, p2y
# CLASS_COUNT = 2  # background, face

# DATASET_NAME = "face_nemo"
# DATA_ROOT = "data/face_nemo/train"
# DATA_TEST_ROOT = "data/face_nemo/test"
# PARSE_MODE = "json"  # LabelMe json format
# CLASS_COUNT = 2  # background, face

MODEL_NAME = "faster_rcnn"

USE_AMP = True

OUTPUT_SAVE_ROOT = "weights"
COMMON_FILENAME = f"{OUTPUT_SAVE_ROOT}/{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_FILE = f"{COMMON_FILENAME}.pt"
WEIGHT_INFO_FILE = f"{COMMON_FILENAME}_info.log"
SCATTER_FILE = f"{COMMON_FILENAME}_dist"
LOSS_RESULT_FILE = f"{COMMON_FILENAME}.png"

# EPOCHS = 10
EPOCHS = 4
BATCH_SIZE = 2  # VRAM 4GB : 1, 8GB : 2
