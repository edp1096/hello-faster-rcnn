# DATASET_NAME = "face_mask"
# DATA_ROOT = "data/mask"
# PARSE_MODE = "xml"  # VOC xml format
# CLASS_COUNT = 4  # background, no_mask, mask, wrong mask

# DATASET_NAME = "humanface_bbox"
# DATA_ROOT = "data/vggface_bbox_dst"
# PARSE_MODE = "csv"  # single line csv format
# CLASS_COUNT = 2  # background, face

DATASET_NAME = "animalface"
DATA_ROOT = "data/face_nemo"
PARSE_MODE = "json"  # LabelMe json format
CLASS_COUNT = 2  # background, face

MODEL_NAME = "faster_rcnn_resnet50_fpn_v2"

# USE_AMP = False
USE_AMP = True

OUTPUT_SAVE_ROOT = "weights"
COMMON_FILENAME = f"{OUTPUT_SAVE_ROOT}/{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_FILE = f"{COMMON_FILENAME}.pt"
WEIGHT_INFO_FILE = f"{COMMON_FILENAME}_info.log"
SCATTER_FILE = f"{COMMON_FILENAME}_dist"
LOSS_RESULT_FILE = f"{COMMON_FILENAME}.png"


EPOCHS = 10
BATCH_SIZE = 2  # VRAM 4GB : 1, 8GB : 3
