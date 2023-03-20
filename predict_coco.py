from config import *

import torchvision
from torchvision import models, transforms

from modules.plot import combineImageWithPrediction, showImageResult
from modules.image import getImage, cropImage, saveImages
from modules.prediction import getPredictionsCOCO

import os

# dog index = 18
COCO_CLASS_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
)

xfrm = transforms.Compose([transforms.ToTensor()])


DATA_ROOT = "data/sample"
IMAGE_PATH_SRC = os.path.join(DATA_ROOT, "original")
IMAGE_PATH_DST = os.path.join(DATA_ROOT, "cropped")

IMG_FILENAME = f"{IMAGE_PATH_SRC}/traffic.jpg"
# IMG_FILENAME = f"{IMAGE_PATH_SRC}/dog.jpg"
# IMG_FILENAME = f"{IMAGE_PATH_SRC}/cat.jpg"
# IMG_FILENAME = f"{IMAGE_PATH_SRC}/tom_cruise.jpg"
# IMG_FILENAME = "mask1.png"
# IMG_FILENAME = "cat1.jpg"
# IMG_FILENAME = "dog1.jpg"

THRESHOLD = 0.6


img = getImage(IMG_FILENAME)
xfrm_img = xfrm(img)
boxes, classes, scores = getPredictionsCOCO(xfrm_img, model, THRESHOLD, COCO_CLASS_NAMES)
img_predicted = combineImageWithPrediction(img.copy(), boxes, classes, scores)

imgs_cropped = cropImage(img, boxes)
saveImages(IMAGE_PATH_DST, imgs_cropped, classes, scores)

print("boxes:", boxes)
print("classes:", classes)
print("scores:", scores)
showImageResult(img, img_predicted)
