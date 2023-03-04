from config import *

import torch
import torchvision
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from modules.plot import combineImageWithPrediction, showImageResult, plotImage, plotImageCV
from modules.image import getImage, cropImage, saveImages
from modules.prediction import getPredictions, getPredictionsCOCO

import os


device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


SAVE_PATH = "z_crop_result"
MODEL_FILENAME = "model_face_crop.pt"
RESIZE_W = 224
RESIZE_H = 224

FACE_LABELS = ["", "face"]
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


model_crop_body = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
)

model_crop_face = models.detection.fasterrcnn_resnet50_fpn()
in_features = model_crop_face.roi_heads.box_predictor.cls_score.in_features
model_crop_face.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model_crop_face.load_state_dict(torch.load(MODEL_FILENAME))

model_crop_body.to(device)
model_crop_face.to(device)

model_crop_body.eval()
model_crop_face.eval()


xfrm = transforms.ToTensor()
COCO_THRESHOLD = 0.6


def cropBody(img, xfrm_img):
    boxes, classes, scores = getPredictionsCOCO(xfrm_img, model_crop_body, COCO_THRESHOLD, COCO_CLASS_NAMES)

    body_boxes = []
    body_classes = []
    body_scores = []

    for i, class_name in enumerate(classes):
        if class_name == "dog" and scores[i] > 0.8:
            body_boxes.append(boxes[i])
            body_classes.append(classes[i])
            body_scores.append(scores[i])

    imgs_cropped = cropImage(img, body_boxes)

    return imgs_cropped, body_boxes, body_classes, body_scores


def cropFace(img):
    with torch.no_grad():
        preds = getPredictions(model_crop_face, [img], 0.6)

    t_boxes = preds[0]["boxes"]
    t_classes = preds[0]["labels"]
    t_scores = preds[0]["scores"]

    boxes = []
    classes = []
    scores = []

    imgs_croped = []
    for i, class_name in enumerate(t_classes):
        if class_name == 1 and t_scores[i] > 0.8:
            boxes.append(t_boxes[i])
            classes.append(t_classes[i])
            scores.append(t_scores[i])

            croped_img = img[:, int(t_boxes[i][1]) : int(t_boxes[i][3]), int(t_boxes[i][0]) : int(t_boxes[i][2])]
            imgs_croped.append(croped_img)

    return imgs_croped, boxes, classes, scores


img = getImage("dog0.jpg")
xfrm_img = xfrm(img).to(device)

body_imgs, body_boxes, body_classes, body_scores = cropBody(img, xfrm_img)
xfrm_body_imgs = [xfrm(im).to(device) for im in body_imgs]
face_imgs, face_boxes, face_classes, face_scores = cropFace(xfrm_body_imgs[0])


result_imgs = []
for im in face_imgs:
    size = min(im.shape[1], im.shape[2])
    im = TF.center_crop(im, size)
    im = TF.resize(im, (224, 224))

    result_imgs.append(im)

os.makedirs(SAVE_PATH, exist_ok=True)


saveImages(SAVE_PATH, result_imgs, face_classes, face_scores, color_range=255, filename="face1")
