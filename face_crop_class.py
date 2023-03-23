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


# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}, Device: {device}")


SAVE_PATH = "z_crop_result"
# RESIZE_W, RESIZE_H = 224, 224
RESIZE_W, RESIZE_H = 384, 384
# RESIZE_W, RESIZE_H = 512, 512

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


model_detect_body = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
)

model_detect_face = models.detection.fasterrcnn_resnet50_fpn_v2()
in_features = model_detect_face.roi_heads.box_predictor.cls_score.in_features
model_detect_face.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model_detect_face.load_state_dict(torch.load(WEIGHT_FILE)["model"])

model_detect_body.to(device)
model_detect_face.to(device)

model_detect_body.eval()
model_detect_face.eval()

xfrm = transforms.ToTensor()
COCO_THRESHOLD = 0.6


def cropBody(img, xfrm_img):
    boxes, classes, scores = getPredictionsCOCO(xfrm_img, model_detect_body, COCO_THRESHOLD, COCO_CLASS_NAMES)

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
        preds = getPredictions(model_detect_face, [img], 0.6)

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


def faceCropRun(savedir, fnames):
    for i, fname in enumerate(fnames):
        img = getImage(fname)
        xfrm_img = xfrm(img).to(device)

        body_imgs, body_boxes, body_classes, body_scores = cropBody(img, xfrm_img)
        xfrm_body_imgs = [xfrm(im).to(device) for im in body_imgs]

        if len(xfrm_body_imgs) > 0:
            face_imgs, face_boxes, face_classes, face_scores = cropFace(xfrm_body_imgs[0])
        else:
            print(f"no body detected, try to find face: {i} - {fname}")
            face_imgs, face_boxes, face_classes, face_scores = cropFace(xfrm_img)

        # result_imgs = []
        # for im in face_imgs:
        #     size = min(im.shape[1], im.shape[2])
        #     im = TF.center_crop(im, size)
        #     im = TF.resize(im, (RESIZE_W, RESIZE_H), antialias=False)
        #     result_imgs.append(im)

        result_imgs = face_imgs
        saveImages(savedir, result_imgs, face_classes, face_scores, color_range=255, filename=i)


fpaths = [
    "data/sample/human0.jpg",
    "data/sample/human1.jpg",
    "data/sample/human2.jpg",
]

src_root = "D:/dev/datasets/0_animalface/5face_1k"
dst_root = SAVE_PATH
os.makedirs(dst_root, exist_ok=True)

files_list = {}
for dir in os.listdir(src_root):
    fpaths = []
    for fname in os.listdir(os.path.join(src_root, dir)):
        fpaths.append(os.path.join(src_root, dir, fname))
    
    files_list[dir] = fpaths

for dir, fpaths in files_list.items():
    dst_root = os.path.join(SAVE_PATH, dir)

    os.makedirs(dst_root, exist_ok=True)
    faceCropRun(dst_root, fpaths)

    # for fpath in fpaths:
    #     fname = os.path.basename(fpath)
    #     print(dir, fname)

