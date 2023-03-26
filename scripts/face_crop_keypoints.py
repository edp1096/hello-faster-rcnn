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
import numpy as np


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


def faceCropRun(savedir, im_fnames, an_fnames):
    j = 0
    for i, im_fname in enumerate(im_fnames):
        img = getImage(im_fname)
        xfrm_img = xfrm(img).to(device)
        crop_box = [0.0, 0.0, 0.0, 0.0]

        body_imgs, body_boxes, body_classes, body_scores = cropBody(img, xfrm_img)
        xfrm_body_imgs = [xfrm(im).to(device) for im in body_imgs]

        if len(xfrm_body_imgs) > 0:
            crop_box = [
                body_boxes[0][0][0].cpu().numpy(),
                body_boxes[0][0][1].cpu().numpy(),
                body_boxes[0][1][0].cpu().numpy(),
                body_boxes[0][1][1].cpu().numpy(),
            ]

            face_imgs, face_boxes, face_classes, face_scores = cropFace(xfrm_body_imgs[0])
        else:
            print(f"no body detected, try to find face: {i} - {im_fname}")
            face_imgs, face_boxes, face_classes, face_scores = cropFace(xfrm_img)

        if len(face_boxes) == 0:
            print(f"facebox is missing: {i} - {im_fname}")
            continue

        crop_box = [
            crop_box[0] + face_boxes[0][0].cpu().numpy(),
            crop_box[1] + face_boxes[0][1].cpu().numpy(),
            crop_box[0] + face_boxes[0][2].cpu().numpy(),
            crop_box[1] + face_boxes[0][3].cpu().numpy(),
        ]

        result_imgs = face_imgs

        an_fname = an_fnames[i]
        points = np.loadtxt(an_fname, delimiter=",", dtype=np.float32)

        if len(points) < 6:
            print(f"incorrect facebox: {i} - {im_fname} / no points")
            continue

        new_points = np.array(
            [
                points[0] - crop_box[0],
                points[1] - crop_box[1],
                points[2] - crop_box[0],
                points[3] - crop_box[1],
                points[4] - crop_box[0],
                points[5] - crop_box[1],
            ]
        )

        if new_points[0] < 0 or new_points[1] < 0:
            print(f"incorrect facebox: {i} - {im_fname} / upper(left eye) side")
            continue
        if new_points[2] > crop_box[2] - crop_box[0] or new_points[3] < 0:
            print(f"incorrect facebox: {i} - {im_fname} / upper(right eye) side")
            continue
        if new_points[4] < 0 or new_points[4] > crop_box[2] - crop_box[0] or new_points[5] > crop_box[3] - crop_box[1]:
            print(f"incorrect facebox: {i} - {im_fname} / lower(nose) side")
            continue

        saveImages(f"{savedir}/images", [result_imgs[0]], face_classes, face_scores, color_range=255, filename=j)
        np.savetxt(f"{savedir}/annotations/{j}.csv", [new_points], delimiter=",", fmt="%f")

        j += 1


src_root = "data/face_keypoints"
dst_root = SAVE_PATH
os.makedirs(dst_root, exist_ok=True)

images_list, annotations_list = {}, {}
for dir in os.listdir(src_root):
    im_fpaths, an_fpaths = [], []
    for fname in os.listdir(os.path.join(src_root, f"{dir}/images")):
        fname_base = os.path.splitext(fname)[0]

        im_fpaths.append(os.path.join(src_root, f"{dir}/images", fname))
        an_fpaths.append(os.path.join(src_root, f"{dir}/annotations", fname_base + ".csv"))

    images_list[dir] = im_fpaths
    annotations_list[dir] = an_fpaths

for dir, im_fpaths in images_list.items():
    dst_path = os.path.join(SAVE_PATH, dir)

    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(f"{dst_path}/images", exist_ok=True)
    os.makedirs(f"{dst_path}/annotations", exist_ok=True)

    an_fpaths = annotations_list[dir]
    faceCropRun(dst_path, im_fpaths, an_fpaths)
