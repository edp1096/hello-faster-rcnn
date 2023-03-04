import torch

import cv2
import numpy as np


def getImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def cropImage(img, box_pts):
    cropped = []
    for i in range(len(box_pts)):
        bbox_p1 = (int(box_pts[i][0][0]), int(box_pts[i][0][1]))
        bbox_p2 = (int(box_pts[i][1][0]), int(box_pts[i][1][1]))

        cropped.append(img[bbox_p1[1] : bbox_p2[1], bbox_p1[0] : bbox_p2[0]])

    return cropped


def saveImages(target_path, imgs, classes, scores, color_range=1, filename=None):
    for i in range(len(imgs)):
        im = imgs[i]
        if type(imgs[i]) == torch.Tensor:
            im = imgs[i].cpu().permute(1, 2, 0).numpy().copy()

        fname = f"{target_path}/{classes[i]}_{scores[i]:>.2f}"
        if filename is not None:
            fname = f"{target_path}/{filename}"

        result = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{fname}.jpg", result * color_range)
