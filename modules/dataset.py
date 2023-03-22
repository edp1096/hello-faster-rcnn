import torch

import os
from PIL import Image
from bs4 import BeautifulSoup
import json
import numpy as np

adjust_label = 1  # For background


def getBoxPoints(obj):
    x1 = float(obj.find("xmin").text)
    y1 = float(obj.find("ymin").text)
    x2 = float(obj.find("xmax").text)
    y2 = float(obj.find("ymax").text)

    return [x1, y1, x2, y2]


def getLabel(obj):
    if obj.find("name").text == "with_mask":
        return 1 + adjust_label

    elif obj.find("name").text == "mask_weared_incorrect":
        return 2 + adjust_label

    return 0 + adjust_label


def generate_target(file, mode):
    boxes = []
    labels = []

    with open(file) as f:
        data = f.read()

        if mode == "json":
            jdata = json.loads(data)

            if "shapes" in jdata and  len(jdata["shapes"]) > 0:
                if "points" in jdata["shapes"][0] and len(jdata["shapes"][0]["points"]) > 1:
                    box = [
                        jdata["shapes"][0]["points"][0][0],
                        jdata["shapes"][0]["points"][0][1],
                        jdata["shapes"][0]["points"][1][0],
                        jdata["shapes"][0]["points"][1][1],
                    ]
                    if jdata["shapes"][0]["points"][0][0] > jdata["shapes"][0]["points"][1][0]:
                        box[0] = jdata["shapes"][0]["points"][1][0]
                        box[1] = jdata["shapes"][0]["points"][1][1]
                        box[2] = jdata["shapes"][0]["points"][0][0]
                        box[3] = jdata["shapes"][0]["points"][0][1]

                    boxes.append(box)
                    labels.append(0 + adjust_label)
        elif mode == "xml":
            soup = BeautifulSoup(data, "html.parser")
            objects = soup.find_all("object")
            num_objs = len(objects)

            for obj in objects:
                boxes.append(getBoxPoints(obj))
                labels.append(getLabel(obj))
        else:
            # single line csv
            boxes = np.array([np.fromstring(data, sep=",", dtype=np.float32)])
            labels = [1, 2]  # background, object(bbox)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target


class BoxDataset(object):
    def __init__(self, transforms, data_path, parse_mode=None):
        self.transforms = transforms
        self.path = data_path
        self.imgs = list(sorted(os.listdir(os.path.join(self.path, "images"))))
        self.mode = "xml"
        if parse_mode is not None and parse_mode.lower() in ["xml", "json", "csv"]:
            self.mode = parse_mode.lower()

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + self.mode

        img_path = os.path.join(self.path, "images", file_image)
        label_path = os.path.join(self.path, "annotations", file_label)

        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path, self.mode)  # Generate Label

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
