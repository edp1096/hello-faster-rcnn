from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from modules.image import getImage
from modules.prediction import getPredictions
from modules.plot import plotImage, plotImageCV
from modules.stats import getBatchStats, getAveragePrecisions

from tqdm import tqdm
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

LABELS = ["", "no mask", "mask", "wrong mask"]

colors = []
for i in range(CLASS_COUNT):
    colors.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))


model = models.detection.fasterrcnn_resnet50_fpn_v2()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model.load_state_dict(torch.load(f"model_{DATASET_NAME}_{EPOCHS}.pt"))
model.to(device)
# print(model)

model.eval()


xfrm = transforms.ToTensor()

IMG_FILENAME = "data/sample/mask1.png"
# IMG_FILENAME = "data/sample/cat1.jpg"
# IMG_FILENAME = "data/sample/dog1.jpg"

img = xfrm(getImage(IMG_FILENAME)).to(device)
# imgs = list([getImage(IMG_FILENAME)])

imgs = [img]

with torch.no_grad():
    preds = getPredictions(model, imgs, 0.6)

print("Prediction : ", preds[0]["labels"])
print(type(img))
plotImageCV(img, preds[0], labels=LABELS, colors=colors)
