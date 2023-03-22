from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from modules.dataset import BoxDataset
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

# LABELS = ["", "no mask", "mask", "wrong mask"]
LABELS = ["", "face"]

colors = []
for i in range(CLASS_COUNT):
    colors.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))

def collate_fn(batch):
    return tuple(zip(*batch))


data_transform = transforms.ToTensor()
test_dataset = BoxDataset(data_transform, f"{DATA_ROOT}/test", parse_mode=PARSE_MODE)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

model = models.detection.fasterrcnn_resnet50_fpn_v2()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model.load_state_dict(torch.load(f"{WEIGHT_FILE}")["model"])

model.to(device)
# print(model)

model.eval()

for imgs, annotations in test_data_loader:
    imgs = list(img.to(device) for img in imgs)

    preds = getPredictions(model, imgs, 0.5)

    break

_idx = random.randint(0, len(annotations) - 1)  # 랜덤 숫자
print(f"Index: {_idx}/{len(annotations)}")

print("Target : ", annotations[_idx]["labels"])
print("Prediction : ", preds[_idx]["labels"])

print(type(imgs[_idx]))

plotImageCV(imgs[_idx], annotations[_idx], labels=LABELS, colors=colors)
plotImageCV(imgs[_idx], preds[_idx], labels=LABELS, colors=colors)


"""
Predictions for all test images
"""
labels = []
preds_adj_all = []
annot_all = []

for im, annot in tqdm(test_data_loader, position=0, leave=True):
    im = list(img.to(device) for img in im)
    # annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

    for t in annot:
        labels += t["labels"]

    with torch.no_grad():
        preds_adj = getPredictions(model, im, 0.5)
        preds_adj = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in preds_adj]
        preds_adj_all.append(preds_adj)
        annot_all.append(annot)

"""
Statistics
"""
sample_metrics = []
for batch_i in range(len(preds_adj_all)):
    sample_metrics += getBatchStats(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
precision, recall, AP, f1, ap_class = getAveragePrecisions(true_positives, pred_scores, pred_labels, torch.tensor(labels))
mAP = torch.mean(AP)
print(f"mAP : {mAP}")
print(f"AP : {AP}")
