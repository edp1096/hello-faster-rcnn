from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.cuda.amp as amp

from modules.dataset import BoxDataset
import modules.fit as fit
from modules.file import saveWeights, saveEpochInfo, loadWeights

import time
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}, Device: {device}")

scaler = None
if USE_AMP:
    BATCH_SIZE *= 2
    scaler = amp.GradScaler()


def collate_fn(batch):
    return tuple(zip(*batch))


data_transform = transforms.ToTensor()
train_dataset = BoxDataset(data_transform, f"{DATA_ROOT}/train", parse_mode=PARSE_MODE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

model = models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model.to(device)
print(model)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

best_epoch, best_loss = 0, 1000000
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_dataset)}]\n-------------------------------")
    start = time.time()

    if USE_AMP:
        train_loss = fit.runAMP(device, train_loader, model, optimizer, scaler)
    else:
        train_loss = fit.run(device, train_loader, model, optimizer)

    print(f"Loss : {train_loss}, time : {time.time() - start:>.5f}")

    # 모델 저장
    if train_loss < best_loss:
        saveWeights(model.state_dict(), optimizer.state_dict(), best_loss, train_loss)  # Save weights
        saveEpochInfo(epoch, train_loss, train_loss, math.nan, train_loss)  # Write epoch info

        best_epoch = epoch
        best_loss = train_loss
