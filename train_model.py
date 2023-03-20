from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from modules.dataset import MaskDataset
import modules.fit as fit

import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def collate_fn(batch):
    return tuple(zip(*batch))

data_transform = transforms.ToTensor()
train_dataset = MaskDataset(data_transform, f"{DATA_ROOT}/train", parse_mode=PARSE_MODE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

model = models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model.to(device)
# print(model)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_dataset)}]\n-------------------------------")
    start = time.time()

    epoch_loss = fit.run(device, train_loader, model, optimizer)
    print(f"Loss : {epoch_loss}, time : {time.time() - start:>.5f}")

torch.save(model.state_dict(), f"model_{DATASET_NAME}_{EPOCHS}.pt")
