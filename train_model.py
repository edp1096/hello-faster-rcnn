from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.cuda.amp as amp

from modules.dataset import FacialBoxDataset
import modules.fit as fit
from modules.file import saveWeights, saveEpochInfo

import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

scaler = None
if USE_AMP:
    # BATCH_SIZE *= 2
    scaler = amp.GradScaler()


def collate_fn(batch):
    return tuple(zip(*batch))


data_transform = transforms.ToTensor()

image_path, annotation_path = f"{DATA_ROOT}/images", f"{DATA_ROOT}/annotations"
data_set = FacialBoxDataset(image_path, annotation_path, transforms=None, parse_mode=PARSE_MODE)
data_set.transforms = data_transform

train_size = int(0.8 * len(data_set))
valid_size = len(data_set) - train_size
train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

model_weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
backbone_weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=model_weights, weights_backbone=backbone_weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CLASS_COUNT)

model.to(device)
# print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

total_start_time = time.time()
train_losses, valid_losses = [], []
train_loss, valid_loss, best_loss = 9999, 9999, 9999
best_epoch = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")
    epoch_start_time = time.time()

    if USE_AMP:
        train_loss = fit.runAMP(device, train_loader, model, optimizer, scaler)
    else:
        train_loss = fit.run(device, train_loader, model, optimizer)
    valid_loss = fit.run(device, valid_loader, model, optimizer)

    print(f"Train - Loss: {train_loss:>3.5f}")
    print(f"Valid - Loss: {valid_loss:>3.5f}")

    train_losses.append(train_loss), valid_losses.append(valid_loss)

    print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

    # 모델 저장
    if valid_loss < best_loss:
        saveWeights(model.state_dict(), optimizer.state_dict(), best_loss, valid_loss)  # Save weights
        saveEpochInfo(epoch, train_loss, train_loss, valid_loss, valid_loss)  # Write epoch info

        best_epoch = epoch
        best_loss = valid_loss

print(f"Total time: {time.time() - total_start_time:.2f} seconds")
print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss:.5f}")

result = {
    "train_losses": train_losses,
    "valid_losses": valid_losses,
    "best_epoch": best_epoch,
    "best_loss": best_loss,
}
