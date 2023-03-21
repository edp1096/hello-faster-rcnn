import torch


def run(device, dataloader, model, optimizer):
    epoch_loss = 0.0

    model.train()
    for imgs, annotations in dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        with torch.set_grad_enabled(True):
            pred = model(imgs, annotations)
            losses = sum(loss for loss in pred.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        epoch_loss += losses

    return epoch_loss / len(dataloader)

def runAMP(device, dataloader, model, optimizer, scaler):
    epoch_loss = 0.0

    model.train()
    for imgs, annotations in dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        with torch.set_grad_enabled(True):
            pred = model(imgs, annotations)
            losses = sum(loss for loss in pred.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += losses

    return epoch_loss / len(dataloader)
