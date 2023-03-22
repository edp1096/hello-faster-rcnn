import torch


def run(device, loader, model, optimizer):
    model.train()

    epoch_loss = 0.0

    for imgs, annotations in loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = model(imgs, annotations)
            losses = sum(loss for loss in pred.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        epoch_loss += losses

    return epoch_loss / len(loader)


def runAMP(device, loader, model, optimizer, scaler):
    model.train()

    epoch_loss = 0.0

    for imgs, annotations in loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast():
                pred = model(imgs, annotations)
                losses = sum(loss for loss in pred.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += losses

    return epoch_loss / len(loader)
