import torch


def run(device, loader, model, optimizer):
    model.train()

    epoch_loss = 0.0

    for imgs, annotations in loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # 예측 오류 계산
            pred = model(imgs, annotations)
            losses = sum(loss for loss in pred.values())

            # 역전파
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        epoch_loss += losses

    return epoch_loss / len(loader)
