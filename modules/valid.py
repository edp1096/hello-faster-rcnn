import torch


def run(device, dataloader, model):
    loss_total = 0.0

    model.eval()
    for imgs, annotations in dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        with torch.no_grad():
            pred = model(imgs, annotations)
            losses = sum(loss for loss in pred.values())

        loss_total += losses

    valid_loss = loss_total / len(dataloader)

    return valid_loss
