import torch



def getPredictionsCOCO(img, model, threshold, class_names):
    model.eval()
    with torch.no_grad():
        preds = model([img])

    scores = list(preds[0]["scores"].cpu().detach().numpy())

    cutline_idx = 0
    for x in scores:
        if x > threshold:
            cutline_idx = scores.index(x)

    classes = []
    # for i in list(preds[0]["labels"].cpu().detach().numpy()):
    for i in list(preds[0]["labels"]):
        classes.append(class_names[i])

    boxes = []
    # for i in list(preds[0]["boxes"].cpu().detach().numpy()):
    for i in list(preds[0]["boxes"]):
        boxes.append([(i[0], i[1]), (i[2], i[3])])

    boxes = boxes[: cutline_idx + 1]
    classes = classes[: cutline_idx + 1]

    return boxes, classes, scores


def getPredictions(model, img, threshold):
    model.eval()

    with torch.no_grad():
        preds = model(img)

    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]["scores"]):
            if score > threshold:
                idx_list.append(idx)

        preds[id]["boxes"] = preds[id]["boxes"][idx_list]
        preds[id]["labels"] = preds[id]["labels"][idx_list]
        preds[id]["scores"] = preds[id]["scores"][idx_list]

    return preds
