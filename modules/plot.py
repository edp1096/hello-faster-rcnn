import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


def plotImage(img, annotation):
    img = img.cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx].cpu()

        if annotation["labels"][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor="r", facecolor="none")
        elif annotation["labels"][idx] == 2:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor="g", facecolor="none")
        else:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor="orange", facecolor="none")

        ax.add_patch(rect)

    plt.show()


def plotImageCV(img, annotation, labels=None, colors=None):
    img = img.cpu().permute(1, 2, 0).numpy().copy()

    boxes_n = annotation["boxes"].cpu().numpy()
    boxes = []
    for i in range(len(boxes_n)):
        boxes.append([tuple([boxes_n[i][0], boxes_n[i][1]]), tuple([boxes_n[i][2], boxes_n[i][3]])])

    classes = annotation["labels"].cpu().numpy().tolist()

    scores = ["" for i in range(len(annotation["labels"]))]
    if "scores" in annotation:
        scores = annotation["scores"].tolist()

    img_predicted = combineImageWithPrediction(img.copy(), boxes, classes, scores, labels=labels, colors=colors)
    showImageResult(img, img_predicted)


def combineImageWithPrediction(img, boxes, classes, scores, labels=None, colors=None):
    rect_line_width = 2
    text_size = 1
    text_line_width = 1

    for i in range(len(boxes)):
        bbox_p1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        bbox_p2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))

        score = ""
        if scores[i] != "":
            score = f"{scores[i]:>.2f}"

        label = classes[i]
        if labels is not None:
            label = labels[classes[i]]

        color = (0, 255, 0)
        if colors is not None:
            color = colors[classes[i]]

        cv2.rectangle(img, bbox_p1, bbox_p2, color=color, thickness=rect_line_width)
        cv2.putText(img, f"{label} {score}", bbox_p1, cv2.FONT_HERSHEY_PLAIN, text_size, color, thickness=text_line_width)

    return img


def showImageResult(img_source, img_result):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Object detection")

    fig.tight_layout()
    fig.suptitle("Faster R-CNN result", fontsize=16)

    ax[0].imshow(img_source)
    ax[0].set_title("Source Image")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(img_result)
    ax[1].set_title("Result Image")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()
