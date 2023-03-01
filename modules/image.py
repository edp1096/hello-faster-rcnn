import cv2

def getImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def cropImages(img, boxes):
    for i in range(len(boxes)):
        bbox_p1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        bbox_p2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))

        cropped = img[bbox_p1[1] : bbox_p2[1], bbox_p1[0] : bbox_p2[0]]

    return cropped


def saveImages(target_path, imgs, boxes, classes, scores):
    for i in range(len(boxes)):
        result = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{target_path}/{classes[i]}_{scores[i]:>.2f}.jpg", result)
