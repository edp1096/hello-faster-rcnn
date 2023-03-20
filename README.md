# Faster R-CNN 찍먹


## 찍먹들

* ArcFace - https://github.com/edp1096/hello-arcface
* Faster R-CNN - https://github.com/edp1096/hello-faster-rcnn
* Facial keypoints - https://github.com/edp1096/hello-keypoint


## 실행

* 마스크 데이터셋 분리
```sh
python3 scripts/split_dataset.py
```

* 마스크
```sh
python3 ./train_model.py
python3 ./test_model.py
```

* Detect object using COCO pretrained
```sh
python3 ./detect_coco.py
```


## 출처
* Face mask detection
    * https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch5-Faster-R-CNN.html
    * Dataset - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
* Detection
    * https://haochen23.github.io/2020/04/object-detection-faster-rcnn.html
* https://www.kaggle.com/code/sovitrath/fasterrcnn-efficientnet-training
    * https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
