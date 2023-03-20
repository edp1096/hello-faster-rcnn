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


## ~~출처~~ 참고
* Face mask detection
    * https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch5-Faster-R-CNN.html
    * Dataset - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
* Detection
    * https://haochen23.github.io/2020/04/object-detection-faster-rcnn.html
* RPN 설명
    * https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
* 갖다 붙이기
    * https://www.kaggle.com/code/sovitrath/fasterrcnn-efficientnet-training
        * https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
    * https://medium.com/analytics-vidhya/fpn-feature-pyramid-networks-77d8be41817c
        * https://github.com/potterhsu/easy-fpn.pytorch
    * https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70