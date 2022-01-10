#!/bin/bash

DEEPSORT_PATH="None"
#DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_market1501_74.4%.pth"
#DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_mars_76.0%.pth"
#DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_skku_88.2%.pth"

# Track
if [ "$DEEPSORT_PATH" = "None" ]
then
    # ImageNet Pretrained OSNet_x_25
    python track.py\
        --source ../daegu/1.mp4 \
        --yolo_model yolov5s.pt \
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &

    python track.py\
        --source ../daegu/2.mp4 \
        --yolo_model yolov5s.pt \
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &

    python track.py\
        --source ../daegu/3.mp4\
        --yolo_model yolov5s.pt\
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &
else
    # ReID Dataset finetuned OSNet_x_25
    python track.py\
        --source ../daegu/1.mp4 \
        --yolo_model yolov5s.pt \
        --deep_sort_path $DEEPSORT_PATH\
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &

    python track.py\
        --source ../daegu/2.mp4 \
        --yolo_model yolov5s.pt \
        --deep_sort_path $DEEPSORT_PATH\
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &

    python track.py\
        --source ../daegu/3.mp4\
        --yolo_model yolov5s.pt\
        --deep_sort_path $DEEPSORT_PATH\
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &
fi