#!/bin/bash
# Bash control argumetns
LAST_NUM=3

# Tracker control arguments
YOLO_MODEL=yolov5s.pt

#DEEPSORT_PATH="None"
DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_market1501_74.4%.pth"
#DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_mars_76.0%.pth"
#DEEPSORT_PATH="./deep_sort/deep/checkpoint/osnet_x0_25_skku_88.2%.pth"

# Track
if [ "$DEEPSORT_PATH" = "None" ]
then
    # ImageNet Pretrained OSNet_x_25
    for ((idx=1 ; idx <= $LAST_NUM ; idx++));
    do
        python track.py\
        --source ../daegu/$idx.mp4 \
        --yolo_model $YOLO_MODEL \
        --save-txt \
        --save-vid \
        --evaluate \
        --exist-ok \
        --classes 0 &

    done
else
    # ReID Dataset finetuned OSNet_x_25
    for ((idx=1 ; idx <= $LAST_NUM ; idx++));
    do
        python track.py\
            --source ../daegu/$idx.mp4 \
            --yolo_model $YOLO_MODEL \
            --deep_sort_path $DEEPSORT_PATH \
            --save-txt \
            --save-vid \
            --evaluate \
            --exist-ok \
            --classes 0 &
    done
fi

# wait the finish of deepsort trackers
wait < <(jobs -p)

bash ./eval.sh