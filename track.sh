#!/bin/bash

python track.py\
	--source ../daegu/1.mp4\
	--yolo_model yolov5s.pt\
	--save-txt\
	--save-vid\
	--classes 0 &

python track.py\
	--source ../daegu/2.mp4\
	--yolo_model yolov5s.pt\
	--save-txt\
	--save-vid\
	--classes 0 &

python track.py\
	--source ../daegu/3.mp4\
	--yolo_model yolov5s.pt\
	--save-txt\
	--save-vid\
	--classes 0 &