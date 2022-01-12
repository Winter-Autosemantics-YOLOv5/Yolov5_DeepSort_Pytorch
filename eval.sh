# remove already exist .txt files
# rm ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/*.txt

# copy Yolov5 + DeepSort results
cp ./runs/track/exp/1.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data

cp ./runs/track/exp/2.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data

cp ./runs/track/exp/3.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data

# rename model results
mv ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/1.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/seq-01.txt

mv ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/2.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/seq-02.txt

mv ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/3.txt \
    ./MOT16_eval/TrackEval/data/trackers/mot_challenge/MyDataset-test/data/data/seq-03.txt

# evaluate track results

python ./MOT16_eval/TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK MyDataset \
    --SPLIT_TO_EVAL test \
    --METRICS CLEAR Identity \
    --DO_PREPROC False\
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 4