#!/bin/bash

BUCKET_NAME="YOUR_BUCKET_NAME_HERE"

./train_gpt2cu -o "question_1_experiment_one" -b 128 -1d 1 -1c 64 -1h 1 -l 0.01 -u 70 -x 1000
python upload_to_s3.py "$BUCKET_NAME" "question_1_experiment_one" "question_1_experiments"

./train_gpt2cu -o "question_1_experiment_two" -b 128 -1d 1 -1c 64 -1h 1 -l 0.001 -u 70 -x 1000
python upload_to_s3.py "$BUCKET_NAME" "question_1_experiment_two" "question_1_experiments"

sudo shutdown -h now
