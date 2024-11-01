#!/bin/bash

# Define the experiment directory (DON'T CHANGE THIS)
experiment_dir="experiments"

# ------------------------ Experiment 1 ------------------------
# Parameters for Experiment 1
model_depth=3
model_channels=192
model_heads=3
model_family="tiny"
total_steps=5
learning_rates="0.01,0.02,0.03,0.05,0.05,0.06"

# Run Experiment 1
./run_experiment.sh "$model_depth" "$model_channels" "$model_heads" "$model_family" "$experiment_dir" "$total_steps" "$learning_rates"


# ------------------------ Experiment 2 ------------------------
# Parameters for Experiment 2
model_depth=2
model_channels=128
model_heads=2
model_family="tinier"
total_steps=5
learning_rates="0.006,0.007,0.008,0.009,0.01,0.011,0.012"

# Run Experiment 2
./run_experiment.sh "$model_depth" "$model_channels" "$model_heads" "$model_family" "$experiment_dir" "$total_steps" "$learning_rates"


# ------------------------ Experiment 3 ------------------------
# Parameters for Experiment 3
model_depth=1
model_channels=64
model_heads=1
model_family="tiniest"
total_steps=5
learning_rates="0.013,0.014,0.015,0.016,0.017,0.018,0.019"

# Run Experiment 3
./run_experiment.sh "$model_depth" "$model_channels" "$model_heads" "$model_family" "$experiment_dir" "$total_steps" "$learning_rates"

# ------------------------ End of Experiments ------------------------

# Define your S3 bucket name, 
s3_bucket_name = 'your-bucket-name'  # Replace with your actual bucket name

# Upload experiment data to S3 bucket
echo "Uploading Experiment data to $s3_bucket_name bucket"
python3 upload_to_s3.py "$s3_bucket_name"

# Shutdown the instance
sudo shutdown -h now
