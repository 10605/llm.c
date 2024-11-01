#!/bin/bash

# Check if enough arguments are provided
if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <model_depth> <model_channels> <model_heads> <model_family> <experiment_dir> <total_steps> <learning_rates (comma-separated)>"
    exit 1
fi

# Parse command-line arguments
model_depth=$1
model_channels=$2
model_heads=$3
model_family=$4
experiment_dir=$5
total_steps=$6
learning_rates=(${7//,/ })  # Convert comma-separated learning rates to array

# Create the main experiment directory if it doesn't exist
mkdir -p "$experiment_dir/$model_family"

# Loop through each learning rate and run the experiment
for lr in "${learning_rates[@]}"; do
    # Create a folder for this specific learning rate inside the model family folder
    lr_folder="${experiment_dir}/${model_family}/lr_${lr//./_}"
    mkdir -p "$lr_folder"

    echo "Running experiment in: $lr_folder, learning rate: $lr, depth: $model_depth, channels: $model_channels, heads: $model_heads, total steps: $total_steps"

    # Execute the training script with specified parameters
    ./train_gpt2cu \
        -i "/home/ubuntu/data/fineweb_train_*.bin" \
        -j "/home/ubuntu/data/fineweb_val_*.bin" \
        -o "$lr_folder" \
        -e "d12" \
        -b 64 -t 1024 \
        -d 524288 \
        -r 1 \
        -z 1 \
        -c 0.1 \
        -q 0.0 \
        -u 700 \
        -n 5000 \
        -v 250 -s 20000 \
        -h 1 \
        -1d "$model_depth" \
        -1c "$model_channels" \
        -1h "$model_heads" \
        -l "$lr" \
        -x "$total_steps"
done
