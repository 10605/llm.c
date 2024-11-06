#!/bin/bash

# ------------------------------- Experiment 1 --------------------------------

# Set the name of this set of experiments
cluster_name="testing_model_arch_1"

# Decide which variable to iterate over
learning_rates=(0.01 0.02 0.03)

# Create the directory for your experiment cluster
mkdir -p "$cluster_name"

# Initialize the experiment counter
experiment_count=1

# Loop through each learning rate and run the experiment
for lr in "${learning_rates[@]}"; do

    # Create a folder for this specific learning rate inside the cluster folder
    output_dir="${cluster_name}/experiment_${experiment_count}"

    mkdir -p "$output_dir"

    # Execute the training script with specified parameters
    ./train_gpt2cu \
        -i "/home/ubuntu/data/fineweb_train_*.bin" \
        -j "/home/ubuntu/data/fineweb_val_*.bin" \
        -o "$output_dir" \
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
        -1d 5 \
        -1c 320 \
        -1h 5 \
        -l "$lr" \
        -x 5

        # Increment the experiment counter
        ((experiment_count++))

    # Upload results to our S3 bucket
    bucket_name="10605chrisbucket"

    echo "Uploading to S3 Bucket"

    python3 upload_to_s3.py "$output_dir" "$bucket_name" "$cluster_name" "experiment_${experiment_count}"

# ------------------------------- Experiment 2 --------------------------------

# Set the name of this set of experiments
cluster_name="testing_model_arch_2"

# Decide which variable to iterate over
learning_rates=(0.01 0.02 0.03)

# Create the directory for your experiment cluster
mkdir -p "$cluster_name"

# Initialize the experiment counter
experiment_count=1

# Loop through each learning rate and run the experiment
for lr in "${learning_rates[@]}"; do

    # Create a folder for this specific learning rate inside the cluster folder
    output_dir="${cluster_name}/experiment_${experiment_count}"

    mkdir -p "$output_dir"

    # Execute the training script with specified parameters
    ./train_gpt2cu \
        -i "/home/ubuntu/data/fineweb_train_*.bin" \
        -j "/home/ubuntu/data/fineweb_val_*.bin" \
        -o "$output_dir" \
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
        -1d 4 \
        -1c 256 \
        -1h 4 \
        -l "$lr" \
        -x 5

        # Increment the experiment counter
        ((experiment_count++))

    # Upload results to our S3 bucket
    bucket_name="10605chrisbucket"

    echo "Uploading to S3 Bucket"

    python3 upload_to_s3.py "$output_dir" "$bucket_name" "$cluster_name" "experiment_${experiment_count}"


# ------------------------------- Initiate Shutdown --------------------------------
sudo shutdown -h now

done
