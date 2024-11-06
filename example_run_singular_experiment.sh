/#!/bin/bash

# ------------------------------- Experiment 1 --------------------------------

# still need to set a name for this experiment cluster, even though its a singular experiment
cluster_name="experiment_1"

# Create the directory for your experiment cluster
mkdir -p "$cluster_name"

# Create a folder for this experiment 
output_dir="${cluster_name}/logfiles/"

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
    -l 0.0006 \
    -x 5

# Upload results to our S3 bucket
bucket_name="10605chrisbucket"

echo "Uploading to S3 Bucket"

python3 upload_to_s3.py "$output_dir" "$bucket_name" "$cluster_name" "logfiles"

# ------------------------------- Experiment 2 --------------------------------

# still need to set a name for this experiment cluster, even though its a singular experiment
cluster_name="experiment_2"

# Create the directory for your experiment cluster
mkdir -p "$cluster_name"

# Create a folder for this experiment 
output_dir="${cluster_name}/logfiles/"

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
    -1c 256 \
    -1h 5 \
    -l 0.0006 \
    -x 5

# Upload results to our S3 bucket
bucket_name="10605chrisbucket"

echo "Uploading to S3 Bucket"

python3 upload_to_s3.py "$output_dir" "$bucket_name" "$cluster_name" "logfiles"


# ------------------------------- Initiate Shutdown --------------------------------
sudo shutdown -h now

done
