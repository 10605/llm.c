import os
import boto3
import sys

def upload_directory_to_s3(local_directory, s3_bucket, s3_directory):
    # Use the S3 resource
    s3 = boto3.resource('s3', region_name='us-east-1')
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # Only proceed if its .txt or .log file
            if file.endswith('.txt') or file.endswith('.log'):
                local_file_path = os.path.join(root, file)

                # Create the relative path to the file in S3
                relative_path = os.path.relpath(local_file_path, local_directory)
                s3_file_path = os.path.join(s3_directory, relative_path)
                try:
                    # Upload the file to S3
                    s3.Bucket(s3_bucket).upload_file(local_file_path, s3_file_path)
                    print(f'Uploaded {local_file_path} to s3://{s3_bucket}/{s3_file_path}')
                except Exception as e:
                    print(f'Error uploading {local_file_path}: {e}')

if __name__ == "__main__":
    # Check that we have the required arguments
    if len(sys.argv) < 4:
        print("Usage: python script.py <s3_bucket_name> <path_to_experiment_logs> <cluster_name>")
        sys.exit(1)

    # Get parameters from command line arguments
    s3_bucket_name = sys.argv[1]          # Your S3 bucket name
    path = sys.argv[2]                    # Path to your local folder
    cluster_name = sys.argv[3]            # Cluster name
    
    # Extract the run_name from the path
    run_name = os.path.basename(path)

    # Define the S3 target directory path
    s3_directory_name = f"experiments/{cluster_name}/{run_name}"

    # Upload the directory to the specified location in S3
    upload_directory_to_s3(path, s3_bucket_name, s3_directory_name)
    print("Done uploading to S3 Bucket")
