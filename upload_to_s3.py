import os
import sys
import boto3

def upload_directory_to_s3(local_directory, s3_bucket, s3_directory):
    # Use the S3 resource
    s3 = boto3.resource('s3', region_name='us-east-1')

    for root, dirs, files in os.walk(local_directory):
        for file in files:
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
    local_experiments_directory = 'experiments'  # Path to your local experiments directory
    s3_directory_name = 'experiments'            # Folder in S3 where you want to upload

    s3_bucket_name = sys.argv[0]

    upload_directory_to_s3(local_experiments_directory, s3_bucket_name, s3_directory_name)
    print("Done uploading to S3 Bucket")