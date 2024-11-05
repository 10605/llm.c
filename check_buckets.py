import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def list_s3_buckets():
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')
    
    try:
        # List all buckets
        response = s3.list_buckets()
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        
        if buckets:
            print("Accessible S3 Buckets:")
            for bucket in buckets:
                print(f"- {bucket}")
        else:
            print("No accessible S3 buckets found.")
    
    except NoCredentialsError:
        print("No AWS credentials found. Please configure your credentials.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials found. Please check your credentials.")
    except ClientError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_s3_buckets()
