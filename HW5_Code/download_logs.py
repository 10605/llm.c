"""Download a group of S3 run logs for local plotting."""
import argparse
from experiment_logs import download_logs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bucket")
    p.add_argument("group")
    p.add_argument("--dest", default="downloaded_logs")
    p.add_argument("--profile")
    args = p.parse_args()
    import boto3
    client = boto3.Session(profile_name=args.profile).client("s3")
    paths = download_logs(client, args.bucket, args.group, args.dest)
    if not paths:
        p.error("No run logs found at the requested experiments/GROUP/ prefix")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
