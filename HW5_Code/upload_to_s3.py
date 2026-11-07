"""Upload run logs to experiments/GROUP/RUN/ using normal AWS credentials."""
import argparse
from experiment_logs import upload_plan, upload_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bucket")
    p.add_argument("run_dir")
    p.add_argument("group")
    p.add_argument("--profile", help="optional local AWS profile; EC2 normally uses its instance role")
    p.add_argument("--include-checkpoints", action="store_true", help="also upload model/state/DONE files; logs only by default")
    p.add_argument("--dry-run", action="store_true", help="list uploads without credentials or network access")
    args = p.parse_args()
    try:
        plan = upload_plan(args.run_dir, args.group, args.include_checkpoints)
    except ValueError as e:
        p.error(str(e))
    for source, key in plan:
        print(f"{source} -> s3://{args.bucket}/{key}")
    if args.dry_run:
        return
    import boto3
    client = boto3.Session(profile_name=args.profile).client("s3")
    upload_run(client, args.bucket, args.run_dir, args.group, args.include_checkpoints)
    print(f"Uploaded {len(plan)} files.")


if __name__ == "__main__":
    main()
