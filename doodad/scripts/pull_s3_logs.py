import argparse
import os
import subprocess


def aws_sync(bucket_name, s3_log_dir, target_dir, exclude="*.pkl"):
    cmd = "aws s3 sync s3://%s/doodad/logs/%s %s --exclude %s" % (
        bucket_name,
        s3_log_dir,
        target_dir,
        exclude,
    )
    subprocess.call(cmd, shell=True)


def main():

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("log_dir", type=str, help="S3 Log dir")
    parser.add_argument("-b", "--bucket", type=str, default="doodad", help="S3 Bucket")
    parser.add_argument("-e", "--exclude", type=str, default="*.pkl", help="Exclude")

    args = parser.parse_args()
    s3_log_dir = args.log_dir
    os.makedirs(s3_log_dir, exist_ok=True)
    aws_sync(args.bucket, s3_log_dir, s3_log_dir, exclude=args.exclude)


if __name__ == "__main__":
    main()
