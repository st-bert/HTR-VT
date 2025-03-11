import os
import argparse
import shutil
from pathlib import Path
from utils.boto_lib import download_folder_from_s3
from dotenv import load_dotenv

load_dotenv()

def download_dataset(bucket_name: str, s3_folder: str, force: bool = False):
    """
    Download the dataset into data/custom_dataset directory.
    
    Args:
        force (bool): If True, delete existing dataset before downloading.
                     If False, raise error if dataset already exists.
    """

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    if aws_access_key_id is None:
        raise Exception("AWS_ACCESS_KEY_ID environment variable is not set")
    
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if aws_secret_access_key is None:
        raise Exception("AWS_SECRET_ACCESS_KEY environment variable is not set")
    
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")

    dataset_path = Path("data/custom_dataset")
    
    # Check if dataset directory exists and has content
    if dataset_path.exists() and any(dataset_path.iterdir()):
        if force:
            print("Force flag enabled. Removing existing dataset...")
            shutil.rmtree(dataset_path)
        else:
            raise Exception(
                "Dataset directory already contains files! "
                "Use -f/--force flag to override existing dataset."
            )
    
    # Create directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    
    download_folder_from_s3(
        bucket_name=bucket_name,
        s3_folder=s3_folder,
        local_dir=dataset_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
            
    print("Dataset downloaded successfully to data/custom_dataset/")

def main():
    parser = argparse.ArgumentParser(description="Download dataset for HTR-VT project")
    parser.add_argument(
        "bucket_name",
        type=str,
        help="Name of the S3 bucket containing the dataset"
    )
    parser.add_argument(
        "s3_folder",
        type=str,
        help="Folder path within the S3 bucket"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force download even if dataset already exists"
    )
    
    args = parser.parse_args()
    download_dataset(force=args.force)

if __name__ == "__main__":
    main() 