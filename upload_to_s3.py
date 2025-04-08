import os
import boto3
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

def upload_to_s3(file_path, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket
    
    :param file_path: Path to the file to upload
    :param bucket_name: Name of the S3 bucket
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = Path(file_path).name
    
    # Create S3 client using credentials from .env
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    
    try:
        print(f"Uploading {file_path} to {bucket_name}/{object_name}...")
        s3_client.upload_file(file_path, bucket_name, object_name)
        print("Upload successful!")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload a file to an S3 bucket')
    parser.add_argument('file_path', help='Path to the file to upload')
    parser.add_argument('bucket_name', help='Name of the S3 bucket')
    parser.add_argument('--object-name', help='S3 object name (default: filename)')
    
    args = parser.parse_args()
    
    upload_to_s3(args.file_path, args.bucket_name, args.object_name) 