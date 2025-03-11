from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import urllib.parse
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
import logging, os
from tqdm import tqdm


def download_file_from_s3_url(s3_url, local_file_path, aws_access_key_id=None, aws_secret_access_key=None,
                              aws_session_token=None, endpoint_url=None):
    """Download a file from an S3 URL

    :param s3_url: S3 URL to download from
    :param local_file_path: Local file path to save the downloaded file
    :param aws_access_key_id: AWS access key
    :param aws_secret_access_key: AWS secret key
    :param aws_session_token: AWS session token, if using temporary credentials
    :param endpoint_url: S3 endpoint URL
    :return: None
    """

    # Parse the S3 URL
    parsed_url = urllib.parse.urlparse(s3_url)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')

    # Create an S3 client with explicit credentials if provided
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,  # This is optional and only needed for temporary credentials
            region_name='eu-central-1',
            endpoint_url=endpoint_url
        )
    else:
        # If no explicit credentials are provided, boto3 will look for credentials in the default locations
        s3 = boto3.client('s3', endpoint_url=endpoint_url)

    try:
        s3.download_file(bucket_name, object_key, local_file_path)
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError as e:
        print(f"An error occurred: with file {s3_url} : {e}")

def upload_file_to_s3_url(file_path, s3_url, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, endpoint_url=None):

    # Parse the S3 URL
    parsed_url = urllib.parse.urlparse(s3_url)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')

    # Create an S3 client with explicit credentials if provided

    # Create an S3 client with explicit credentials if provided
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,  # This is optional and only needed for temporary credentials
            region_name='eu-central-1' if endpoint_url is None else None,
            endpoint_url=endpoint_url,
            config=Config(request_checksum_calculation='when_required',
                          response_checksum_validation='when_required')
        )
    else:
        # If no explicit credentials are provided, boto3 will look for credentials in the default locations
        s3 = boto3.client('s3', endpoint_url=endpoint_url,
                          config=Config(request_checksum_calculation='when_required',
                                        response_checksum_validation='when_required'))

    # Set up progress callback
    file_size = os.stat(file_path).st_size

    def progress_callback(bytes_transferred):
        percentage = (bytes_transferred / file_size) * 100
        logging.info(f"Upload progress: {percentage:.2f}%")

    try:
        response = s3.upload_file(file_path, bucket_name, object_key, Callback=progress_callback)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except ClientError as e:
        print(f"An error occurred: {e} , while uploading file {file_path} to {s3_url}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e} , while uploading file {file_path} to {s3_url}")
        return False

    return True


def upload_file_to_s3(file_path, bucket_name, object_key, aws_access_key_id, aws_secret_access_key,
                      aws_session_token=None, endpoint_url=None):
    """Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket_name: Bucket to upload to
    :param object_key: S3 object name
    :param aws_access_key_id: AWS access key
    :param aws_secret_access_key: AWS secret key
    :param aws_session_token: AWS session token, if using temporary credentials
    :param endpoint_url: S3 endpoint URL
    :return: True if file was uploaded, else False
    """

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create an S3 client
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key,
                             aws_session_token=aws_session_token,
                             endpoint_url=endpoint_url,
                             config=Config(request_checksum_calculation='when_required',
                                           response_checksum_validation='when_required'))

    # Set up progress callback
    file_size = os.stat(file_path).st_size

    def progress_callback(bytes_transferred):
        percentage = (bytes_transferred / file_size) * 100
        logging.info(f"Upload progress: {percentage:.2f}%")

    try:
        response = s3_client.upload_file(file_path, bucket_name, object_key,
                                         Callback=progress_callback)
    except ClientError as e:
        logging.error(e)
        return False

    return True


def download_folder_from_s3(bucket_name, s3_folder, local_dir, aws_access_key_id, aws_secret_access_key,
                            aws_session_token=None, max_workers=10, endpoint_url=None):
    """Download all files in a folder from an S3 bucket

    :param bucket_name: S3 bucket name
    :param s3_folder: S3 folder to download
    :param local_dir: Local directory to save files
    :param aws_access_key_id: AWS access key
    :param aws_secret_access_key: AWS secret key
    :param aws_session_token: AWS session token, if using temporary credentials
    :param max_workers: Number of concurrent download threads
    :param endpoint_url: S3 endpoint URL
    :return: None
    """
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    s3 = session.client('s3', endpoint_url=endpoint_url)

    paginator = s3.get_paginator('list_objects_v2')

    # First, count the total number of files
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
    total_files = sum(len(page.get('Contents', [])) for page in pages)

    # Reset the paginator for the actual download
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page in pages:
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                base_name = obj['Key'][len(s3_folder):].replace('/','',1)
                if not base_name:
                    continue
                local_path = os.path.join(local_dir, base_name)
                futures.append(
                    executor.submit(download_file, s3, bucket_name, s3_key, local_path)
                )

        successful_downloads = 0
        with tqdm(total=total_files, unit='file') as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful_downloads += 1
                pbar.update(1)

    print(f"\nDownload complete. {successful_downloads} out of {total_files} files were successfully downloaded.")




def download_file(s3, bucket_name, s3_key, local_path):
    """Download a file from an S3 bucket
    :param s3: S3 client
    :param bucket_name: S3 bucket name
    :param s3_key: S3 object key
    :param local_path: Local file path
    :return: True if file was downloaded, else False
    """

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading {s3_key}: {e}")
        return False


def download_s3_bucket(bucket_name, local_dir, aws_access_key_id, aws_secret_access_key, aws_session_token=None,
                       max_workers=10, endpoint_url=None):
    """Download all files in an S3 bucket
    :param bucket_name: S3 bucket name
    :param local_dir: Local directory to save files
    :param aws_access_key_id: AWS access key
    :param aws_secret_access_key: AWS secret key
    :param aws_session_token: AWS session token, if using temporary credentials
    :param max_workers: Number of concurrent download threads
    :param endpoint_url: S3 endpoint URL
    :return: None
    """

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        endpoint_url=endpoint_url
    )
    s3 = session.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    # First, count the total number of files
    total_files = sum(len(page.get('Contents', [])) for page in pages)

    # Reset the paginator for the actual download
    pages = paginator.paginate(Bucket=bucket_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page in pages:
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                local_path = os.path.join(local_dir, s3_key)
                futures.append(
                    executor.submit(download_file, s3, bucket_name, s3_key, local_path)
                )

        successful_downloads = 0
        with tqdm(total=total_files, unit='file') as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful_downloads += 1
                pbar.update(1)

    print(f"\nDownload complete. {successful_downloads} out of {total_files} files were successfully downloaded.")


