"""S3/MinIO client for uploading job output files."""

import logging
import os
import time
import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

_client = None
_bucket = None


def init():
    """Initialize S3 client from environment variables."""
    global _client, _bucket
    _client = boto3.client(
        's3',
        endpoint_url=os.environ.get('AWS_ENDPOINT', 'http://localhost:9000'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4'),
    )
    _bucket = os.environ.get('AWS_BUCKET', 'morphvox')
    logger.info(f"S3 client initialized: endpoint={os.environ.get('AWS_ENDPOINT')}, bucket={_bucket}")


def upload_file(local_path: str, s3_key: str, retries: int = 3) -> bool:
    """Upload a local file to S3/MinIO with retry. Returns True on success."""
    for attempt in range(retries):
        try:
            _client.upload_file(local_path, _bucket, s3_key)
            logger.info(f"Uploaded {local_path} → s3://{_bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.warning(f"S3 upload failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    logger.error(f"All {retries} upload attempts failed for {s3_key}")
    return False
