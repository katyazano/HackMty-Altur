import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from src.api.config import settings

logger = logging.getLogger("altur.s3_audit")


class S3AuditService:
    """
    Asynchronous AWS S3 Service for Compliance and Telephony Audit Trail.
    """

    def __init__(self):
        self.bucket = settings.s3_audit_bucket
        self.region = settings.aws_region
        self.enabled = settings.enable_s3_audit
        self._s3_client = None

        if self.enabled:
            try:
                self._s3_client = boto3.client("s3", region_name=self.region)
                logger.info(f"S3AuditService initialized for bucket: {self.bucket} ({self.region})")
            except Exception as e:
                logger.warning(f"Could not initialize S3 client: {e}. S3 auditing will be bypassed.")
                self.enabled = False

    async def upload_audio_async(self, file_bytes: bytes, call_id: str) -> Optional[str]:
        """
        Uploads audio to S3 in a background thread executor without blocking inference.
        """
        if not self.enabled or self._s3_client is None:
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._upload_sync, file_bytes, call_id)

    def _upload_sync(self, file_bytes: bytes, call_id: str) -> Optional[str]:
        now = datetime.now(timezone.utc)
        key = f"raw-recordings/8khz/{now.strftime('%Y/%m/%d')}/{call_id}.wav"
        s3_uri = f"s3://{self.bucket}/{key}"

        try:
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_bytes,
                ContentType="audio/wav"
            )
            logger.info(f"Audited call_id={call_id} -> {s3_uri}")
            return s3_uri
        except (NoCredentialsError, ClientError, BotoCoreError) as e:
            logger.warning(f"S3 audit upload failed for call_id={call_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload for call_id={call_id}: {e}")
            return None


s3_audit_service = S3AuditService()
