"""
s3_audit.py — Non-blocking Banking Compliance Audit Logger.
"""
import logging
from typing import Optional

logger = logging.getLogger("altur.s3")


class S3AuditService:
    def __init__(self, bucket_name: str = "altur-telephony-audit-vault", region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._s3_client = None

    def _get_client(self):
        if self._s3_client is None:
            try:
                import boto3
                self._s3_client = boto3.client("s3", region_name=self.region)
            except Exception:
                pass
        return self._s3_client

    async def upload_audio_async(self, audio_bytes: bytes, call_id: str):
        try:
            client = self._get_client()
            if client is not None:
                key = f"telephony_audit/{call_id}.wav"
                client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=audio_bytes,
                    ContentType="audio/wav"
                )
                logger.info(f"Successfully archived audit recording for {call_id} to S3.")
        except Exception as e:
            logger.debug(f"S3 audit logging bypassed: {e}")


s3_audit_service = S3AuditService()
