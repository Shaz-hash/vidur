# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..config import MultipleProcessTrainingConfig


def _require_boto3():
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required for the distributed Network pipeline. Install boto3 on both head and worker nodes."
        ) from exc
    return boto3


def join_s3_key(*parts: str) -> str:
    cleaned = [str(p).strip("/").replace("\\", "/") for p in parts if str(p).strip("/")]
    return "/".join(cleaned)


class AwsNetworkBackend:
    def __init__(self, cfg: MultipleProcessTrainingConfig) -> None:
        boto3 = _require_boto3()
        region = str(cfg.network.aws_region)
        self.cfg = cfg
        self.s3 = boto3.client("s3", region_name=region)
        self.sqs = boto3.client("sqs", region_name=region)
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.job_table = self.dynamodb.Table(str(cfg.network.dynamodb_job_table))
        self.worker_table = self.dynamodb.Table(str(cfg.network.dynamodb_worker_table))

    @property
    def bucket(self) -> str:
        return str(self.cfg.network.s3_bucket)

    @property
    def root_prefix(self) -> str:
        return str(self.cfg.network.s3_prefix).strip("/")

    def model_key(self, session_id: str, generation: int, filename: str) -> str:
        return join_s3_key(
            self.root_prefix,
            self.cfg.network.s3_model_prefix,
            session_id,
            f"gen_{generation:06d}",
            filename,
        )

    def result_prefix(self, session_id: str, task_id: str) -> str:
        return join_s3_key(
            self.root_prefix,
            self.cfg.network.s3_result_prefix,
            session_id,
            task_id,
        )

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        local_path = Path(local_path)
        self.s3.upload_file(str(local_path), self.bucket, str(s3_key))

    def download_file(self, s3_key: str, local_path: Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, str(s3_key), str(local_path))
        return local_path

    def upload_relative_files(self, local_root: Path, rel_paths: Iterable[str], s3_prefix: str) -> List[str]:
        uploaded: List[str] = []
        for rel in sorted({str(p).replace("\\", "/") for p in rel_paths}):
            src = Path(local_root) / rel
            if not src.exists() or not src.is_file():
                continue
            key = join_s3_key(s3_prefix, rel)
            self.upload_file(src, key)
            uploaded.append(rel)
        return uploaded

    def download_relative_files(self, local_root: Path, rel_paths: Iterable[str], s3_prefix: str) -> List[Path]:
        downloaded: List[Path] = []
        for rel in sorted({str(p).replace("\\", "/") for p in rel_paths}):
            dst = Path(local_root) / rel
            self.download_file(join_s3_key(s3_prefix, rel), dst)
            downloaded.append(dst)
        return downloaded

    def send_job(self, payload: Dict[str, Any]) -> None:
        self.sqs.send_message(
            QueueUrl=str(self.cfg.network.sqs_job_queue_url),
            MessageBody=json.dumps(payload),
        )

    def send_result(self, payload: Dict[str, Any]) -> None:
        self.sqs.send_message(
            QueueUrl=str(self.cfg.network.sqs_result_queue_url),
            MessageBody=json.dumps(payload),
        )

    def receive_jobs(self, *, max_number: int = 1, visibility_timeout: int | None = None) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "QueueUrl": str(self.cfg.network.sqs_job_queue_url),
            "MaxNumberOfMessages": int(max_number),
            "WaitTimeSeconds": int(self.cfg.network.worker_poll_wait_sec),
        }
        if visibility_timeout is not None:
            kwargs["VisibilityTimeout"] = int(visibility_timeout)
        resp = self.sqs.receive_message(**kwargs)
        return list(resp.get("Messages", []))

    def receive_results(self, *, max_number: int = 10, wait_time_sec: int | None = None) -> List[Dict[str, Any]]:
        resp = self.sqs.receive_message(
            QueueUrl=str(self.cfg.network.sqs_result_queue_url),
            MaxNumberOfMessages=int(max_number),
            WaitTimeSeconds=int(wait_time_sec or self.cfg.network.result_poll_timeout_sec),
        )
        return list(resp.get("Messages", []))

    def delete_job_message(self, receipt_handle: str) -> None:
        self.sqs.delete_message(
            QueueUrl=str(self.cfg.network.sqs_job_queue_url),
            ReceiptHandle=str(receipt_handle),
        )

    def delete_result_message(self, receipt_handle: str) -> None:
        self.sqs.delete_message(
            QueueUrl=str(self.cfg.network.sqs_result_queue_url),
            ReceiptHandle=str(receipt_handle),
        )

    def change_job_visibility(self, receipt_handle: str, timeout_sec: int) -> None:
        self.sqs.change_message_visibility(
            QueueUrl=str(self.cfg.network.sqs_job_queue_url),
            ReceiptHandle=str(receipt_handle),
            VisibilityTimeout=int(timeout_sec),
        )

    def put_job_record(self, *, task_id: str, session_id: str, generation: int, task_type: str, status: str, worker_id: str = "", extra: Dict[str, Any] | None = None) -> None:
        item: Dict[str, Any] = {
            "job_id": str(task_id),
            "session_id": str(session_id),
            "generation": int(generation),
            "task_type": str(task_type),
            "status": str(status),
            "updated_at": int(time.time()),
        }
        if worker_id:
            item["worker_id"] = str(worker_id)
        if extra:
            item.update(extra)
        self.job_table.put_item(Item=item)

    def heartbeat_worker(self, *, worker_id: str, status: str, current_task_id: str = "") -> None:
        item: Dict[str, Any] = {
            "worker_id": str(worker_id),
            "hostname": socket.gethostname(),
            "status": str(status),
            "current_task_id": str(current_task_id),
            "last_heartbeat": int(time.time()),
        }
        self.worker_table.put_item(Item=item)

    def active_workers(self) -> List[Dict[str, Any]]:
        resp = self.worker_table.scan()
        items = list(resp.get("Items", []))
        cutoff = int(time.time()) - int(self.cfg.network.worker_stale_after_sec)
        return [item for item in items if int(item.get("last_heartbeat", 0)) >= cutoff]

    def wait_for_workers(self, *, min_count: int = 1, timeout_sec: int) -> List[Dict[str, Any]]:
        deadline = time.time() + float(timeout_sec)
        while True:
            active = self.active_workers()
            if len(active) >= int(min_count):
                return active
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for remote workers: required={min_count}, active={len(active)}"
                )
            time.sleep(5.0)
