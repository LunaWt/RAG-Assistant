import threading
import time
import uuid
from dataclasses import dataclass

# Parsing owns a real span now: vision transcription runs one model call per batch of pages,
# so on a long PDF it is the slowest stage, not the instant one it was when pdfplumber did it.
PARSING_PERCENT = 40
CHUNKING_PERCENT = 45


@dataclass
class Job:
    id: str
    filename: str
    status: str = "pending"
    stage: str = "queued"
    done_pages: int = 0
    total_pages: int = 0
    done_chunks: int = 0
    total_chunks: int = 0
    chunks: int = 0
    error: str | None = None
    updated_at: float = 0.0


def _percent(job: Job) -> int:
    if job.status == "done":
        return 100
    if job.stage == "parsing":
        if not job.total_pages:
            return 0
        return int(PARSING_PERCENT * job.done_pages / job.total_pages)
    if job.stage == "chunking":
        return CHUNKING_PERCENT
    if job.stage == "embedding" and job.total_chunks:
        span = 100 - CHUNKING_PERCENT
        return CHUNKING_PERCENT + int(span * job.done_chunks / job.total_chunks)
    return 0


class JobStore:
    """Indexing progress, readable from the event loop while a worker thread writes it."""

    def __init__(self, ttl: float = 3600.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def create(self, filename: str) -> str:
        job = Job(id=uuid.uuid4().hex, filename=filename, updated_at=time.monotonic())
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        return job.id

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for name, value in fields.items():
                setattr(job, name, value)
            job.updated_at = time.monotonic()

    def snapshot(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.id,
                "filename": job.filename,
                "status": job.status,
                "stage": job.stage,
                "progress": _percent(job),
                "pages": job.total_pages,
                "chunks": job.chunks,
                "error": job.error,
            }

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for job_id in [
            j.id
            for j in self._jobs.values()
            if j.status in ("done", "error") and j.updated_at < cutoff
        ]:
            del self._jobs[job_id]


jobs = JobStore()
