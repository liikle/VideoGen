"""Job queue: a single worker thread owns the GPU and runs jobs one at a time.

Each job lives in its own folder under the output directory (job.json, input.png,
video.mp4, poster.jpg), so history survives restarts.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .core import Cancelled, GenerationRequest, Progress, UserError
from .engines import EngineManager
from .postprocess import finalize

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)


@dataclass
class Job:
    id: str
    request: dict[str, Any]
    created: float
    status: str = QUEUED
    progress: float = 0.0
    message: str = "Waiting in queue"
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    has_input_image: bool = False
    result: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("cancel_requested")
        base = f"/files/{self.id}"
        data["video_url"] = f"{base}/video.mp4" if self.status == DONE else None
        data["poster_url"] = f"{base}/poster.jpg" if self.status == DONE else None
        data["input_url"] = f"{base}/input.png" if self.has_input_image else None
        if self.started:
            data["elapsed"] = round((self.finished or time.time()) - self.started, 1)
        return data


def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, UserError):
        return str(exc)
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower() or type(exc).__name__ == "OutOfMemoryError":
        return (
            "The GPU ran out of memory. Try 720p instead of 1080p, a shorter duration, or a lower-memory "
            "setting (VIDEOGEN_QUANTIZE=nf4 or VIDEOGEN_OFFLOAD=sequential), and close other GPU apps."
        )
    return text


class JobManager:
    def __init__(self, settings: Settings, engines: EngineManager, start_worker: bool = True) -> None:
        self.settings = settings
        self.engines = engines
        self.root = Path(settings.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.cv = threading.Condition()
        self._load_history()
        self._stop = False
        self.worker = threading.Thread(target=self._work, name="videogen-worker", daemon=True)
        if start_worker:
            self.worker.start()

    # ---- persistence --------------------------------------------------------------

    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        path = self._dir(job.id) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), indent=1))
        tmp.replace(path)

    def _load_history(self) -> None:
        loaded = []
        for path in self.root.glob("*/job.json"):
            try:
                job = Job(**json.loads(path.read_text()))
            except Exception as exc:
                log.warning("Skipping unreadable job file %s: %s", path, exc)
                continue
            if job.status not in FINISHED:
                job.status, job.error, job.message = FAILED, "Interrupted: the server was restarted.", "Interrupted"
                job.finished = job.finished or time.time()
                self._save(job)
            loaded.append(job)
        for job in sorted(loaded, key=lambda j: j.created):
            self.jobs[job.id] = job

    # ---- public API ------------------------------------------------------------------

    def submit(self, req: GenerationRequest) -> Job:
        engine = self.engines.get(req.engine)
        req = req.normalized()
        if req.duration > engine.max_duration:
            raise UserError(f"{engine.label} supports clips up to {engine.max_duration:g} seconds.")
        if req.seed < 0:
            req.seed = random.randint(0, 2**31 - 1)
        if not engine.supports_audio:
            req.audio = False
        if not engine.supports_prompt_enhancement:
            req.enhance_prompt = False
        job = Job(
            id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            request=req.to_dict(),
            created=time.time(),
            has_input_image=req.image is not None,
        )
        folder = self._dir(job.id)
        folder.mkdir(parents=True)
        if req.image is not None:
            req.image.convert("RGB").save(folder / "input.png")
        self._save(job)
        with self.cv:
            self.jobs[job.id] = job
            self.queue.append(job.id)
            self._update_queue_messages()
            self.cv.notify_all()
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self.cv:
            if job.status == QUEUED:
                self.queue.remove(job.id)
                job.status, job.message, job.finished = CANCELLED, "Cancelled", time.time()
                self._save(job)
                self._update_queue_messages()
            elif job.status == RUNNING:
                job.cancel_requested = True
                job.message = "Cancelling…"
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status in (QUEUED, RUNNING):
            raise UserError("Cancel the job before deleting it.")
        with self.cv:
            self.jobs.pop(job_id, None)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    def shutdown(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    # ---- worker ----------------------------------------------------------------------

    def _update_queue_messages(self) -> None:
        for position, job_id in enumerate(self.queue, 1):
            job = self.jobs[job_id]
            job.message = "Next up" if position == 1 else f"Queued · position {position}"

    def _work(self) -> None:
        while True:
            with self.cv:
                while not self.queue and not self._stop:
                    self.cv.wait()
                if self._stop:
                    return
                job = self.jobs[self.queue.popleft()]
                job.status, job.started, job.message = RUNNING, time.time(), "Starting"
                self._update_queue_messages()
            self._save(job)
            try:
                self._run(job)
            except BaseException as exc:  # the worker must survive anything a job throws
                job.finished = time.time()
                if isinstance(exc, Cancelled):
                    job.status, job.message = CANCELLED, "Cancelled"
                else:
                    log.error("Job %s failed:\n%s", job.id, traceback.format_exc())
                    job.status, job.message, job.error = FAILED, "Failed", _friendly_error(exc)
                (self._dir(job.id) / "video.mp4").unlink(missing_ok=True)
            self._save(job)

    def _run(self, job: Job) -> None:
        folder = self._dir(job.id)
        image = None
        if job.has_input_image:
            with Image.open(folder / "input.png") as img:
                image = img.convert("RGB")
        req = GenerationRequest.from_dict(job.request, image=image)

        def report(fraction: float, message: str | None) -> None:
            job.progress = round(max(job.progress, fraction), 4)
            if message:
                job.message = message

        progress = Progress(report, lambda: job.cancel_requested)
        started = time.time()
        engine = self.engines.acquire(req.engine, progress.sub(0.0, 0.1))
        load_seconds = time.time() - started
        progress(0.0, "Generating")
        raw = engine.generate(req, progress.sub(0.1, 0.92))
        generate_seconds = time.time() - started - load_seconds
        output = finalize(
            raw,
            req,
            folder / "video.mp4",
            progress.sub(0.92, 1.0),
            device=self.engines.hw.device,
            poster_path=folder / "poster.jpg",
        )
        job.result = {
            **output,
            **raw.info,
            "seed": req.seed,
            "engine_label": engine.label,
            "load_seconds": round(load_seconds, 1),
            "generate_seconds": round(generate_seconds, 1),
        }
        job.status, job.progress, job.message, job.finished = DONE, 1.0, "Done", time.time()
