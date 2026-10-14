"""FastAPI app: REST API + the single-page web UI."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from .config import Settings
from .core import GenerationRequest, UserError
from .engines import EngineManager
from .jobs import JobManager

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_IMAGE_SIDE = 4096


def _read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise UserError("Image is too large (max 40 MB).")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise UserError("Could not read the image. Use a PNG, JPEG or WebP file.") from None
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    if min(image.size) < 64:
        raise UserError("Image is too small (at least 64 pixels per side).")
    return image


def _checkbox(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")


def create_app(settings: Settings | None = None, jobs: JobManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if jobs is None:
        jobs = JobManager(settings, EngineManager(settings))
    engines = jobs.engines

    app = FastAPI(title="VideoGen", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.state.jobs = jobs

    if settings.password:

        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            header = request.headers.get("authorization", "")
            ok = False
            if header.lower().startswith("basic "):
                try:
                    _, _, password = base64.b64decode(header[6:]).decode().partition(":")
                    ok = secrets.compare_digest(password, settings.password)
                except (binascii.Error, UnicodeDecodeError):
                    ok = False
            if not ok:
                return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="VideoGen"'})
            return await call_next(request)

    @app.exception_handler(UserError)
    async def user_error(request: Request, exc: UserError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/info")
    def info():
        hw = engines.hw
        return {
            "version": __version__,
            "hardware": hw.to_dict(),
            "default_engine": engines.default_engine,
            "engines": engines.infos(),
        }

    @app.get("/api/jobs")
    def list_jobs(limit: int = 200):
        return [job.to_dict() for job in jobs.list()[: max(1, min(limit, 1000))]]

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        prompt: str = Form(...),
        engine: str = Form(None),
        negative_prompt: str = Form(""),
        duration: float = Form(5.0),
        resolution: str = Form("1080p"),
        aspect: str = Form("16:9"),
        seed: int = Form(-1),
        steps: str = Form(""),
        guidance: str = Form(""),
        audio: str = Form(None),
        enhance_prompt: str = Form(None),
        upscaler: str = Form("lanczos"),
        smooth_fps: int = Form(0),
        image: UploadFile | None = File(None),
    ):
        picture = None
        if image is not None and image.filename:
            picture = _read_image(await image.read())
        try:
            steps_value = int(steps) if steps.strip() else None
            guidance_value = float(guidance) if guidance.strip() else None
        except ValueError:
            raise UserError("Steps and guidance must be numbers.") from None
        req = GenerationRequest(
            prompt=prompt,
            engine=engine or engines.default_engine,
            negative_prompt=negative_prompt,
            duration=duration,
            resolution=resolution,
            aspect=aspect,
            seed=seed,
            steps=steps_value,
            guidance=guidance_value,
            audio=_checkbox(audio, True),
            enhance_prompt=_checkbox(enhance_prompt, True),
            upscaler=upscaler,
            smooth_fps=smooth_fps,
            image=picture,
        )
        return jobs.submit(req).to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id).to_dict()
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return jobs.cancel(job_id).to_dict()
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        try:
            jobs.delete(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found") from None
        return Response(status_code=204)

    app.mount("/files", StaticFiles(directory=jobs.root), name="files")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def serve(settings: Settings, open_browser: bool = False) -> None:
    import threading
    import webbrowser

    import uvicorn

    app = create_app(settings)
    host = settings.host
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1', '::') else host}:{settings.port}"
    if host not in ("127.0.0.1", "localhost", "::1") and not settings.password:
        log.warning(
            "VideoGen is listening on %s without a password. Anyone on your network can use your GPU. "
            "Set VIDEOGEN_PASSWORD to require one.",
            host,
        )
    print(f"\n  VideoGen {__version__} is running at {url}\n")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=settings.port, log_level="warning")
