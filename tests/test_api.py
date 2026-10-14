"""End-to-end tests of the HTTP API using the mock engine (no GPU needed)."""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from videogen.engines import EngineManager
from videogen.hardware import Hardware
from videogen.jobs import JobManager
from videogen.server import create_app

from .conftest import ffprobe

CPU = Hardware("cpu", None, 0.0, 16.0)


@pytest.fixture
def client(settings):
    jobs = JobManager(settings, EngineManager(settings, CPU))
    with TestClient(create_app(settings, jobs)) as c:
        yield c
    jobs.shutdown()


def wait(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def png_bytes(size=(320, 180), color=(30, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_info_lists_engines(client):
    info = client.get("/api/info").json()
    ids = [e["id"] for e in info["engines"]]
    assert ids == ["ltx", "wan-5b", "wan-14b", "mock"]
    ltx = info["engines"][0]
    assert ltx["supports_audio"] and ltx["max_duration"] == 20
    assert any("GPU" in w for w in ltx["warnings"])
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_text_to_video_end_to_end(client):
    res = client.post("/api/jobs", data={"prompt": "a calm lake", "engine": "mock", "duration": "3",
                                         "resolution": "720p", "aspect": "16:9", "seed": "7"})  # fmt: skip
    assert res.status_code == 201, res.text
    job = wait(client, res.json()["id"])
    assert job["status"] == "done", job
    assert job["result"]["width"] == 1280 and job["result"]["height"] == 720
    assert job["result"]["has_audio"] and job["request"]["seed"] == 7
    assert job["result"]["enhanced_prompt"].startswith("a calm lake")
    video = client.get(job["video_url"])
    assert video.status_code == 200 and video.content[4:8] == b"ftyp"
    assert client.get(job["poster_url"]).headers["content-type"] == "image/jpeg"


def test_video_stream_properties(client, settings):
    res = client.post("/api/jobs", data={"prompt": "x", "engine": "mock", "duration": "2", "resolution": "1080p",
                                         "aspect": "9:16", "audio": "false", "smooth_fps": "48"})  # fmt: skip
    job = wait(client, res.json()["id"])
    streams = ffprobe(settings.output_dir / job["id"] / "video.mp4")
    assert streams["video"]["codec_name"] == "h264"
    assert (streams["video"]["width"], streams["video"]["height"]) == ("1080", "1920")
    assert streams["video"]["r_frame_rate"] == "48/1"
    assert "audio" not in streams


def test_image_to_video_with_auto_aspect(client):
    res = client.post(
        "/api/jobs",
        data={"prompt": "zoom in", "engine": "mock", "duration": "2", "resolution": "720p", "aspect": "auto"},
        files={"image": ("in.png", png_bytes((180, 320)), "image/png")},
    )
    assert res.status_code == 201, res.text
    job = wait(client, res.json()["id"])
    assert job["status"] == "done"
    assert job["request"]["mode"] == "i2v" and job["request"]["aspect"] == "9:16"
    assert (job["result"]["width"], job["result"]["height"]) == (720, 1280)
    assert client.get(job["input_url"]).status_code == 200


@pytest.mark.parametrize(
    "data,files,message",
    [
        ({"prompt": " ", "engine": "mock"}, None, "prompt"),
        ({"prompt": "x", "engine": "nope"}, None, "Unknown engine"),
        ({"prompt": "x", "engine": "mock", "duration": "25"}, None, "up to 20"),
        ({"prompt": "x", "engine": "wan-5b", "duration": "18"}, None, "up to 15"),
        ({"prompt": "x", "engine": "mock"}, {"image": ("a.png", b"not an image", "image/png")}, "Could not read"),
        ({"prompt": "x", "engine": "mock", "steps": "abc"}, None, "numbers"),
    ],
)
def test_validation_errors(client, data, files, message):
    res = client.post("/api/jobs", data=data, files=files)
    assert res.status_code == 400
    assert message in res.json()["detail"]


def test_failure_is_reported(client):
    job = wait(client, client.post("/api/jobs", data={"prompt": "fail", "engine": "mock"}).json()["id"])
    assert job["status"] == "failed" and "Mock failure" in job["error"]
    assert job["video_url"] is None


def test_engine_without_gpu_fails_cleanly(client):
    job = wait(client, client.post("/api/jobs", data={"prompt": "x", "engine": "wan-5b"}).json()["id"])
    assert job["status"] == "failed" and "needs a GPU" in job["error"]


def test_cancel_running_job(settings):
    settings.mock_step_delay = 0.2
    jobs = JobManager(settings, EngineManager(settings, CPU))
    with TestClient(create_app(settings, jobs)) as client:
        first = client.post("/api/jobs", data={"prompt": "a", "engine": "mock", "steps": "40"}).json()
        second = client.post("/api/jobs", data={"prompt": "b", "engine": "mock"}).json()
        assert client.get(f"/api/jobs/{second['id']}").json()["status"] == "queued"
        # cancel the queued one immediately, the running one once it has started
        assert client.post(f"/api/jobs/{second['id']}/cancel").json()["status"] == "cancelled"
        deadline = time.time() + 10
        while client.get(f"/api/jobs/{first['id']}").json()["status"] != "running" and time.time() < deadline:
            time.sleep(0.05)
        client.post(f"/api/jobs/{first['id']}/cancel")
        assert wait(client, first["id"])["status"] == "cancelled"
        assert client.delete(f"/api/jobs/{first['id']}").status_code == 204
        assert client.get(f"/api/jobs/{first['id']}").status_code == 404
        assert not (settings.output_dir / first["id"]).exists()
    jobs.shutdown()


def test_history_survives_restart_and_marks_interrupted(settings):
    jobs = JobManager(settings, EngineManager(settings, CPU))
    with TestClient(create_app(settings, jobs)) as client:
        done = wait(client, client.post("/api/jobs", data={"prompt": "keep", "engine": "mock"}).json()["id"])
    jobs.shutdown()
    # simulate a job that was mid-flight when the server died
    ghost = settings.output_dir / "ghost"
    ghost.mkdir()
    (ghost / "job.json").write_text(json.dumps({"id": "ghost", "request": {"prompt": "x"}, "created": 1.0,
                                                "status": "running"}))  # fmt: skip
    jobs = JobManager(settings, EngineManager(settings, CPU), start_worker=False)
    assert jobs.get(done["id"]).status == "done"
    assert jobs.get("ghost").status == "failed" and "restarted" in jobs.get("ghost").error


def test_password_protection(settings):
    settings.password = "s3cret"
    jobs = JobManager(settings, EngineManager(settings, CPU), start_worker=False)
    with TestClient(create_app(settings, jobs)) as client:
        assert client.get("/api/info").status_code == 401
        assert client.get("/api/info", auth=("any", "wrong")).status_code == 401
        assert client.get("/api/info", auth=("any", "s3cret")).status_code == 200
