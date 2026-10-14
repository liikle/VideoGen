"""Drive the web UI in headless Chromium against a live server running the mock engine."""

import os
import shutil
import socket
import threading
import time

import pytest
from PIL import Image

playwright_api = pytest.importorskip("playwright.sync_api")

from videogen.engines import EngineManager  # noqa: E402
from videogen.hardware import Hardware  # noqa: E402
from videogen.jobs import JobManager  # noqa: E402
from videogen.server import create_app  # noqa: E402

pytestmark = pytest.mark.browser


@pytest.fixture
def server(settings):
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    jobs = JobManager(settings, EngineManager(settings, Hardware("cuda", "Test GPU", 24.0, 64.0)))
    config = uvicorn.Config(create_app(settings, jobs), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)
    jobs.shutdown()


@pytest.fixture
def page():
    with playwright_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - depends on the machine
            # Fall back to a system Chromium when Playwright's own build isn't installed.
            fallback = os.environ.get("CHROMIUM_PATH") or shutil.which("chromium") or "/opt/pw-browsers/chromium"
            if not os.path.exists(fallback):
                pytest.skip(f"Chromium not available: {exc}")
            browser = p.chromium.launch(executable_path=fallback)
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        yield page
        browser.close()
        assert not errors, errors


def test_text_then_image_to_video(server, page, tmp_path):
    page.goto(server)
    page.wait_for_selector("#engine-mock")
    assert "Test GPU" in page.inner_text("#hw")
    assert page.locator(".engine").count() == 4

    page.check("#engine-mock", force=True)
    page.fill("#prompt", "A paper boat drifting down a rainy street")
    page.locator("#duration").fill("2")
    page.click("#resolution [data-value='720p']")
    assert "1280×720" in page.inner_text("#estimate")
    page.click("#generate")
    page.wait_for_selector("#player:not([hidden])", timeout=30000)
    can_play_h264 = page.evaluate(
        """document.createElement('video').canPlayType('video/mp4; codecs="avc1.640028, mp4a.40.2"')"""
    )
    if can_play_h264:  # open-source Chromium builds ship without an H.264 decoder
        page.wait_for_function("document.querySelector('#player').readyState >= 1", timeout=10000)
    video_url = page.evaluate("document.querySelector('#player').src")
    response = page.request.get(video_url)
    assert response.ok and response.headers["content-type"] == "video/mp4"
    assert page.locator(".card").count() == 1
    assert "A paper boat" in page.inner_text("#detail-prompt")
    assert page.get_attribute("#btn-download", "href").endswith("video.mp4")

    # Image to video: switch mode, upload, generate.
    page.click("#mode-i2v")
    image = tmp_path / "ref.png"
    Image.new("RGB", (400, 400), (200, 80, 40)).save(image)
    page.set_input_files("#image-input", str(image))
    page.wait_for_selector("#dz-preview:not([hidden])")
    assert page.get_attribute("#aspect-auto", "aria-checked") == "true"
    page.fill("#prompt", "Slow push in")
    page.click("#generate")
    page.wait_for_function("document.querySelectorAll('.card').length === 2")
    page.wait_for_function(
        "document.querySelector('#player:not([hidden])') && document.querySelector('#player').src.includes('video.mp4')"
        " && !document.querySelector('.card .status.running')",
        timeout=30000,
    )
    assert "image → video" in page.inner_text("#detail-chips")
    assert "720×720" in page.inner_text("#detail-chips")

    # Reuse settings of the first job brings back text mode and its prompt.
    page.locator(".card").nth(1).click()
    page.click("#btn-reuse")
    assert page.input_value("#prompt").startswith("A paper boat")
    assert page.get_attribute("#mode-t2v", "aria-selected") == "true"

    page.screenshot(path=str(tmp_path / "desktop.png"), full_page=True)


def test_validation_message_and_mobile_layout(server, page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(server)
    page.wait_for_selector("#engine-mock")
    page.click("#generate")
    assert "describe the video" in page.inner_text("#form-error")
    page.click("#mode-i2v")
    page.fill("#prompt", "x")
    page.click("#generate")
    assert "reference image" in page.inner_text("#form-error")
    # no horizontal scrolling on phones
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
