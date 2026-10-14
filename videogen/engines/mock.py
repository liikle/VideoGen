"""A fast, dependency-free fake engine for tests and UI development (VIDEOGEN_ENABLE_MOCK=1).

Text-to-video renders a moving gradient; image-to-video does a slow Ken Burns zoom
over the reference image. It also produces a stereo tone so the audio path is exercised.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from ..core import GenerationRequest, Progress, RawVideo
from ..sizing import fit_area, fit_image, frames_for
from .base import Engine


class MockEngine(Engine):
    id = "mock"
    label = "Mock engine (testing)"
    tagline = "Instant fake output for testing the app without a GPU"
    description = "Produces synthetic clips with a test tone. Enabled with VIDEOGEN_ENABLE_MOCK=1."
    license = "MIT"
    license_url = ""
    fps = 24
    max_duration = 20.0
    supports_audio = True
    supports_prompt_enhancement = True
    tunable = True
    default_steps = 8
    default_guidance = 1.0
    native_resolution = "360p (upscaled)"
    requires_gpu = False
    plan_table = [(0, "none", "none")]

    def download(self, progress=None, force=False) -> None:
        return None

    def is_downloaded(self) -> bool:
        return True

    def load(self, progress: Progress) -> None:
        progress(1.0, "Mock engine ready")
        self.loaded = True

    def generate(self, req: GenerationRequest, progress: Progress) -> RawVideo:
        if req.prompt.strip().lower() == "fail":
            raise RuntimeError("Mock failure requested by prompt")
        width, height = fit_area(req.aspect, 640 * 360, 16)
        count = frames_for(req.duration, self.fps, 8)
        steps = req.steps or self.default_steps
        for step in range(steps):
            time.sleep(self.settings.mock_step_delay)
            progress((step + 1) / steps, f"Denoising · step {step + 1}/{steps}")

        t = np.linspace(0.0, 1.0, count, dtype=np.float32)
        if req.image is not None:
            base = np.asarray(fit_image(req.image, width * 2, height * 2), dtype=np.uint8)
            frames = np.empty((count, height, width, 3), dtype=np.uint8)
            for i, ti in enumerate(t):
                zoom = 1.0 + 0.15 * ti
                cw, ch = int(width * 2 / zoom), int(height * 2 / zoom)
                x0, y0 = (width * 2 - cw) // 2, (height * 2 - ch) // 2
                crop = Image.fromarray(base[y0 : y0 + ch, x0 : x0 + cw]).resize((width, height), Image.BILINEAR)
                frames[i] = np.asarray(crop)
        else:
            seed = (req.seed if req.seed >= 0 else 0) % 360
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            frames = np.empty((count, height, width, 3), dtype=np.uint8)
            for i, ti in enumerate(t):
                phase = 2 * np.pi * (ti + seed / 360.0)
                r = 127 + 120 * np.sin(xx / width * 6 + phase)
                g = 127 + 120 * np.sin(yy / height * 6 - phase)
                b = 127 + 120 * np.sin((xx + yy) / (width + height) * 6 + 2 * phase)
                frames[i] = np.stack([r, g, b], -1).astype(np.uint8)

        audio, sample_rate = None, None
        if req.audio:
            sample_rate = 48000
            seconds = count / self.fps
            ts = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
            tone = 0.2 * np.sin(2 * np.pi * 440 * ts) * np.minimum(1, ts * 4)
            audio = np.stack([tone, np.roll(tone, 120)]).astype(np.float32)
        info = {"generated_size": [width, height], "num_frames": count}
        if req.enhance_prompt:
            info["enhanced_prompt"] = f"{req.prompt} (mock enhanced)"
        return RawVideo(frames=frames, fps=self.fps, audio=audio, sample_rate=sample_rate, info=info)
