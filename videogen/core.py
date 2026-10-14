"""Shared data types: generation requests, raw engine output and progress reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
from PIL import Image

ASPECTS = ("16:9", "9:16", "1:1")
RESOLUTIONS = ("720p", "1080p")
UPSCALERS = ("lanczos", "realesrgan", "none")
MAX_PROMPT_CHARS = 4000


class Cancelled(Exception):
    """Raised inside a generation when the user cancels the job."""


class UserError(Exception):
    """An error whose message is safe and useful to show to the end user."""


@dataclass
class GenerationRequest:
    prompt: str
    engine: str = "ltx"
    negative_prompt: str = ""
    duration: float = 5.0
    resolution: str = "1080p"
    # "auto" picks the aspect closest to the reference image (image-to-video only).
    aspect: str = "16:9"
    seed: int = -1
    # None means "use the engine default".
    steps: int | None = None
    guidance: float | None = None
    audio: bool = True
    enhance_prompt: bool = True
    upscaler: str = "lanczos"
    # 0 disables motion interpolation; otherwise the output frame rate to interpolate to.
    smooth_fps: int = 0
    # Reference image for image-to-video. Never serialized.
    image: Image.Image | None = field(default=None, repr=False, compare=False)

    @property
    def mode(self) -> str:
        return "i2v" if self.image is not None else "t2v"

    def normalized(self) -> "GenerationRequest":
        """Return a copy with values validated and coerced to sane ranges.

        Raises UserError for values that cannot be fixed up silently.
        """
        prompt = (self.prompt or "").strip()
        if not prompt:
            raise UserError("Please enter a prompt describing the video.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise UserError(f"Prompt is too long (max {MAX_PROMPT_CHARS} characters).")
        if self.resolution not in RESOLUTIONS:
            raise UserError(f"Resolution must be one of {', '.join(RESOLUTIONS)}.")
        if self.aspect not in ASPECTS and self.aspect != "auto":
            raise UserError(f"Aspect ratio must be one of {', '.join(ASPECTS)} or auto.")
        if self.upscaler not in UPSCALERS:
            raise UserError(f"Upscaler must be one of {', '.join(UPSCALERS)}.")
        try:
            duration = float(self.duration)
        except (TypeError, ValueError):
            raise UserError("Duration must be a number of seconds.") from None
        if not 1.0 <= duration <= 30.0:
            raise UserError("Duration must be between 1 and 30 seconds.")
        steps = None if self.steps in (None, "", 0) else int(self.steps)
        if steps is not None and not 1 <= steps <= 100:
            raise UserError("Steps must be between 1 and 100.")
        guidance = None if self.guidance in (None, "") else float(self.guidance)
        if guidance is not None and not 0.0 <= guidance <= 20.0:
            raise UserError("Guidance must be between 0 and 20.")
        smooth = int(self.smooth_fps or 0)
        if smooth not in (0, 24, 30, 48, 60):
            raise UserError("Smooth motion target must be 24, 30, 48 or 60 fps (or off).")
        seed = int(self.seed) if self.seed not in (None, "") else -1
        aspect = self.aspect
        if aspect == "auto":
            aspect = closest_aspect(*self.image.size) if self.image is not None else "16:9"
        return GenerationRequest(
            prompt=prompt,
            engine=self.engine,
            negative_prompt=(self.negative_prompt or "").strip()[:MAX_PROMPT_CHARS],
            duration=round(duration, 2),
            resolution=self.resolution,
            aspect=aspect,
            seed=seed,
            steps=steps,
            guidance=guidance,
            audio=bool(self.audio),
            enhance_prompt=bool(self.enhance_prompt),
            upscaler=self.upscaler,
            smooth_fps=smooth,
            image=self.image,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("image", None)
        data["mode"] = self.mode
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], image: Image.Image | None = None) -> "GenerationRequest":
        names = {f.name for f in fields(cls)} - {"image"}
        return cls(**{k: v for k, v in data.items() if k in names}, image=image)


def closest_aspect(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    options = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
    # Compare in log space so 2:1 and 1:2 are treated symmetrically.
    return min(options, key=lambda k: abs(np.log(ratio / options[k])))


@dataclass
class RawVideo:
    """What an engine hands back before post-processing."""

    frames: np.ndarray  # uint8, shape [F, H, W, 3]
    fps: float
    audio: np.ndarray | None = None  # float32, shape [channels, samples], range [-1, 1]
    sample_rate: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(f"frames must be uint8 [F,H,W,3], got {self.frames.dtype} {self.frames.shape}")

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


ReportFn = Callable[[float, "str | None"], None]


class Progress:
    """Maps progress of a sub-task onto a slice of the overall [0, 1] range.

    Calling the object reports progress and is also the cancellation checkpoint:
    if the job was cancelled it raises Cancelled.
    """

    def __init__(
        self,
        report: ReportFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        self._report = report or (lambda fraction, message: None)
        self._should_cancel = should_cancel or (lambda: False)
        self.start = start
        self.end = end

    def __call__(self, fraction: float, message: str | None = None) -> None:
        self.check_cancel()
        fraction = min(max(float(fraction), 0.0), 1.0)
        self._report(self.start + (self.end - self.start) * fraction, message)

    def sub(self, a: float, b: float) -> "Progress":
        span = self.end - self.start
        return Progress(self._report, self._should_cancel, self.start + span * a, self.start + span * b)

    def check_cancel(self) -> None:
        if self._should_cancel():
            raise Cancelled()

    def step_callback(self, total_steps: int, label: str, after: str | None = None) -> Callable:
        """A diffusers `callback_on_step_end` that reports per-step progress and handles cancellation.

        `after` is shown once the last step finishes (e.g. "Decoding video") since pipelines keep
        working after their final callback.
        """

        def callback(pipe, step, timestep, callback_kwargs):
            done = step + 1
            message = after if (after and done >= total_steps) else f"{label} · step {done}/{total_steps}"
            self(done / max(total_steps, 1), message)
            return callback_kwargs

        return callback
