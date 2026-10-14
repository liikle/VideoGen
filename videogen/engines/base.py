"""Engine interface plus helpers shared by the diffusers-based engines.

torch/diffusers are imported lazily so the web server starts instantly and the
mock engine works without them.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import Settings
from ..core import GenerationRequest, Progress, RawVideo, UserError
from ..hardware import Hardware, MemoryPlan, PlanTable, choose_plan

log = logging.getLogger(__name__)


class Engine(ABC):
    id: str = ""
    label: str = ""
    tagline: str = ""
    description: str = ""
    license: str = ""
    license_url: str = ""
    fps: int = 24
    max_duration: float = 15.0
    supports_audio: bool = False
    supports_prompt_enhancement: bool = False
    # Whether steps/guidance are meaningful knobs for this engine (distilled models use fixed schedules).
    tunable: bool = True
    default_steps: int | None = None
    default_guidance: float | None = None
    native_resolution: str = ""
    download_gb: float = 0.0
    min_vram_gb: float = 0.0
    requires_gpu: bool = True
    plan_table: PlanTable = [(0, "model", "none")]

    def __init__(self, settings: Settings, hw: Hardware) -> None:
        self.settings = settings
        self.hw = hw
        self.loaded = False

    # ---- lifecycle -----------------------------------------------------------------

    @property
    def plan(self) -> MemoryPlan:
        return choose_plan(self.plan_table, self.hw, self.settings.offload, self.settings.quantize)

    def check_runnable(self) -> None:
        if self.requires_gpu and self.hw.device == "cpu" and not self.settings.allow_cpu:
            raise UserError(
                f"{self.label} needs a GPU (NVIDIA with CUDA, or Apple Silicon). No GPU was detected. "
                "Set VIDEOGEN_ALLOW_CPU=1 to try anyway (expect many hours per clip)."
            )

    @abstractmethod
    def download(self, progress: Progress | None = None, force: bool = False) -> None: ...

    @abstractmethod
    def is_downloaded(self) -> bool: ...

    @abstractmethod
    def load(self, progress: Progress) -> None: ...

    def unload(self) -> None:
        self.loaded = False
        free_memory()

    @abstractmethod
    def generate(self, req: GenerationRequest, progress: Progress) -> RawVideo: ...

    # ---- description for the UI / API --------------------------------------------

    def info(self) -> dict[str, Any]:
        plan, plan_error = None, None
        if self.requires_gpu:
            try:
                plan = self.plan.describe()
            except ValueError as exc:
                plan_error = str(exc)
        warnings = []
        if self.requires_gpu and self.hw.device == "cpu":
            warnings.append("No GPU detected: this engine needs an NVIDIA (CUDA) or Apple Silicon GPU.")
        elif self.hw.device == "cuda" and not self.hw.bf16:
            warnings.append(
                "This GPU has no native bfloat16 support (pre-RTX 30 series); generation will be very slow "
                "or may fail. An RTX 3090/4090/5090 or newer workstation GPU is recommended."
            )
        if self.hw.device == "cuda" and self.hw.vram_gb < self.min_vram_gb:
            warnings.append(
                f"Your GPU has {self.hw.vram_gb:.0f} GB VRAM; {self.min_vram_gb:.0f} GB+ is recommended. "
                "It may still run with heavy offloading, but slowly."
            )
        if plan_error:
            warnings.append(plan_error)
        try:
            downloaded = self.is_downloaded()
        except Exception:  # never let a cache check break the info endpoint
            downloaded = False
        return {
            "id": self.id,
            "label": self.label,
            "tagline": self.tagline,
            "description": self.description,
            "license": self.license,
            "license_url": self.license_url,
            "fps": self.fps,
            "max_duration": self.max_duration,
            "supports_audio": self.supports_audio,
            "supports_prompt_enhancement": self.supports_prompt_enhancement,
            "tunable": self.tunable,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
            "native_resolution": self.native_resolution,
            "download_gb": self.download_gb,
            "min_vram_gb": self.min_vram_gb,
            "downloaded": downloaded,
            "loaded": self.loaded,
            "memory_plan": plan,
            "warnings": warnings,
        }


# ---- helpers for diffusers engines -------------------------------------------------


def free_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def quantization_config(plan: MemoryPlan, components: list[str]):
    """A diffusers PipelineQuantizationConfig for the plan, or None."""
    if plan.quantize == "none" or not components:
        return None
    import torch
    from diffusers.quantizers import PipelineQuantizationConfig

    if plan.quantize == "int8":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=components,
        )
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=components,
    )


class PipelinePlacer:
    """Applies a memory plan to pipelines that share components.

    accelerate offload hooks live on the modules themselves, so two pipelines that
    share a transformer cannot both have hooks installed. Only one pipeline is
    "active" at a time; switching re-installs the hooks for the new one.
    """

    def __init__(self, plan: MemoryPlan, device: str) -> None:
        self.plan = plan
        self.device = device
        self.active = None

    def activate(self, pipe) -> None:
        if pipe is self.active:
            return
        if self.active is not None and self.plan.offload != "none":
            self.active.remove_all_hooks()
        if self.plan.offload == "none":
            pipe.to(self.device)
        elif self.plan.offload == "model":
            pipe.enable_model_cpu_offload(device=self.device)
        else:
            pipe.enable_sequential_cpu_offload(device=self.device)
        self.active = pipe

    def forget(self, pipe) -> None:
        """Drop a pipeline that is about to be deleted."""
        if pipe is self.active:
            if self.plan.offload != "none":
                pipe.remove_all_hooks()
            self.active = None


def release_hooks(pipe) -> None:
    """After an aborted run, push offloaded models back to the CPU so the next job starts clean."""
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass
    free_memory()


def frames_to_uint8(video: np.ndarray, chunk: int = 16) -> np.ndarray:
    """Convert float frames in [0, 1] ([F,H,W,C]) to uint8 without a full-size float temporary."""
    out = np.empty(video.shape, dtype=np.uint8)
    for i in range(0, len(video), chunk):
        out[i : i + chunk] = np.clip(np.rint(video[i : i + chunk] * 255.0), 0, 255)
    return out


def make_generator(seed: int):
    import torch

    # A CPU generator gives the same noise on every GPU vendor/model for a given seed.
    return torch.Generator("cpu").manual_seed(int(seed) % (2**63))
