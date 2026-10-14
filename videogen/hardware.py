"""GPU/RAM detection and automatic memory-plan selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

OFFLOAD_MODES = ("none", "model", "sequential")
QUANT_MODES = ("none", "int8", "nf4")


@dataclass(frozen=True)
class Hardware:
    device: str  # "cuda", "mps" or "cpu"
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    # Native bfloat16 support (NVIDIA Ampere / RTX 30xx and newer, Apple Silicon).
    bf16: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MemoryPlan:
    """How a pipeline is placed in memory.

    offload:  none       everything lives on the GPU (fastest, most VRAM)
              model      whole sub-models are moved to the GPU only while they run
              sequential individual layers are streamed to the GPU (slowest, least VRAM)
    quantize: none / int8 / nf4 weight quantization of the big components (bitsandbytes)
    """

    offload: str
    quantize: str

    def describe(self) -> str:
        offload = {
            "none": "all weights on GPU",
            "model": "model-level CPU offload",
            "sequential": "layer-level CPU offload (slow, minimal VRAM)",
        }[self.offload]
        quant = {"none": "bf16 weights", "int8": "8-bit weights", "nf4": "4-bit weights"}[self.quantize]
        return f"{quant}, {offload}"


def _ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:  # pragma: no cover - psutil is a hard dependency, this is belt and braces
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError, AttributeError):
            return 0.0


@lru_cache(maxsize=1)
def detect() -> Hardware:
    ram = _ram_gb()
    try:
        import torch
    except ImportError:
        return Hardware("cpu", None, 0.0, ram)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return Hardware("cuda", props.name, props.total_memory / 1024**3, ram, bf16=props.major >= 8)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon shares RAM between CPU and GPU.
        return Hardware("mps", "Apple Silicon (MPS)", ram, ram)
    return Hardware("cpu", None, 0.0, ram)


# Each table row is (minimum VRAM in GB, offload, quantize); first match wins.
PlanTable = list[tuple[float, str, str]]


def choose_plan(table: PlanTable, hw: Hardware, offload: str = "auto", quantize: str = "auto") -> MemoryPlan:
    if hw.device == "mps":
        # bitsandbytes and accelerate offload hooks target CUDA; unified memory makes them unnecessary.
        auto = MemoryPlan("none", "none")
    elif hw.device == "cpu":
        auto = MemoryPlan("none", "none")
    else:
        auto = next(
            (MemoryPlan(o, q) for min_vram, o, q in table if hw.vram_gb >= min_vram),
            MemoryPlan(table[-1][1], table[-1][2]),
        )
    offload = auto.offload if offload in (None, "", "auto") else offload
    quantize = auto.quantize if quantize in (None, "", "auto") else quantize
    if offload not in OFFLOAD_MODES:
        raise ValueError(f"offload must be one of {OFFLOAD_MODES} or auto, got {offload!r}")
    if quantize not in QUANT_MODES:
        raise ValueError(f"quantize must be one of {QUANT_MODES} or auto, got {quantize!r}")
    if hw.device != "cuda":
        if quantize != "none":
            raise ValueError("Weight quantization (int8/nf4) requires an NVIDIA GPU.")
        if offload != "none":
            raise ValueError("CPU offloading requires an NVIDIA GPU.")
    if quantize != "none" and offload == "sequential":
        # accelerate's layer streaming cannot move bitsandbytes-quantized weights.
        offload = "model"
    return MemoryPlan(offload, quantize)
