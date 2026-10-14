"""Runtime settings, read from environment variables (all optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"VIDEOGEN_{name}")
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    output_dir: Path = Path("outputs")
    host: str = "127.0.0.1"
    port: int = 7860
    default_engine: str = "ltx"
    ltx_model: str = "Lightricks/LTX-2.5-Diffusers"
    wan5b_model: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    wan14b_t2v_model: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    wan14b_i2v_model: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    # "auto" picks per engine from detected VRAM; see hardware.choose_plan.
    offload: str = "auto"
    quantize: str = "auto"
    # Download the LTX prompt enhancer (~10 GB). Strongly recommended by Lightricks.
    ltx_prompt_enhancer: bool = True
    # A fake CPU engine used for tests and UI development.
    enable_mock: bool = False
    mock_step_delay: float = 0.15
    # Real models on CPU would take many hours per clip; refuse unless explicitly allowed.
    allow_cpu: bool = False
    # Optional HTTP basic-auth password (username is ignored) for exposing the UI on a network.
    password: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            output_dir=Path(_env("OUTPUT_DIR", "outputs")).expanduser(),
            host=_env("HOST", "127.0.0.1"),
            port=int(_env("PORT", "7860")),
            default_engine=_env("DEFAULT_ENGINE", "ltx"),
            ltx_model=_env("LTX_MODEL", cls.ltx_model),
            wan5b_model=_env("WAN5B_MODEL", cls.wan5b_model),
            wan14b_t2v_model=_env("WAN14B_T2V_MODEL", cls.wan14b_t2v_model),
            wan14b_i2v_model=_env("WAN14B_I2V_MODEL", cls.wan14b_i2v_model),
            offload=_env("OFFLOAD", "auto"),
            quantize=_env("QUANTIZE", "auto"),
            ltx_prompt_enhancer=_env_bool("LTX_PROMPT_ENHANCER", True),
            enable_mock=_env_bool("ENABLE_MOCK", False),
            mock_step_delay=float(_env("MOCK_STEP_DELAY", "0.15")),
            allow_cpu=_env_bool("ALLOW_CPU", False),
            password=_env("PASSWORD"),
        )
