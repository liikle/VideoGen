"""Wan 2.2 by Alibaba (Apache-2.0): TI2V-5B and the A14B mixture-of-experts models.

Wan generates ~5 s per pass. Longer clips are built by chaining: each new segment
is an image-to-video continuation that starts from the previous segment's last
frame. Every segment is color-matched to that frame so the joins don't drift.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .. import hub
from ..core import GenerationRequest, Progress, RawVideo
from ..sizing import fit_area, fit_image, frames_for, plan_segments
from .base import (
    Engine,
    PipelinePlacer,
    frames_to_uint8,
    free_memory,
    make_generator,
    quantization_config,
    release_hooks,
)

log = logging.getLogger(__name__)

# Wan's official negative prompt (it was trained with the Chinese wording). In English: garish colors,
# overexposed, static, blurry details, subtitles, style, artwork, painting, still frame, grayish,
# worst/low quality, JPEG artifacts, ugly, mutilated, extra fingers, badly drawn hands/face,
# deformed, disfigured, malformed limbs, fused fingers, motionless, cluttered background,
# three legs, crowded background, walking backwards.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SHARED = ("text_encoder", "tokenizer", "vae")


def match_color(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Correct a continuation segment's color so its first frame matches the frame it continues from.

    Only small per-channel gain/offset corrections are applied; this removes the slow color
    drift that builds up over chained segments without fighting intended lighting changes.
    """
    src = frames[0].reshape(-1, 3).astype(np.float32)
    ref = reference.reshape(-1, 3).astype(np.float32)
    gain = np.clip((ref.std(0) + 1e-3) / (src.std(0) + 1e-3), 0.85, 1.15)
    offset = np.clip(ref.mean(0) - src.mean(0) * gain, -20.0, 20.0)
    out = np.empty_like(frames)
    for i in range(0, len(frames), 16):
        out[i : i + 16] = np.clip(frames[i : i + 16].astype(np.float32) * gain + offset, 0, 255)
    return out


class _WanEngine(Engine):
    segment_frames = 121
    area = 1280 * 704
    multiple = 32
    i2v_guidance = 5.0
    # Low-noise expert guidance for the A14B models (None for single-transformer models).
    t2v_guidance_2: float | None = None
    i2v_guidance_2: float | None = None
    t2v_flow_shift: float | None = None
    i2v_flow_shift: float | None = None
    quantize_components = ("transformer", "transformer_2", "text_encoder")

    def __init__(self, settings, hw) -> None:
        super().__init__(settings, hw)
        self.pipes: dict = {}
        self.placer: PipelinePlacer | None = None

    def size_for(self, aspect: str) -> tuple[int, int]:
        return fit_area(aspect, self.area, self.multiple)

    def _new_placer(self) -> PipelinePlacer:
        return PipelinePlacer(self.plan, self.hw.device)

    def _set_shift(self, pipe, shift: float | None) -> None:
        if shift is None:
            return
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=shift)

    def _load_vae(self, path):
        import torch
        from diffusers import AutoencoderKLWan

        # The Wan VAE is numerically sensitive; keep it in float32 as upstream recommends.
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", dtype=torch.float32)
        vae.enable_tiling()
        return vae

    def _get_pipe(self, kind: str, progress: Progress):
        raise NotImplementedError

    def unload(self) -> None:
        if self.placer is not None:
            for pipe in self.pipes.values():
                self.placer.forget(pipe)
        self.pipes = {}
        self.placer = None
        super().unload()

    def generate(self, req: GenerationRequest, progress: Progress) -> RawVideo:
        width, height = self.size_for(req.aspect)
        total = frames_for(req.duration, self.fps, 4)
        segments = plan_segments(total, self.segment_frames, 4)
        steps = req.steps or self.default_steps
        negative = WAN_NEGATIVE_PROMPT + (f", {req.negative_prompt}" if req.negative_prompt else "")
        cond = fit_image(req.image, width, height) if req.image is not None else None

        pieces: list[np.ndarray] = []
        last_frame = None
        count = len(segments)
        for k, length in enumerate(segments):
            part = progress.sub(k / count, (k + 1) / count)
            label = f"Segment {k + 1}/{count}" if count > 1 else "Generating"
            kind = "t2v" if cond is None else "i2v"
            pipe = self._get_pipe(kind, part.sub(0.0, 0.05))
            self.placer.activate(pipe)
            if kind == "t2v":
                guidance, guidance_2, extra = self.default_guidance, self.t2v_guidance_2, {}
            else:
                guidance, guidance_2, extra = self.i2v_guidance, self.i2v_guidance_2, {"image": cond}
            if req.guidance is not None:
                guidance, guidance_2 = req.guidance, (req.guidance if guidance_2 is not None else None)
            if guidance_2 is not None:
                extra["guidance_scale_2"] = guidance_2
            denoise = part.sub(0.05, 1.0)
            denoise(0.0, f"{label} · starting")
            try:
                out = pipe(
                    prompt=req.prompt,
                    negative_prompt=negative,
                    height=height,
                    width=width,
                    num_frames=length,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=make_generator(req.seed + k),
                    output_type="np",
                    callback_on_step_end=denoise.step_callback(steps, label, after=f"{label} · decoding frames"),
                    **extra,
                ).frames[0]
            except BaseException:
                release_hooks(pipe)
                raise
            # Drop our reference so _get_pipe can free this pipeline if the next segment needs the other one.
            del pipe
            frames = frames_to_uint8(out)
            del out
            if last_frame is not None:
                # The first frame repeats the previous segment's last frame; drop it after color matching.
                frames = match_color(frames, last_frame)[1:]
            pieces.append(frames)
            last_frame = frames[-1]
            cond = Image.fromarray(last_frame)

        video = np.concatenate(pieces)[:total]
        free_memory()
        info = {"generated_size": [width, height], "num_frames": int(len(video)), "segments": count}
        return RawVideo(frames=video, fps=self.fps, info=info)


class Wan5BEngine(_WanEngine):
    id = "wan-5b"
    label = "Wan 2.2 TI2V · 5B"
    tagline = "Fastest · runs on 12–16 GB GPUs · Apache-2.0"
    description = (
        "Alibaba's compact text+image-to-video model at 720p/24 fps. Clips longer than 5 s are built "
        "from chained segments, then upscaled to 1080p."
    )
    license = "Apache-2.0 (free for any use, including commercial)"
    license_url = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    fps = 24
    max_duration = 15.0
    default_steps = 50
    default_guidance = 5.0
    i2v_guidance = 5.0
    native_resolution = "720p native, upscaled"
    download_gb = 34.0
    min_vram_gb = 12.0
    segment_frames = 121
    area = 1280 * 704
    multiple = 32
    plan_table = [(40, "none", "none"), (16, "model", "none"), (10, "model", "int8"), (0, "sequential", "none")]

    @property
    def repo(self) -> str:
        return self.settings.wan5b_model

    def download(self, progress=None, force=False) -> None:
        hub.fetch_pipeline(self.repo, progress=progress, force=force)

    def is_downloaded(self) -> bool:
        return hub.is_pipeline_cached(self.repo)

    def load(self, progress: Progress) -> None:
        if self.loaded:
            return
        self.check_runnable()
        import torch
        from diffusers import WanImageToVideoPipeline, WanPipeline

        path = hub.fetch_pipeline(self.repo, progress=progress.sub(0.0, 0.85))
        plan = self.plan
        progress(0.86, f"Loading Wan 2.2 5B into memory ({plan.describe()})")
        t2v = WanPipeline.from_pretrained(
            path,
            vae=self._load_vae(path),
            dtype=torch.bfloat16,
            quantization_config=quantization_config(plan, ["transformer", "text_encoder"]),
        )
        # TI2V-5B does image-to-video with the very same weights; only the conditioning differs.
        i2v = WanImageToVideoPipeline(
            tokenizer=t2v.tokenizer,
            text_encoder=t2v.text_encoder,
            vae=t2v.vae,
            scheduler=t2v.scheduler,
            transformer=t2v.transformer,
            transformer_2=None,
            boundary_ratio=None,
            expand_timesteps=True,
        )
        self.pipes = {"t2v": t2v, "i2v": i2v}
        self.placer = self._new_placer()
        self.placer.activate(t2v)
        self.loaded = True
        progress(1.0, "Wan 2.2 5B ready")

    def _get_pipe(self, kind: str, progress: Progress):
        return self.pipes[kind]


class Wan14BEngine(_WanEngine):
    id = "wan-14b"
    label = "Wan 2.2 A14B · 27B MoE"
    tagline = "High quality · 24 GB+ GPU and 64 GB+ RAM · Apache-2.0"
    description = (
        "Alibaba's flagship open video model: two 14B experts (layout, then detail) at 720p/16 fps. "
        "Slower and much larger to download, with excellent motion and realism."
    )
    license = "Apache-2.0 (free for any use, including commercial)"
    license_url = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    fps = 16
    max_duration = 15.0
    default_steps = 40
    default_guidance = 4.0
    t2v_guidance_2 = 3.0
    i2v_guidance = 3.5
    i2v_guidance_2 = 3.5
    t2v_flow_shift = 12.0
    i2v_flow_shift = 5.0
    native_resolution = "720p native, upscaled"
    download_gb = 240.0
    min_vram_gb = 24.0
    segment_frames = 81
    area = 1280 * 720
    multiple = 16
    plan_table = [(40, "model", "none"), (24, "model", "int8"), (14, "model", "nf4"), (0, "sequential", "none")]

    def __init__(self, settings, hw) -> None:
        super().__init__(settings, hw)
        self.shared: dict = {}

    def _repo(self, kind: str) -> str:
        return self.settings.wan14b_t2v_model if kind == "t2v" else self.settings.wan14b_i2v_model

    # Text encoder, tokenizer and VAE are identical in both repos; they come from the T2V repo once.
    def _fetch_shared(self, progress=None, force=False):
        return hub.fetch_pipeline(
            self._repo("t2v"), skip={"transformer", "transformer_2"}, progress=progress, force=force
        )

    def _fetch_experts(self, kind: str, progress=None, force=False):
        return hub.fetch_pipeline(self._repo(kind), skip=set(SHARED), progress=progress, force=force)

    def download(self, progress=None, force=False) -> None:
        progress = progress or Progress()
        self._fetch_shared(progress.sub(0.0, 0.05), force)
        self._fetch_experts("t2v", progress.sub(0.05, 0.5), force)
        self._fetch_experts("i2v", progress.sub(0.5, 1.0), force)

    def is_downloaded(self) -> bool:
        return (
            hub.is_pipeline_cached(self._repo("t2v"), skip={"transformer", "transformer_2"})
            and hub.is_pipeline_cached(self._repo("t2v"), skip=set(SHARED))
            and hub.is_pipeline_cached(self._repo("i2v"), skip=set(SHARED))
        )

    def _expert_ram_gb(self) -> float:
        return {"none": 57.0, "int8": 29.0, "nf4": 16.0}[self.plan.quantize]

    def load(self, progress: Progress) -> None:
        if self.loaded:
            return
        self.check_runnable()
        import torch
        from diffusers import WanPipeline

        path = self._fetch_shared(progress.sub(0.0, 0.8))
        plan = self.plan
        progress(0.85, f"Loading Wan 2.2 A14B text encoder ({plan.describe()})")
        base = WanPipeline.from_pretrained(
            path,
            transformer=None,
            transformer_2=None,
            vae=self._load_vae(path),
            dtype=torch.bfloat16,
            quantization_config=quantization_config(plan, ["text_encoder"]),
        )
        self.shared = {name: getattr(base, name) for name in SHARED}
        del base
        self.placer = self._new_placer()
        self.loaded = True
        progress(1.0, "Wan 2.2 A14B ready")

    def _get_pipe(self, kind: str, progress: Progress):
        if kind in self.pipes:
            return self.pipes[kind]
        import torch
        from diffusers import WanImageToVideoPipeline, WanPipeline

        other = "i2v" if kind == "t2v" else "t2v"
        if other in self.pipes and self.hw.ram_gb < 2 * self._expert_ram_gb() + 28:
            # Not enough system RAM to keep both expert pairs resident.
            self.placer.forget(self.pipes.pop(other))
            free_memory()
        path = self._fetch_experts(kind, progress.sub(0.0, 0.6))
        progress(0.65, f"Loading Wan 2.2 A14B {'image' if kind == 'i2v' else 'text'}-to-video experts")
        cls = WanPipeline if kind == "t2v" else WanImageToVideoPipeline
        pipe = cls.from_pretrained(
            path,
            **self.shared,
            dtype=torch.bfloat16,
            quantization_config=quantization_config(self.plan, ["transformer", "transformer_2"]),
        )
        self._set_shift(pipe, self.t2v_flow_shift if kind == "t2v" else self.i2v_flow_shift)
        self.pipes[kind] = pipe
        progress(1.0)
        return pipe
