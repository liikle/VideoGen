"""LTX-2.5 by Lightricks: native 1080p video with synchronized audio, up to 20 s per clip.

Uses the distilled two-stage recipe from the diffusers docs:
  stage 1  8 distilled sigmas at half resolution
  upscale  x2 in latent space with the shipped latent upsampler
  stage 2  3 distilled sigmas at full resolution, re-noised from the stage 1 result
"""

from __future__ import annotations

import inspect
import json
import logging

import numpy as np

from .. import hub
from ..core import GenerationRequest, Progress, RawVideo
from ..sizing import ceil_to, fit_image, frames_for, target_size
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

ENHANCER_COMPONENTS = ("prompt_enhancer", "processor")


def _init_kwargs(cls, components: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in components.items() if k in params}


class LTXEngine(Engine):
    id = "ltx"
    label = "LTX-2.5 · 22B"
    tagline = "Best quality · native 1080p · synced audio · up to 20 s"
    description = (
        "Lightricks' open-weight audio+video model. Generates a draft at half resolution, upsamples it "
        "in latent space and refines it at full 1080p, with matching sound effects, ambience and speech."
    )
    license = "LTX-2 Community License: free for individuals and companies under $10M annual revenue"
    license_url = "https://huggingface.co/Lightricks/LTX-2.5-Diffusers"
    fps = 24
    max_duration = 20.0
    supports_audio = True
    supports_prompt_enhancement = True
    tunable = False
    native_resolution = "1080p native"
    download_gb = 81.0
    min_vram_gb = 24.0
    plan_table = [(60, "model", "none"), (32, "model", "int8"), (16, "model", "nf4"), (0, "sequential", "none")]

    def __init__(self, settings, hw) -> None:
        super().__init__(settings, hw)
        self.t2v = None
        self.i2v = None
        self.upsampler = None
        self.placer: PipelinePlacer | None = None

    @property
    def repo(self) -> str:
        return self.settings.ltx_model

    def _skip(self) -> set[str]:
        return set() if self.settings.ltx_prompt_enhancer else set(ENHANCER_COMPONENTS)

    def download(self, progress=None, force=False) -> None:
        hub.fetch_pipeline(self.repo, skip=self._skip(), extra=["latent_upsampler"], progress=progress, force=force)

    def is_downloaded(self) -> bool:
        return hub.is_pipeline_cached(self.repo, skip=self._skip(), extra=["latent_upsampler"])

    def load(self, progress: Progress) -> None:
        if self.loaded:
            return
        self.check_runnable()
        import torch
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        path = hub.fetch_pipeline(
            self.repo, skip=self._skip(), extra=["latent_upsampler"], progress=progress.sub(0.0, 0.85)
        )
        plan = self.plan
        progress(0.86, f"Loading LTX-2.5 into memory ({plan.describe()})")
        model_index = json.loads((path / "model_index.json").read_text())
        skipped = {name: None for name in self._skip() if name in model_index}
        for name in (*ENHANCER_COMPONENTS, "duration_head"):
            # Optional components listed in model_index but not shipped in the repo load as None.
            if name in model_index and not (path / name).is_dir():
                skipped[name] = None
        quantize = [c for c in ("transformer", "text_encoder", "prompt_enhancer") if c in model_index and c not in skipped]

        t2v = LTX2Pipeline.from_pretrained(
            path, dtype=torch.bfloat16, quantization_config=quantization_config(plan, quantize), **skipped
        )
        t2v.vae.enable_tiling()
        self.t2v = t2v
        self.i2v = LTX2ImageToVideoPipeline(**_init_kwargs(LTX2ImageToVideoPipeline, t2v.components))
        self.upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            path, subfolder="latent_upsampler", dtype=torch.bfloat16
        ).eval()
        self.placer = PipelinePlacer(plan, self.hw.device)
        if plan.offload == "none":
            self.upsampler.to(self.placer.device)
        self.placer.activate(t2v)
        self.loaded = True
        progress(1.0, "LTX-2.5 ready")

    def unload(self) -> None:
        if self.placer is not None:
            for pipe in (self.t2v, self.i2v):
                if pipe is not None:
                    self.placer.forget(pipe)
        self.t2v = self.i2v = self.upsampler = self.placer = None
        super().unload()

    def _upsample(self, latents):
        import torch

        device = self.placer.device
        upsampler = self.upsampler
        offload = self.placer.plan.offload != "none"
        if offload:
            upsampler.to(device)
        try:
            # no_grad rather than inference_mode: the result feeds back into the pipeline, which may
            # update it in place, and that is not allowed on inference tensors.
            with torch.no_grad():
                return upsampler(latents.to(device=device, dtype=upsampler.dtype))
        finally:
            if offload:
                upsampler.to("cpu")

    def generate(self, req: GenerationRequest, progress: Progress) -> RawVideo:
        from diffusers.pipelines.ltx2.utils import (
            DISTILLED_SIGMA_VALUES,
            LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
            LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )

        target_w, target_h = target_size(req.resolution, req.aspect)
        # Two-stage generation needs sides divisible by 64; the few extra rows are cropped afterwards.
        width, height = ceil_to(target_w, 64), ceil_to(target_h, 64)
        num_frames = frames_for(req.duration, self.fps, 8)
        image = fit_image(req.image, width, height) if req.image is not None else None
        pipe = self.i2v if image is not None else self.t2v
        self.placer.activate(pipe)
        info: dict = {"generated_size": [width, height], "num_frames": num_frames}

        try:
            prompt = req.prompt
            if req.enhance_prompt and getattr(pipe, "prompt_enhancer", None) is not None:
                progress(0.0, "Enhancing prompt")
                try:
                    prompt = pipe.enhance_prompt(
                        prompt=req.prompt,
                        system_prompt=LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT if image is not None else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
                        seed=req.seed % 2**31,
                        image=image,
                    )
                    info["enhanced_prompt"] = prompt
                except Exception as exc:  # enhancement is a nicety; never fail the job over it
                    log.warning("Prompt enhancement failed, using the original prompt: %s", exc)
                    prompt = req.prompt
                progress.check_cancel()

            generator = make_generator(req.seed)
            common = dict(
                prompt=prompt,
                num_frames=num_frames,
                frame_rate=float(self.fps),
                # Distilled checkpoint: unguided, no STG / modality guidance.
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                generator=generator,
                return_dict=False,
            )
            if image is not None:
                common["image"] = image

            stage1 = progress.sub(0.05, 0.45)
            stage1(0.0, "Stage 1/2 · draft at half resolution")
            video_latent, audio_latent = pipe(
                **common,
                width=width // 2,
                height=height // 2,
                sigmas=DISTILLED_SIGMA_VALUES,
                output_type="latent",
                callback_on_step_end=stage1.step_callback(len(DISTILLED_SIGMA_VALUES), "Stage 1/2 · draft"),
            )

            progress(0.47, "Upscaling latents ×2")
            video_latent = self._upsample(video_latent)

            stage2 = progress.sub(0.5, 1.0)
            stage2(0.0, f"Stage 2/2 · refining at {width}×{height}")
            video, audio = pipe(
                **common,
                width=width,
                height=height,
                latents=video_latent,
                audio_latents=audio_latent,
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                output_type="np",
                callback_on_step_end=stage2.step_callback(
                    len(STAGE_2_DISTILLED_SIGMA_VALUES), "Stage 2/2 · refine", after="Decoding video and audio"
                ),
            )
            del video_latent, audio_latent
        except BaseException:
            release_hooks(pipe)
            raise

        frames = frames_to_uint8(video[0])
        del video
        audio_np, sample_rate = None, None
        if req.audio and audio is not None:
            audio_np = audio[0].float().cpu().numpy().astype(np.float32)
            if audio_np.ndim == 1:
                audio_np = audio_np[None]
            sample_rate = int(pipe.vocoder.config.output_sampling_rate)
        free_memory()
        return RawVideo(frames=frames, fps=self.fps, audio=audio_np, sample_rate=sample_rate, info=info)

