"""Run the real engine code against real diffusers pipelines built with tiny random weights.

This exercises everything except the actual model weights: loading from a repo-shaped
folder, the memory placer, shared components between text- and image-to-video
pipelines, LTX two-stage generation with latent upsampling and audio, Wan segment
chaining, progress callbacks and cancellation. It runs on CPU in about a minute.
"""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from videogen import hub  # noqa: E402
from videogen.core import Cancelled, GenerationRequest, Progress  # noqa: E402
from videogen.engines import ltx as ltx_module  # noqa: E402
from videogen.engines.ltx import LTXEngine  # noqa: E402
from videogen.engines.wan import Wan5BEngine, Wan14BEngine  # noqa: E402
from videogen.hardware import Hardware  # noqa: E402

pytestmark = pytest.mark.pipelines

CPU = Hardware("cpu", None, 0.0, 16.0)


def _tiny_t5():
    from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
    return T5EncoderModel(config).eval(), AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")


def _wan_transformer(in_channels, out_channels):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=in_channels,
        out_channels=out_channels, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
    )  # fmt: skip


@pytest.fixture(scope="module")
def wan5b_dir(tmp_path_factory):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=3, z_dim=48, in_channels=12, out_channels=12, is_residual=True, patch_size=2,
        latents_mean=[0.0] * 48, latents_std=[1.0] * 48, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        scale_factor_spatial=16, scale_factor_temporal=4, temperal_downsample=[False, True, True],
    )  # fmt: skip
    text_encoder, tokenizer = _tiny_t5()
    pipe = WanPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0),
        transformer=_wan_transformer(48, 48), transformer_2=None, boundary_ratio=None, expand_timesteps=True,
    )  # fmt: skip
    path = tmp_path_factory.mktemp("wan5b")
    pipe.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def wan14b_dirs(tmp_path_factory):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline

    def vae():
        torch.manual_seed(0)
        return AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                temperal_downsample=[False, True, True])  # fmt: skip

    def scheduler():
        return UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)

    text_encoder, tokenizer = _tiny_t5()
    t2v = WanPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae(), scheduler=scheduler(),
        transformer=_wan_transformer(16, 16), transformer_2=_wan_transformer(16, 16), boundary_ratio=0.875,
    )  # fmt: skip
    i2v = WanImageToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae(), scheduler=scheduler(),
        transformer=_wan_transformer(36, 16), transformer_2=_wan_transformer(36, 16), boundary_ratio=0.9,
    )  # fmt: skip
    paths = {"t2v": tmp_path_factory.mktemp("wan14b_t2v"), "i2v": tmp_path_factory.mktemp("wan14b_i2v")}
    t2v.save_pretrained(paths["t2v"])
    i2v.save_pretrained(paths["i2v"])
    return paths


@pytest.fixture(scope="module")
def ltx_dir(tmp_path_factory):
    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2Pipeline,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    # Same tiny components diffusers uses in its own LTX-2 tests.
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-gemma3")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-gemma3")
    hidden = text_encoder.config.text_config.hidden_size
    torch.manual_seed(0)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=4, out_channels=4, patch_size=1, patch_size_t=1, num_attention_heads=2, attention_head_dim=8,
        cross_attention_dim=16, audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=4, audio_cross_attention_dim=8, num_layers=2, qk_norm="rms_norm_across_heads",
        caption_channels=hidden, rope_double_precision=False, rope_type="split",
    )  # fmt: skip
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(
        caption_channels=hidden, text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=4, video_connector_attention_head_dim=8, video_connector_num_layers=1,
        video_connector_num_learnable_registers=None, audio_connector_num_attention_heads=4,
        audio_connector_attention_head_dim=8, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None, connector_rope_base_seq_len=32, rope_theta=10000.0,
        rope_double_precision=False, causal_temporal_positioning=False, rope_type="split",
    )  # fmt: skip
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8,), decoder_block_out_channels=(8,),
        layers_per_block=(1,), decoder_layers_per_block=(1, 1), spatio_temporal_scaling=(True,),
        decoder_spatio_temporal_scaling=(True,), decoder_inject_noise=(False, False), downsample_type=("spatial",),
        upsample_residual=(False,), upsample_factor=(1,), timestep_conditioning=False, patch_size=1,
        patch_size_t=1, encoder_causal=True, decoder_causal=False,
    )  # fmt: skip
    torch.manual_seed(0)
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=4, output_channels=2, ch_mult=(1,), num_res_blocks=1, attn_resolutions=None, in_channels=2,
        resolution=32, latent_channels=2, norm_type="pixel", causality_axis="height", dropout=0.0,
        mid_block_add_attention=False, sample_rate=16000, mel_hop_length=160, is_causal=True, mel_bins=8,
    )  # fmt: skip
    torch.manual_seed(0)
    vocoder = LTX2Vocoder(
        in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins, hidden_channels=32,
        out_channels=2, upsample_kernel_sizes=[4, 4], upsample_factors=[2, 2], resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]], leaky_relu_negative_slope=0.1, output_sampling_rate=16000,
    )  # fmt: skip
    pipe = LTX2Pipeline(
        transformer=transformer, vae=vae, audio_vae=audio_vae, scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=text_encoder, tokenizer=tokenizer, connectors=connectors, vocoder=vocoder,
        processor=None, prompt_enhancer=None, duration_head=None,
    )  # fmt: skip
    path = tmp_path_factory.mktemp("ltx")
    pipe.save_pretrained(path)
    torch.manual_seed(0)
    LTX2LatentUpsamplerModel(in_channels=4, mid_channels=32, num_blocks_per_stage=1).save_pretrained(
        path / "latent_upsampler"
    )
    return path


def _engine(cls, settings, **overrides):
    settings.allow_cpu = True
    engine = cls(settings, CPU)
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def _recorder():
    events = []
    return events, Progress(lambda f, m: events.append((f, m)))


def _image(w=96, h=64):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_wan5b_text_and_image_to_video_with_chaining(settings, wan5b_dir, monkeypatch):
    monkeypatch.setattr(hub, "fetch_pipeline", lambda *a, **k: wan5b_dir)
    # Shrink the native size and segment length so CPU inference takes seconds.
    engine = _engine(Wan5BEngine, settings, area=64 * 32, segment_frames=9, default_steps=2)
    events, progress = _recorder()
    engine.load(progress)
    assert engine.loaded and engine.pipes["i2v"].transformer is engine.pipes["t2v"].transformer

    req = GenerationRequest(prompt="a cat", engine="wan-5b", duration=1, seed=1).normalized()
    raw = engine.generate(req, progress)
    # 1 s at 24 fps -> 25 frames from three chained 9-frame segments (9 + 8 + 8).
    assert raw.frames.shape == (25, 32, 64, 3) and raw.frames.dtype == np.uint8
    assert raw.info["segments"] == 3 and raw.fps == 24
    assert any(m and "Segment 3/3" in m for _, m in events)

    req = GenerationRequest(prompt="a cat", engine="wan-5b", duration=1, seed=1, image=_image()).normalized()
    raw = engine.generate(req, progress)
    assert raw.frames.shape == (25, 32, 64, 3)

    engine.unload()
    assert not engine.loaded and not engine.pipes


def test_wan_cancel_mid_generation(settings, wan5b_dir, monkeypatch):
    monkeypatch.setattr(hub, "fetch_pipeline", lambda *a, **k: wan5b_dir)
    engine = _engine(Wan5BEngine, settings, area=64 * 32, segment_frames=9, default_steps=3)
    engine.load(Progress())
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 3

    req = GenerationRequest(prompt="a cat", engine="wan-5b", duration=1, seed=1).normalized()
    with pytest.raises(Cancelled):
        engine.generate(req, Progress(should_cancel=should_cancel))
    # the engine stays usable after a cancel
    raw = engine.generate(req, Progress())
    assert len(raw.frames) == 25


def test_wan14b_moe_with_shared_components(settings, wan14b_dirs, monkeypatch):
    def fake_fetch(repo_id, skip=(), extra=(), progress=None, force=False):
        return wan14b_dirs["i2v" if "I2V" in repo_id else "t2v"]

    monkeypatch.setattr(hub, "fetch_pipeline", fake_fetch)
    engine = _engine(Wan14BEngine, settings, area=32 * 32, segment_frames=9, default_steps=2)
    engine.load(Progress())
    req = GenerationRequest(prompt="a dog", engine="wan-14b", duration=1, seed=3).normalized()
    raw = engine.generate(req, Progress())
    # 1 s at 16 fps -> 17 frames: a text-to-video segment then an image-to-video continuation.
    assert raw.frames.shape[0] == 17 and raw.fps == 16
    # With 16 GB RAM only one expert pair stays resident.
    assert list(engine.pipes) == ["i2v"]
    i2v = engine.pipes["i2v"]
    assert i2v.text_encoder is engine.shared["text_encoder"] and i2v.vae is engine.shared["vae"]
    assert i2v.scheduler.config.flow_shift == 5.0


@pytest.mark.parametrize("with_image", [False, True])
def test_ltx_two_stage_with_audio(settings, ltx_dir, monkeypatch, with_image):
    monkeypatch.setattr(hub, "fetch_pipeline", lambda *a, **k: ltx_dir)
    # Tiny target so the two stages run quickly: 128x64 final, 64x32 draft.
    monkeypatch.setattr(ltx_module, "target_size", lambda resolution, aspect: (128, 64))
    engine = _engine(LTXEngine, settings)
    events, progress = _recorder()
    engine.load(progress)
    req = GenerationRequest(
        prompt="waves on a beach", engine="ltx", duration=1, seed=5, image=_image() if with_image else None
    ).normalized()
    raw = engine.generate(req, progress)
    assert raw.frames.shape == (25, 64, 128, 3)
    assert raw.audio is not None and raw.audio.ndim == 2 and raw.sample_rate == 16000
    assert raw.info["generated_size"] == [128, 64]
    messages = [m for _, m in events if m]
    assert any("Stage 1/2" in m for m in messages) and any("Stage 2/2" in m for m in messages)
    assert any("Upscaling latents" in m for m in messages)
    assert engine.placer.active is (engine.i2v if with_image else engine.t2v)


def test_wan14b_frees_text_experts_before_loading_image_experts(settings, wan14b_dirs, monkeypatch):
    """With little RAM, the text-to-video experts must be garbage-collectable before the image ones load."""
    import gc
    import weakref

    def fake_fetch(repo_id, skip=(), extra=(), progress=None, force=False):
        return wan14b_dirs["i2v" if "I2V" in repo_id else "t2v"]

    monkeypatch.setattr(hub, "fetch_pipeline", fake_fetch)
    engine = _engine(Wan14BEngine, settings, area=32 * 32, segment_frames=9, default_steps=1)
    engine.load(Progress())
    original = engine._get_pipe
    alive_when_loading_i2v = []

    def spy(kind, progress):
        if kind == "t2v" and "t2v" not in engine.pipes:
            pipe = original(kind, progress)
            spy.t2v_ref = weakref.ref(pipe.transformer)
            return pipe
        if kind == "i2v" and "i2v" not in engine.pipes:
            engine.placer.forget(engine.pipes.pop("t2v"))
            gc.collect()
            alive_when_loading_i2v.append(spy.t2v_ref() is not None)
        return original(kind, progress)

    monkeypatch.setattr(engine, "_get_pipe", spy)
    engine.generate(GenerationRequest(prompt="a dog", engine="wan-14b", duration=1, seed=3).normalized(), Progress())
    assert alive_when_loading_i2v == [False]
