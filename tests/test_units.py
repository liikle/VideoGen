import json

import numpy as np
import pytest
from PIL import Image

from videogen import hub
from videogen.core import GenerationRequest, Progress, UserError, closest_aspect
from videogen.engines.wan import match_color
from videogen.hardware import Hardware, choose_plan
from videogen.postprocess import build_filters, output_size, write_wav
from videogen.sizing import (
    ceil_to,
    cover_size,
    crop_to_aspect_size,
    fit_area,
    fit_image,
    frames_for,
    plan_segments,
    snap_frames,
    target_size,
)


def test_snap_frames():
    assert snap_frames(121, 4) == 121
    assert snap_frames(120, 4) == 121
    assert snap_frames(360, 8) == 361
    assert snap_frames(1, 8) == 1


def test_frames_for_targets():
    assert frames_for(15, 24, 8) == 361  # LTX 15 s
    assert frames_for(20, 24, 8) == 481  # LTX max
    assert frames_for(5, 24, 4) == 121  # Wan 5B single pass
    assert frames_for(15, 16, 4) == 241  # Wan A14B 15 s


@pytest.mark.parametrize(
    "total,segment,expected",
    [
        (121, 121, [121]),
        (361, 121, [121, 121, 121]),  # 121 + 120 + 120
        (241, 81, [81, 81, 81]),
        (73, 121, [73]),
        (169, 121, [121, 49]),
    ],
)
def test_plan_segments(total, segment, expected):
    segments = plan_segments(total, segment, 4)
    assert segments == expected
    assert segments[0] + sum(s - 1 for s in segments[1:]) >= total
    assert all((s - 1) % 4 == 0 for s in segments)


def test_sizes():
    assert target_size("1080p", "9:16") == (1080, 1920)
    assert (ceil_to(1920, 64), ceil_to(1080, 64)) == (1920, 1088)
    assert fit_area("16:9", 1280 * 704, 32) == (1280, 704)
    assert fit_area("9:16", 1280 * 704, 32) == (704, 1280)
    assert fit_area("16:9", 1280 * 720, 16) == (1280, 720)
    w, h = fit_area("1:1", 1280 * 704, 32)
    assert w == h and w % 32 == 0
    assert cover_size(1280, 704, 1920, 1080) == (1964, 1080)
    assert cover_size(1920, 1088, 1920, 1080) == (1920, 1088)
    assert crop_to_aspect_size(1920, 1088, 16 / 9) == (1920, 1080)


def test_fit_image_crops_center():
    img = Image.new("RGB", (400, 100), "red")
    img.paste(Image.new("RGB", (100, 100), "blue"), (150, 0))
    out = fit_image(img, 64, 64)
    assert out.size == (64, 64)
    assert out.getpixel((32, 32))[2] > 200  # center (blue) kept


def test_closest_aspect():
    assert closest_aspect(1920, 1080) == "16:9"
    assert closest_aspect(1080, 1920) == "9:16"
    assert closest_aspect(1000, 900) == "1:1"


def test_request_validation():
    req = GenerationRequest(prompt="  hi  ", aspect="auto", image=Image.new("RGB", (90, 160))).normalized()
    assert req.prompt == "hi" and req.aspect == "9:16" and req.mode == "i2v"
    for bad in (
        dict(prompt=""),
        dict(prompt="x", duration=0),
        dict(prompt="x", resolution="4k"),
        dict(prompt="x", upscaler="magic"),
        dict(prompt="x", smooth_fps=25),
    ):
        with pytest.raises(UserError):
            GenerationRequest(**bad).normalized()
    data = GenerationRequest(prompt="x", seed=3).to_dict()
    assert "image" not in data and GenerationRequest.from_dict(data).seed == 3


def test_progress_mapping_and_cancel():
    seen = []
    cancel = {"flag": False}
    p = Progress(lambda f, m: seen.append(round(f, 3)), lambda: cancel["flag"])
    p.sub(0.5, 1.0).sub(0.0, 0.5)(1.0)
    assert seen == [0.75]
    cb = p.sub(0, 1).step_callback(4, "x", after="decoding")
    assert cb(None, 3, 0, {"a": 1}) == {"a": 1}
    cancel["flag"] = True
    with pytest.raises(Exception) as info:
        p(0.1)
    assert type(info.value).__name__ == "Cancelled"


def test_memory_plans():
    table = [(60, "model", "none"), (32, "model", "int8"), (16, "model", "nf4"), (0, "sequential", "none")]
    gpu = lambda vram: Hardware("cuda", "GPU", vram, 64)  # noqa: E731
    assert choose_plan(table, gpu(80)).quantize == "none"
    assert choose_plan(table, gpu(24)).quantize == "nf4"
    assert choose_plan(table, gpu(8)).offload == "sequential"
    # quantized weights cannot be streamed layer by layer
    assert choose_plan(table, gpu(8), quantize="nf4").offload == "model"
    assert choose_plan(table, Hardware("mps", "M", 64, 64)).offload == "none"
    with pytest.raises(ValueError):
        choose_plan(table, Hardware("mps", "M", 64, 64), quantize="int8")


def test_match_color_removes_offset():
    rng = np.random.default_rng(0)
    ref = rng.integers(40, 200, (32, 32, 3), dtype=np.uint8)
    shifted = np.clip(ref.astype(int) + 12, 0, 255).astype(np.uint8)
    seg = np.stack([shifted] * 3)
    fixed = match_color(seg, ref)
    assert abs(fixed[0].astype(int).mean() - ref.mean()) < 1.0
    assert fixed.shape == seg.shape and fixed.dtype == np.uint8


def test_output_size_and_filters():
    req = GenerationRequest(prompt="x", resolution="1080p", aspect="16:9")
    assert output_size(1280, 704, req) == (1920, 1080)
    req.upscaler = "none"
    assert output_size(1920, 1088, req) == (1920, 1080)
    assert output_size(1280, 704, req) == (1252, 704)
    vf, fps = build_filters(1280, 704, 1920, 1080, 24, 0)
    assert vf == "scale=1964:1080:flags=lanczos,crop=1920:1080,format=yuv420p" and fps == 24
    vf, fps = build_filters(1920, 1088, 1920, 1080, 16, 48)
    assert vf.startswith("minterpolate=fps=48") and "crop=1920:1080" in vf and fps == 48


def test_write_wav(tmp_path):
    import wave

    audio = np.stack([np.linspace(-1, 1, 4800), np.zeros(4800)]).astype(np.float32)
    write_wav(tmp_path / "a.wav", audio, 48000)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (2, 48000, 4800)


def test_hub_selects_only_indexed_shards(monkeypatch, tmp_path):
    files = {
        "model_index.json": {"transformer": ["diffusers", "X"], "text_encoder": ["transformers", "Y"],
                             "prompt_enhancer": ["transformers", "Z"], "processor": [None, None]},
        "transformer/diffusion_pytorch_model.safetensors.index.json": {
            "weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                           "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}},
    }  # fmt: skip
    tree = {
        "model_index.json": 1,
        "README.md": 1,
        "transformer/config.json": 1,
        "transformer/diffusion_pytorch_model.safetensors.index.json": 1,
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 10,
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 10,
        # a second, unreferenced shard set (as in LTX-2.5) must be ignored
        "transformer/diffusion_pytorch_model-00001-of-00004.safetensors": 5,
        "text_encoder/config.json": 1,
        "text_encoder/model.safetensors": 20,
        "text_encoder/model.fp16.safetensors": 10,
        "prompt_enhancer/model.safetensors": 30,
        "latent_upsampler/config.json": 1,
        "latent_upsampler/diffusion_pytorch_model.safetensors": 3,
        "transformer_full/diffusion_pytorch_model.safetensors": 40,
    }

    def fake_download(repo_id, filename, token=None):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(files[filename]))
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    selected = hub._select_files("org/repo", tree, {"prompt_enhancer"}, ["latent_upsampler"], None)
    assert selected == sorted([
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "latent_upsampler/config.json",
        "latent_upsampler/diffusion_pytorch_model.safetensors",
    ])  # fmt: skip
    assert hub._is_variant("x/model.fp16.safetensors")
    assert not hub._is_variant("x/diffusion_pytorch_model-00001-of-00002.safetensors")
    assert not hub._is_variant("RealESRGAN_x2plus.pth")
