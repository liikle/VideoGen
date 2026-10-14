"""Turn raw engine output into the delivered MP4.

Steps: optional AI upscale (Real-ESRGAN) → optional motion interpolation →
scale-to-cover + center crop to the exact target size → H.264 encode with AAC
audio → poster thumbnail. Frames are streamed into ffmpeg so memory stays flat.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from . import hub
from .core import GenerationRequest, Progress, RawVideo, UserError
from .sizing import cover_size, crop_to_aspect_size, target_size

log = logging.getLogger(__name__)

ESRGAN_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise UserError("ffmpeg was not found. Install ffmpeg or run `pip install imageio-ffmpeg`.") from None


def output_size(native_w: int, native_h: int, req: GenerationRequest) -> tuple[int, int]:
    target_w, target_h = target_size(req.resolution, req.aspect)
    if req.upscaler != "none":
        return target_w, target_h
    # Keep native pixels: only crop to the requested aspect ratio, never upscale.
    w, h = crop_to_aspect_size(native_w, native_h, target_w / target_h)
    if w > target_w:
        return target_w, target_h
    return w, h


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio [channels, samples] in [-1, 1] as 16-bit PCM."""
    if audio.ndim == 1:
        audio = audio[None]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(pcm.shape[0])
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(np.ascontiguousarray(pcm.T).tobytes())


class ESRGANUpscaler:
    """Real-ESRGAN via spandrel, applied per frame in overlapping tiles to bound VRAM."""

    def __init__(self, scale_needed: float, device: str, progress: Progress | None = None) -> None:
        import spandrel
        import torch

        scale = 2 if scale_needed <= 2.0 else 4
        path = hub.fetch_url(ESRGAN_URLS[scale], "upscalers", progress)
        descriptor = spandrel.ModelLoader().load_from_file(str(path))
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise UserError(f"{path.name} is not an image upscaling model.")
        self.device = torch.device(device)
        self.dtype = torch.float16 if (self.device.type == "cuda" and descriptor.supports_half) else torch.float32
        descriptor.to(self.device).eval()
        if self.dtype == torch.float16:
            descriptor.half()
        self.model = descriptor
        self.scale = descriptor.scale

    def upscale(self, frame: np.ndarray, tile: int = 512, pad: int = 24) -> np.ndarray:
        import torch

        s = self.scale
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].to(self.dtype) / 255.0
        _, _, h, w = x.shape
        out = torch.empty((1, 3, h * s, w * s), device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for y in range(0, h, tile):
                for x0 in range(0, w, tile):
                    ya, xa = max(y - pad, 0), max(x0 - pad, 0)
                    yb, xb = min(y + tile + pad, h), min(x0 + tile + pad, w)
                    result = self.model(x[:, :, ya:yb, xa:xb])
                    th, tw = (min(y + tile, h) - y) * s, (min(x0 + tile, w) - x0) * s
                    oy, ox = (y - ya) * s, (x0 - xa) * s
                    out[:, :, y * s : y * s + th, x0 * s : x0 * s + tw] = result[:, :, oy : oy + th, ox : ox + tw]
        return (out.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()

    def close(self) -> None:
        from .engines.base import free_memory

        self.model = None
        free_memory()


def _frame_source(raw: RawVideo, upscaler: ESRGANUpscaler | None) -> Iterator[np.ndarray]:
    for frame in raw.frames:
        yield upscaler.upscale(frame) if upscaler is not None else frame


def build_filters(in_w: int, in_h: int, out_w: int, out_h: int, fps: float, smooth_fps: int) -> tuple[str, float]:
    filters = []
    out_fps = fps
    if smooth_fps and smooth_fps > fps:
        filters.append(f"minterpolate=fps={smooth_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
        out_fps = float(smooth_fps)
    if (in_w, in_h) != (out_w, out_h):
        cover_w, cover_h = cover_size(in_w, in_h, out_w, out_h)
        if (cover_w, cover_h) != (in_w, in_h):
            filters.append(f"scale={cover_w}:{cover_h}:flags=lanczos")
        if (cover_w, cover_h) != (out_w, out_h):
            filters.append(f"crop={out_w}:{out_h}")
    filters.append("format=yuv420p")
    return ",".join(filters), out_fps


def finalize(
    raw: RawVideo,
    req: GenerationRequest,
    out_path: Path,
    progress: Progress,
    device: str = "cpu",
    poster_path: Path | None = None,
    crf: int = 17,
) -> dict:
    """Encode `raw` to `out_path` (MP4), optionally writing a poster JPEG. Returns output metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, native_h, native_w, _ = raw.frames.shape
    out_w, out_h = output_size(native_w, native_h, req)

    upscaler = None
    scale_needed = max(out_w / native_w, out_h / native_h)
    if req.upscaler == "realesrgan" and scale_needed > 1.05:
        progress(0.0, "Loading Real-ESRGAN upscaler")
        upscaler = ESRGANUpscaler(scale_needed, device, progress.sub(0.0, 0.05))
        in_w, in_h = native_w * upscaler.scale, native_h * upscaler.scale
    else:
        in_w, in_h = native_w, native_h

    vf, out_fps = build_filters(in_w, in_h, out_w, out_h, raw.fps, req.smooth_fps)
    has_audio = raw.audio is not None and raw.sample_rate
    with tempfile.TemporaryDirectory(prefix="videogen-") as tmp:
        cmd = [
            ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{in_w}x{in_h}", "-r", f"{raw.fps}", "-i", "pipe:0",
        ]  # fmt: skip
        if has_audio:
            wav = Path(tmp) / "audio.wav"
            write_wav(wav, raw.audio, raw.sample_rate)
            cmd += ["-i", str(wav)]
        cmd += [
            "-vf", vf, "-r", f"{out_fps}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high",
            "-movflags", "+faststart",
        ]  # fmt: skip
        if has_audio:
            # Pad audio with silence if it is shorter than the video, then stop at the video's end.
            cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path)]

        stderr_path = Path(tmp) / "ffmpeg.log"
        with open(stderr_path, "wb") as stderr:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr)
            encode = progress.sub(0.05 if upscaler else 0.0, 0.97)
            label = "Upscaling (Real-ESRGAN) and encoding" if upscaler else "Encoding video"
            try:
                for i, frame in enumerate(_frame_source(raw, upscaler)):
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                    if i % 8 == 0:
                        encode(i / count, f"{label} · frame {i + 1}/{count}")
                proc.stdin.close()
                encode(1.0, "Finishing encode")
                code = proc.wait()
            except BrokenPipeError:
                # ffmpeg exited early; its log (checked below) says why.
                code = proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                out_path.unlink(missing_ok=True)
                raise
            finally:
                if upscaler is not None:
                    upscaler.close()
        if code != 0:
            message = stderr_path.read_text(errors="replace")[-1500:]
            out_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed (exit {code}): {message}")

    if poster_path is not None:
        frame = Image.fromarray(raw.frames[count // 3])
        frame.thumbnail((640, 640), Image.LANCZOS)
        frame.save(poster_path, quality=85)
    progress(1.0, "Done")
    out_frames = count if out_fps == raw.fps else int(round(count / raw.fps * out_fps))
    return {
        "width": out_w,
        "height": out_h,
        "fps": out_fps,
        "frames": out_frames,
        "duration": round(count / raw.fps, 2),
        "has_audio": bool(has_audio),
        "size_bytes": out_path.stat().st_size,
        "native_size": [native_w, native_h],
        "upscaler": "realesrgan" if upscaler else ("lanczos" if (in_w, in_h) != (out_w, out_h) else "none"),
    }
