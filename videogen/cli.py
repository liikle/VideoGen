"""Command line: `python -m videogen serve | generate | download | info`."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings
from .core import ASPECTS, RESOLUTIONS, UPSCALERS, Cancelled, GenerationRequest, Progress, UserError


class ConsoleProgress:
    def __init__(self) -> None:
        self.last = 0.0
        self.tty = sys.stderr.isatty()

    def __call__(self, fraction: float, message: str | None) -> None:
        now = time.time()
        if self.tty:
            width = shutil.get_terminal_size((100, 20)).columns
            line = f"[{fraction * 100:5.1f}%] {message or ''}"
            sys.stderr.write("\r" + line[: width - 1].ljust(width - 1))
            sys.stderr.flush()
        elif now - self.last > 5 or fraction >= 1.0:
            print(f"[{fraction * 100:5.1f}%] {message or ''}", file=sys.stderr)
        self.last = now

    def done(self) -> None:
        if self.tty:
            sys.stderr.write("\n")


def _settings(args) -> Settings:
    settings = Settings.from_env()
    for name in ("offload", "quantize", "host", "port"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(settings, name, value)
    if getattr(args, "output_dir", None):
        settings.output_dir = Path(args.output_dir)
    if getattr(args, "mock", False):
        settings.enable_mock = True
    return settings


def cmd_serve(args) -> int:
    from .server import serve

    serve(_settings(args), open_browser=args.open)
    return 0


def cmd_info(args) -> int:
    from .engines import EngineManager

    manager = EngineManager(_settings(args))
    hw = manager.hw
    print(f"VideoGen {__version__}")
    if hw.device == "cuda":
        print(f"GPU:  {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM)")
    elif hw.device == "mps":
        print("GPU:  Apple Silicon (MPS)")
    else:
        print("GPU:  none detected (the real engines need an NVIDIA or Apple Silicon GPU)")
    print(f"RAM:  {hw.ram_gb:.1f} GB\n")
    for info in manager.infos():
        print(f"{info['id']:<8} {info['label']}")
        print(f"         {info['tagline']}")
        state = "downloaded" if info["downloaded"] else f"not downloaded (~{info['download_gb']:.0f} GB)"
        print(f"         {state}; memory plan: {info['memory_plan'] or 'n/a'}")
        for warning in info["warnings"]:
            print(f"         ! {warning}")
        print()
    return 0


def cmd_download(args) -> int:
    from . import hub
    from .engines import EngineManager
    from .postprocess import ESRGAN_URLS

    manager = EngineManager(_settings(args))
    targets = args.engines or [manager.default_engine]
    if "all" in targets:
        targets = [e for e in manager.engines if e != "mock"] + ["realesrgan"]
    for target in targets:
        progress = ConsoleProgress()
        if target == "realesrgan":
            print("Downloading Real-ESRGAN upscalers")
            for url in ESRGAN_URLS.values():
                hub.fetch_url(url, "upscalers", progress)
        else:
            engine = manager.get(target)
            print(f"Downloading {engine.label} (~{engine.download_gb:.0f} GB)")
            engine.download(Progress(progress), force=args.force)
        progress.done()
    print("All downloads complete.")
    return 0


def cmd_generate(args) -> int:
    from PIL import Image

    from .engines import EngineManager
    from .postprocess import finalize

    settings = _settings(args)
    manager = EngineManager(settings)
    image = None
    if args.image:
        with Image.open(args.image) as img:
            image = img.convert("RGB")
    req = GenerationRequest(
        prompt=args.prompt,
        engine=args.engine or manager.default_engine,
        negative_prompt=args.negative or "",
        duration=args.duration,
        resolution=args.resolution,
        aspect=args.aspect,
        seed=args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
        steps=args.steps,
        guidance=args.guidance,
        audio=not args.no_audio,
        enhance_prompt=not args.no_enhance,
        upscaler=args.upscaler,
        smooth_fps=args.smooth_fps,
        image=image,
    ).normalized()
    engine = manager.get(req.engine)
    if req.duration > engine.max_duration:
        raise UserError(f"{engine.label} supports clips up to {engine.max_duration:g} seconds.")
    out = Path(args.output or f"videogen-{time.strftime('%Y%m%d-%H%M%S')}.mp4")
    console = ConsoleProgress()
    progress = Progress(console)
    started = time.time()
    try:
        engine = manager.acquire(req.engine, progress.sub(0.0, 0.1))
        raw = engine.generate(req, progress.sub(0.1, 0.92))
        info = finalize(raw, req, out, progress.sub(0.92, 1.0), device=manager.hw.device)
    finally:
        console.done()
    print(f"Saved {out} ({info['width']}x{info['height']}, {info['duration']} s, seed {req.seed}) "
          f"in {time.time() - started:.0f} s")
    if raw.info.get("enhanced_prompt"):
        print(f"Enhanced prompt: {raw.info['enhanced_prompt']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videogen", description="Free, local, unlimited AI video generation.")
    parser.add_argument("--version", action="version", version=f"videogen {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--offload", choices=["auto", "none", "model", "sequential"], help="GPU memory strategy")
    common.add_argument("--quantize", choices=["auto", "none", "int8", "nf4"], help="weight quantization")
    common.add_argument("--mock", action="store_true", help="enable the fake test engine")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("serve", parents=[common], help="start the web UI (default)")
    p.add_argument("--host", help="interface to bind (default 127.0.0.1; use 0.0.0.0 for LAN access)")
    p.add_argument("--port", type=int, help="port (default 7860)")
    p.add_argument("--output-dir", help="where videos are stored (default ./outputs)")
    p.add_argument("--open", action="store_true", help="open the browser")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("generate", parents=[common], help="generate one video from the command line")
    p.add_argument("prompt")
    p.add_argument("-i", "--image", help="reference image for image-to-video")
    p.add_argument("-e", "--engine", help="ltx, wan-5b or wan-14b")
    p.add_argument("-d", "--duration", type=float, default=5.0, help="seconds (default 5)")
    p.add_argument("-r", "--resolution", choices=RESOLUTIONS, default="1080p")
    p.add_argument("-a", "--aspect", choices=[*ASPECTS, "auto"], default="16:9")
    p.add_argument("-s", "--seed", type=int)
    p.add_argument("-n", "--negative", help="negative prompt (Wan engines)")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-enhance", action="store_true", help="skip LTX prompt enhancement")
    p.add_argument("--upscaler", choices=UPSCALERS, default="lanczos")
    p.add_argument("--smooth-fps", type=int, default=0, choices=[0, 24, 30, 48, 60])
    p.add_argument("-o", "--output", help="output .mp4 path")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("download", parents=[common], help="pre-download model weights")
    p.add_argument("engines", nargs="*", help="ltx, wan-5b, wan-14b, realesrgan or all (default: default engine)")
    p.add_argument("--force", action="store_true", help="re-check files against the Hub")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("info", parents=[common], help="show hardware and engine status")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve", *(argv or [])])
    try:
        return args.func(args)
    except UserError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except (Cancelled, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 130
