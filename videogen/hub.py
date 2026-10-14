"""Model downloads from the Hugging Face Hub.

Diffusers repos often carry more than a pipeline needs: duplicate shard sets,
alternative transformers, prompt enhancers, sample assets. `fetch_pipeline`
works out the exact files the pipeline will load (following each component's
safetensors index) and downloads only those, with byte-level progress.
A small manifest is stored after a successful download so later runs work
fully offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from .core import UserError

log = logging.getLogger(__name__)

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack", ".h5", ".onnx")
# Default (non-variant) weight index names used by diffusers and transformers.
DEFAULT_INDEXES = ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json")

ProgressCb = Callable[[float, str], None]


def cache_root() -> Path:
    root = os.environ.get("VIDEOGEN_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "videogen")
    return Path(root)


def _token() -> str | None:
    # huggingface_hub also reads HF_TOKEN and the `hf auth login` token on its own; this just
    # makes an explicit env var win.
    return os.environ.get("HF_TOKEN") or None


def _gated_message(repo_id: str) -> str:
    return (
        f"The model {repo_id} requires accepting its license on Hugging Face (free, instant). "
        f"1) Open https://huggingface.co/{repo_id} and click 'Agree and access repository'. "
        "2) Create a read token at https://huggingface.co/settings/tokens. "
        "3) Run `hf auth login` (or set HF_TOKEN) and restart VideoGen."
    )


def _is_variant(filename: str) -> bool:
    """True for variant weights like `model.fp16.safetensors` or `x.fp16-00001-of-00002.safetensors`."""
    name = filename.rsplit("/", 1)[-1]
    for ext in WEIGHT_EXTS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return "." in name


def _manifest_path(repo_id: str, skip: Iterable[str], extra: Iterable[str]) -> Path:
    key = json.dumps({"skip": sorted(skip), "extra": sorted(extra)})
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return cache_root() / "manifests" / f"{repo_id.replace('/', '--')}-{digest}.json"


def _select_files(
    repo_id: str, tree: dict[str, int], skip: set[str], extra: list[str], token: str | None
) -> list[str]:
    from huggingface_hub import hf_hub_download

    if "model_index.json" not in tree:
        raise UserError(f"{repo_id} is not a diffusers pipeline repository (no model_index.json).")
    with open(hf_hub_download(repo_id, "model_index.json", token=token)) as f:
        model_index = json.load(f)
    components = [
        name
        for name, value in model_index.items()
        if isinstance(value, list) and len(value) == 2 and value[0] is not None and value[1] is not None
    ]
    folders = [c for c in components if c not in skip] + [e for e in extra if e not in components]

    selected = {"model_index.json"}
    for folder in folders:
        in_folder = [p for p in tree if p.startswith(folder + "/")]
        if not in_folder:
            if folder in extra:
                raise UserError(f"{repo_id} has no '{folder}' folder; is the model id correct?")
            continue
        selected.update(p for p in in_folder if not p.endswith(WEIGHT_EXTS))
        indexes = [p for p in in_folder if p.rsplit("/", 1)[-1] in DEFAULT_INDEXES]
        if indexes:
            for index in indexes:
                with open(hf_hub_download(repo_id, index, token=token)) as f:
                    shards = set(json.load(f)["weight_map"].values())
                selected.update(f"{folder}/{shard}" for shard in shards)
        else:
            weights = [p for p in in_folder if p.endswith(".safetensors") and not _is_variant(p)]
            if not weights:
                weights = [p for p in in_folder if p.endswith(WEIGHT_EXTS) and not _is_variant(p)]
            selected.update(weights)
    return sorted(selected)


def _local_snapshot(repo_id: str, files: list[str]) -> Path | None:
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, allow_patterns=files, local_files_only=True))
    except Exception:
        return None
    return path if all((path / f).is_file() for f in files) else None


def fetch_pipeline(
    repo_id: str,
    skip: Iterable[str] = (),
    extra: Iterable[str] = (),
    progress: ProgressCb | None = None,
    force: bool = False,
) -> Path:
    """Download (or locate in the cache) exactly the files a diffusers pipeline needs.

    skip:  model_index components not to download (they must be passed as None/objects at load time)
    extra: additional folders to download that model_index does not list (e.g. latent_upsampler)
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    skip, extra = set(skip), list(extra)
    progress = progress or (lambda fraction, message: None)
    manifest = _manifest_path(repo_id, skip, extra)

    if manifest.is_file() and not force:
        files = json.loads(manifest.read_text())["files"]
        local = _local_snapshot(repo_id, files)
        if local is not None:
            return local

    token = _token()
    progress(0.0, f"Checking {repo_id}")
    try:
        tree = {
            entry.path: getattr(entry, "size", 0) or 0
            for entry in HfApi().list_repo_tree(repo_id, recursive=True, token=token)
            if hasattr(entry, "size")
        }
        files = _select_files(repo_id, tree, skip, extra, token)
    except GatedRepoError:
        raise UserError(_gated_message(repo_id)) from None
    except RepositoryNotFoundError:
        raise UserError(f"Model repository {repo_id} was not found (or needs a Hugging Face login).") from None
    except UserError:
        raise
    except Exception as exc:
        raise UserError(
            f"Could not reach Hugging Face to download {repo_id}: {exc}. "
            "Check your internet connection (models download once, then work offline)."
        ) from exc

    total = sum(tree.get(f, 0) for f in files) or 1
    done = 0
    snapshot: Path | None = None
    for i, filename in enumerate(files, 1):
        size = tree.get(filename, 0)
        message = f"Downloading {repo_id} ({total / 1e9:.1f} GB, one-time) · file {i}/{len(files)}"
        progress(done / total, message)
        try:
            local = hf_hub_download(repo_id, filename, token=token)
        except GatedRepoError:
            raise UserError(_gated_message(repo_id)) from None
        done += size
        if filename == "model_index.json":
            snapshot = Path(local).parent
    progress(1.0, f"{repo_id} ready")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"repo": repo_id, "files": files}, indent=1))
    assert snapshot is not None
    return snapshot


def is_pipeline_cached(repo_id: str, skip: Iterable[str] = (), extra: Iterable[str] = ()) -> bool:
    manifest = _manifest_path(repo_id, set(skip), list(extra))
    if not manifest.is_file():
        return False
    files = json.loads(manifest.read_text())["files"]
    return _local_snapshot(repo_id, files) is not None


def fetch_url(url: str, subdir: str, progress: ProgressCb | None = None) -> Path:
    """Download a single file into the VideoGen cache (atomic, skipped if already present)."""
    target = cache_root() / subdir / url.rsplit("/", 1)[-1]
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda fraction, message: None)
    progress(0.0, f"Downloading {target.name}")
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, os.fdopen(fd, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                read += len(chunk)
                if total:
                    progress(read / total, f"Downloading {target.name}")
        shutil.move(tmp, target)
    except Exception as exc:
        Path(tmp).unlink(missing_ok=True)
        raise UserError(f"Could not download {url}: {exc}") from exc
    return target
