import shutil
import subprocess

import pytest

from videogen.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOGEN_CACHE", str(tmp_path / "cache"))
    return Settings(output_dir=tmp_path / "outputs", enable_mock=True, mock_step_delay=0.0)


def ffprobe(path) -> dict:
    """Return {"video": {...}, "audio": {...}} stream info for a media file."""
    exe = shutil.which("ffprobe")
    if exe is None:
        pytest.skip("ffprobe not installed")
    out = subprocess.run(
        [exe, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,sample_rate,channels",
         "-of", "default=noprint_wrappers=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams, current = {}, None
    for line in out.splitlines():
        if line == "[STREAM]":
            current = {}
        elif line == "[/STREAM]":
            streams[current["codec_type"]] = current
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            current[key] = value
    return streams
