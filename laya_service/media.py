"""Small media helpers kept in the service so the public laya-omni package is enough."""

from __future__ import annotations

from typing import Any


def load_video(path: str, n_frames: int = 4, sampling_rate: int = 16_000) -> tuple[list[Any], Any]:
    """Turn a video into evenly spaced PIL frames and its optional mono audio track."""
    import av
    import numpy as np
    from laya_omni.audio import _decode_av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("file has no video stream")
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
        has_audio = bool(container.streams.audio)
    if not frames:
        raise ValueError("could not decode any video frames")
    indexes = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
    audio = _decode_av(path, sampling_rate) if has_audio else None
    if audio is not None and not audio.size:
        audio = None
    return [frames[index] for index in indexes], audio
