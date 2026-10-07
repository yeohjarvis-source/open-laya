from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from .config import Settings


class PredictionEngine(Protocol):
    default_model: str

    def load(self, model_id: str | None = None) -> None: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def model_revision(self, model_id: str) -> str | None: ...
    def predict(
        self,
        model_id: str,
        state: Any,
        questions: dict[str, Any],
        media: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class LayaTextEngine:
    """Lazy, process-local laya-mlx model holder."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._agent: Any = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._agent is not None:
            return
        with self._load_lock:
            if self._agent is not None:
                return
            try:
                import laya_mlx as laya
            except ModuleNotFoundError as exc:
                if exc.name != "laya_mlx":
                    raise RuntimeError("laya-mlx could not initialize its runtime") from exc
                raise RuntimeError(
                    "laya-mlx is not installed; install this project with the 'mlx' extra"
                ) from exc
            except ImportError as exc:
                raise RuntimeError("laya-mlx could not initialize its MLX/Metal runtime") from exc
            kwargs: dict[str, Any] = {"dtype": self.settings.model_dtype}
            if self.settings.model_revision:
                kwargs["revision"] = self.settings.model_revision
            self._agent = laya.load(self.settings.model_id, **kwargs)

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.load()
        return self._agent.predict(state, questions)


class LayaOmniEngine:
    """Lazy laya-omni holder. Media is decoded only for an omni request."""

    def __init__(self, kit_path: Path):
        self.kit_path = kit_path
        self._agent: Any = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def load(self) -> None:
        if self._agent is not None:
            return
        with self._load_lock:
            if self._agent is not None:
                return
            if not self.kit_path.joinpath("laya_kit.py").is_file():
                raise RuntimeError(f"Invalid laya-omni kit path: {self.kit_path}")
            sys.path.insert(0, str(self.kit_path))
            try:
                from laya_kit import load

                self._agent = load()
            except ImportError as exc:
                raise RuntimeError(
                    "laya-omni dependencies are unavailable; run laya-omni-kit/setup.sh and "
                    "start this service with that environment"
                ) from exc
            finally:
                if sys.path and sys.path[0] == str(self.kit_path):
                    sys.path.pop(0)

    @staticmethod
    def _temp_file(item: dict[str, Any]) -> str:
        suffix = Path(item.get("filename") or "").suffix or ".bin"
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        handle.write(item["data"])
        handle.close()
        return handle.name

    def predict(
        self, state: Any, questions: dict[str, Any], media: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.load()
        media = media or {}
        images = []
        for item in media.get("images", []):
            try:
                from PIL import Image

                images.append(Image.open(io.BytesIO(item["data"])).convert("RGB"))
            except Exception as exc:
                raise ValueError(f"{item.get('filename') or 'image'}: not a readable image") from exc

        paths: list[str] = []
        wav = None
        try:
            audio = media.get("audio")
            if audio:
                paths.append(self._temp_file(audio))
                try:
                    from laya_omni.encoders import load_audio

                    wav = load_audio(paths[-1], self._agent.audio_encoder.sampling_rate)
                except Exception as exc:
                    raise ValueError(f"{audio.get('filename') or 'audio'}: could not decode audio") from exc

            video = media.get("video")
            if video:
                paths.append(self._temp_file(video))
                try:
                    from laya_omni.video import load_video

                    room = 8 - len(images)
                    if room < 1:
                        raise ValueError("at most 8 images including video frames")
                    frames, video_wav = load_video(
                        paths[-1], max(1, min(int(media.get("video_frames", 4)), room))
                    )
                    images.extend(frames)
                    if wav is None and self._agent.audio_encoder is not None:
                        wav = video_wav
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(f"{video.get('filename') or 'video'}: could not decode video") from exc
        finally:
            for path in paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if len(images) > 8:
            raise ValueError("at most 8 images including video frames")
        image: Any = images if len(images) > 1 else (images[0] if images else None)
        with self._predict_lock:
            return self._agent.predict(
                state, questions, image=image, audio=wav, detail=bool(media.get("detail", False))
            )


class ModelRouter:
    """Stable public model IDs over the two different local runtimes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.default_model = settings.default_model
        self._engines: dict[str, Any] = {}
        if "laya" in settings.enabled_models:
            self._engines["laya"] = LayaTextEngine(settings)
        if "laya-omni" in settings.enabled_models:
            assert settings.omni_kit_path is not None
            self._engines["laya-omni"] = LayaOmniEngine(settings.omni_kit_path)

    def load(self, model_id: str | None = None) -> None:
        self._get(model_id or self.default_model).load()

    def _get(self, model_id: str) -> Any:
        engine = self._engines.get(model_id)
        if engine is None:
            raise KeyError(model_id)
        return engine

    def list_models(self) -> list[dict[str, Any]]:
        descriptions = {
            "laya": ("Laya", ["text"]),
            "laya-omni": ("Laya Omni", ["text", "image", "audio", "video"]),
        }
        return [
            {"id": model_id, "name": descriptions[model_id][0], "inputs": descriptions[model_id][1]}
            for model_id in self._engines
        ]

    def model_revision(self, model_id: str) -> str | None:
        return self.settings.model_revision if model_id == "laya" else None

    def predict(
        self,
        model_id: str,
        state: Any,
        questions: dict[str, Any],
        media: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        engine = self._get(model_id)
        if model_id == "laya":
            if media and any(media.get(key) for key in ("images", "audio", "video")):
                raise ValueError("The laya model accepts text only; select laya-omni for media")
            return engine.predict(state, questions)
        return engine.predict(state, questions, media)


# Backwards-compatible import for callers that used the old class name.
LayaEngine = ModelRouter
