"""Speaker identification using pyannote speaker embeddings."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import inspect
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.config import Settings
from common.schemas import SpeakerIdentificationResult

logger = logging.getLogger(__name__)


class _CompatAudioMetaData(NamedTuple):
    """Backwards-compatible torchaudio metadata shape expected by pyannote."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def _infer_bits_per_sample(subtype: str | None) -> int:
    if not subtype:
        return 0
    match = re.search(r"(\d+)", subtype)
    return int(match.group(1)) if match else 0


def _ensure_torchaudio_compat() -> None:
    """Patch torchaudio 2.9+ API changes that break current pyannote releases."""
    import torch
    import soundfile as sf
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _CompatAudioMetaData  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        def _info(uri: str | Path, backend: str | None = None) -> _CompatAudioMetaData:
            metadata = sf.info(str(uri))
            return _CompatAudioMetaData(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
                bits_per_sample=_infer_bits_per_sample(metadata.subtype),
                encoding=metadata.format or "UNKNOWN",
            )

        torchaudio.info = _info  # type: ignore[attr-defined]

    original_load = torchaudio.load
    if not getattr(original_load, "_pab_patched", False):
        def _compat_load(
            uri: str | Path,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format: str | None = None,
            buffer_size: int = 4096,
            backend: str | None = None,
        ) -> tuple[torch.Tensor, int]:
            try:
                return original_load(
                    uri,
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=normalize,
                    channels_first=channels_first,
                    format=format,
                    buffer_size=buffer_size,
                    backend=backend,
                )
            except (ImportError, ModuleNotFoundError):
                frames = -1 if num_frames == -1 else num_frames
                waveform, sample_rate = sf.read(
                    str(uri),
                    start=frame_offset,
                    frames=frames,
                    dtype="float32",
                    always_2d=True,
                )
                tensor = torch.from_numpy(waveform.T if channels_first else waveform)
                if not normalize:
                    tensor = (tensor * 32767.0).to(torch.int16)
                return tensor, int(sample_rate)

        setattr(_compat_load, "_pab_patched", True)
        torchaudio.load = _compat_load  # type: ignore[assignment]


def _ensure_huggingface_hub_compat() -> None:
    """Patch renamed hf_hub_download auth parameter expected by current pyannote."""
    import huggingface_hub

    signature = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" in signature.parameters:
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def _compat_hf_hub_download(*args: Any, use_auth_token: str | None = None, **kwargs: Any) -> Any:
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _compat_hf_hub_download  # type: ignore[assignment]


def _ensure_torch_load_compat() -> None:
    """Restore pre-2.6 torch.load behavior needed by current pyannote checkpoints."""
    import torch

    signature = inspect.signature(torch.load)
    if "weights_only" not in signature.parameters:
        return

    original_torch_load = torch.load
    if getattr(original_torch_load, "_pab_patched", False):
        return

    def _compat_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    setattr(_compat_torch_load, "_pab_patched", True)
    torch.load = _compat_torch_load  # type: ignore[assignment]


def _ensure_lightning_cloud_io_compat() -> None:
    """Force Lightning checkpoint loader to use trusted-checkpoint semantics."""
    import lightning_fabric.utilities.cloud_io as cloud_io

    original_load = cloud_io._load
    if getattr(original_load, "_pab_patched", False):
        return

    def _compat_load(
        path_or_url: Any,
        map_location: Any = None,
        weights_only: bool | None = None,
    ) -> Any:
        if weights_only is None:
            weights_only = False
        return original_load(path_or_url, map_location=map_location, weights_only=weights_only)

    setattr(_compat_load, "_pab_patched", True)
    cloud_io._load = _compat_load  # type: ignore[assignment]


def _prepare_runtime_cache_dirs(temp_dir: Path) -> None:
    """Redirect matplotlib/font caches to writable project-local locations."""
    mpl_dir = temp_dir / "mplconfig"
    xdg_cache_dir = temp_dir / "xdg-cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


class SpeakerIdentifier:
    """Extract speaker embeddings and identify the closest enrolled resident."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_runtime_dirs()
        self._inference: Any | None = None

    def _get_inference(self) -> Any:
        if self._inference is not None:
            return self._inference

        try:
            _prepare_runtime_cache_dirs(self.settings.temp_dir)
            _ensure_torchaudio_compat()
            _ensure_huggingface_hub_compat()
            _ensure_torch_load_compat()
            _ensure_lightning_cloud_io_compat()
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is required for speaker identification."
            ) from exc

        logger.info(
            "Loading pyannote speaker embedding model: %s",
            self.settings.pyannote_model_name,
        )
        try:
            model = Model.from_pretrained(
                self.settings.pyannote_model_name,
                use_auth_token=self.settings.huggingface_token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the pyannote speaker model. "
                "Check internet access to huggingface.co, and if the model is gated, "
                "set HF_TOKEN in .env after accepting the model terms."
            ) from exc
        if model is None:
            raise RuntimeError(
                "Hugging Face denied access to the pyannote speaker model. "
                "Open https://huggingface.co/pyannote/embedding, sign in, accept the model terms, "
                "and ensure HF_TOKEN in .env belongs to that same account."
            )
        inference = Inference(model, window="whole")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference.to(device)
        self._inference = inference
        return self._inference

    def extract_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Extract a single whole-file speaker embedding."""
        inference = self._get_inference()
        embedding = inference(str(audio_path))
        vector = np.asarray(embedding).reshape(-1).astype(np.float32)
        if vector.size == 0:
            raise RuntimeError("Received an empty speaker embedding.")
        return vector

    def load_voice_db(self) -> dict[str, np.ndarray]:
        """Load enrolled speaker embeddings from disk."""
        embeddings: dict[str, np.ndarray] = {}
        for path in sorted(self.settings.voice_db_dir.glob("*.npy")):
            embeddings[path.stem] = np.load(path).astype(np.float32)
        logger.info("Loaded %s enrolled speaker embedding(s)", len(embeddings))
        return embeddings

    def identify(self, audio_path: str | Path) -> SpeakerIdentificationResult:
        """Return the closest enrolled resident and a relative confidence score."""
        embeddings = self.load_voice_db()
        if not embeddings:
            logger.warning("Voice DB is empty; speaker will be returned as UNKNOWN")
            return SpeakerIdentificationResult()

        query_embedding = self.extract_embedding(audio_path)
        names = list(embeddings.keys())
        similarities = np.array(
            [
                cosine_similarity(
                    query_embedding.reshape(1, -1),
                    stored.reshape(1, -1),
                )[0, 0]
                for stored in embeddings.values()
            ],
            dtype=np.float32,
        )
        similarity_map = {
            name: round(float(score), 4)
            for name, score in zip(names, similarities, strict=True)
        }

        probabilities = np.exp(similarities - np.max(similarities))
        probabilities = probabilities / probabilities.sum()
        best_index = int(np.argmax(probabilities))

        result = SpeakerIdentificationResult(
            speaker=names[best_index],
            confidence=round(float(probabilities[best_index]), 4),
            similarities=similarity_map,
        )
        logger.info(
            "Speaker identified as %s with confidence %.3f",
            result.speaker,
            result.confidence,
        )
        return result
