from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional dependency for tests
    _np = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency for tests
    WhisperModel = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np


class ASR:
    """Wrapper around Faster-Whisper for quick transcription."""

    def __init__(self, model_name: str = "small", device: str = "auto") -> None:
        if _np is None or WhisperModel is None:
            raise RuntimeError(
                "ASR dependencies missing. Install numpy and faster-whisper to enable transcription."
            )
        self._np = _np
        self.model = WhisperModel(model_name, device=device, compute_type="int8")

    def transcribe(self, audio_np: "np.ndarray", sample_rate: int) -> Tuple[str, float]:
        segments, info = self.model.transcribe(audio_np, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language_probability
