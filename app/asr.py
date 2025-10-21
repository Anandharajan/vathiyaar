from typing import Tuple

import numpy as np
from faster_whisper import WhisperModel


class ASR:
    """Wrapper around Faster-Whisper for quick transcription."""

    def __init__(self, model_name: str = "small", device: str = "auto") -> None:
        self.model = WhisperModel(model_name, device=device, compute_type="int8")

    def transcribe(self, audio_np: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        segments, info = self.model.transcribe(audio_np, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language_probability
