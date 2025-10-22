import tempfile
from typing import Optional

try:  # pragma: no cover - optional heavy deps
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from TTS.api import TTS as CoquiTTS
except Exception:  # pragma: no cover
    CoquiTTS = None  # type: ignore[assignment]


class Speech:
    """Coqui TTS wrapper that returns a temporary WAV path."""

    def __init__(self, model: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        if CoquiTTS is None:
            raise RuntimeError("Coqui TTS is unavailable. Install TTS to enable speech synthesis.")
        self._tts = CoquiTTS(model)

    def synth(self, text: str, speaker: Optional[str] = None) -> str:
        if sf is None:
            raise RuntimeError("soundfile is required to write audio. Install soundfile to enable speech synthesis.")
        wav = self._tts.tts(text=text, speaker=speaker)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, wav, 22050)
        return path
