import tempfile
from typing import Optional

import soundfile as sf
from TTS.api import TTS


class Speech:
    """Coqui TTS wrapper that returns a temporary WAV path."""

    def __init__(self, model: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        self._tts = TTS(model)

    def synth(self, text: str, speaker: Optional[str] = None) -> str:
        wav = self._tts.tts(text=text, speaker=speaker)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, wav, 22050)
        return path
