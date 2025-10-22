"""Gradio entry-point for the Vathiyaar voice assistant."""

from __future__ import annotations

import io
from typing import Generator, List, Optional, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional heavy dependency for tests
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependency for tests
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependency for tests
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np_typing

    AudioArray = np_typing.ndarray
else:  # pragma: no cover
    AudioArray = object

from app.asr import ASR
from app.config import CFG
from app.llm import format_prompt, generate
from app.memory import Memory
from app.rag import ingest, query
from app.tools import cite_blocks, safe_calculate
from app.tts import Speech

MEMORY = Memory()
_ASR_ENGINE: Optional[ASR] = None
_TTS_ENGINE: Optional[Speech] = None


def _ensure_numpy(audio: Tuple[int, AudioArray] | Tuple[int, bytes]) -> Tuple[int, AudioArray]:
    if np is None:
        raise RuntimeError("numpy is required for audio processing. Install numpy to enable transcription.")

    sample_rate, payload = audio
    data = payload
    if isinstance(payload, (bytes, bytearray)):
        if sf is None:
            raise RuntimeError(
                "soundfile is required to decode audio payloads. Install soundfile to enable audio uploads."
            )
        data, sample_rate = sf.read(io.BytesIO(payload))
    array = np.asarray(data, dtype=np.float32)
    return sample_rate, array  # type: ignore[return-value]


def _get_asr_engine() -> ASR:
    global _ASR_ENGINE
    if _ASR_ENGINE is None:
        _ASR_ENGINE = ASR(CFG["asr_model"], CFG["device"])
    return _ASR_ENGINE


def _get_tts_engine() -> Speech:
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        _TTS_ENGINE = Speech(CFG["tts_model"])
    return _TTS_ENGINE


def _synthesize(answer: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        engine = _get_tts_engine()
    except RuntimeError as exc:
        return None, f"[speech unavailable: {exc}]"
    try:
        return engine.synth(answer), None
    except Exception as exc:  # noqa: BLE001
        return None, f"[speech synthesis failed: {exc}]"


def agentic_plan(user_text: str) -> Tuple[bool, str, List[str]]:
    """Decide whether to answer directly or call the LLM."""
    lowered = user_text.lower()
    if any(token in lowered for token in ["solve", "calculate", "evaluate", "="]):
        return True, safe_calculate(user_text), []

    context = ""
    citations: List[str] = []
    try:
        blocks = query(user_text, k=CFG["max_refs"])
    except RuntimeError:
        blocks = []
    else:
        context = "\n---\n".join(block for block, _ in blocks)
        citations = cite_blocks(blocks, limit=CFG["max_refs"])
    history = MEMORY.window()
    prompt = format_prompt(user_text, context, " ".join(citations), history)
    return False, prompt, citations


def pipeline(audio: Optional[Tuple[int, AudioArray]], history: List[Tuple[str, str]]) -> Generator:
    if audio is None:
        yield None, history, history
        return

    try:
        sample_rate, samples = _ensure_numpy(audio)
    except RuntimeError as exc:
        updated = history + [("[error]", str(exc))]
        yield None, updated, updated
        return

    try:
        asr_engine = _get_asr_engine()
    except RuntimeError as exc:
        updated = history + [("[error]", str(exc))]
        yield None, updated, updated
        return

    try:
        transcript, _ = asr_engine.transcribe(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001
        updated = history + [("[error]", f"transcription_error: {exc}")]
        yield None, updated, updated
        return

    if not transcript:
        updated = history + [("[error]", "Did not catch that, please try again.")]
        yield None, updated, updated
        return

    direct, payload, citations = agentic_plan(transcript)

    if direct:
        answer = payload
        wav_path, warn = _synthesize(answer)
        if warn:
            answer = f"{answer}\n\n{warn}"
        updated = history + [(transcript, answer)]
        MEMORY.add(transcript, answer)
        yield wav_path, updated, updated
        return

    collected = ""
    updated_history = history
    try:
        for chunk in generate(payload):
            collected += chunk
            updated_history = history + [(transcript, collected)]
            yield None, updated_history, updated_history
    except Exception as exc:  # noqa: BLE001
        message = f"llm_error: {exc}"
        updated_history = history + [(transcript, message)]
        MEMORY.add(transcript, message)
        yield None, updated_history, updated_history
        return

    answer = collected.strip() or "I could not generate an answer right now."
    if citations:
        answer = f"{answer}\n\n{' '.join(citations)}"

    wav_path, warn = _synthesize(answer)
    if warn:
        answer = f"{answer}\n\n{warn}"
    MEMORY.add(transcript, answer)
    final_history = history + [(transcript, answer)]
    yield wav_path, final_history, final_history


if gr is not None:  # pragma: no cover - UI wiring
    with gr.Blocks(css=".wrap {max-width: 920px !important;}") as demo:
        gr.Markdown("## Vathiyaar - Agentic Voice Doubt Solver")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", streaming=False, label="Hold & Ask")
            out_audio = gr.Audio(label="Vathiyaar", autoplay=True, type="filepath")
        chat = gr.Chatbot(label="Transcript", height=360)
        ask = gr.Button("Answer")
        state = gr.State([])

        ask.click(
            pipeline,
            inputs=[mic, state],
            outputs=[out_audio, chat, state],
            api_name="ask",
        )
else:  # pragma: no cover - tests without Gradio
    demo = None


def rebuild_index() -> None:
    """Convenience entry-point for local reindexing."""
    ingest()


if __name__ == "__main__":  # pragma: no cover - manual launch
    if demo is None:
        raise RuntimeError("Gradio is not installed. Install gradio to launch the interface.")
    demo.queue(max_size=64, concurrency_count=2).launch()
