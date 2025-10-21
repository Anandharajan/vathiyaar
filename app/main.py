import io
from typing import Generator, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf

from app.asr import ASR
from app.config import CFG
from app.llm import format_prompt, generate
from app.memory import Memory
from app.rag import ingest, query
from app.tools import cite_blocks, safe_calculate
from app.tts import Speech

ASR_ENGINE = ASR(CFG["asr_model"], CFG["device"])
TTS_ENGINE = Speech(CFG["tts_model"])
MEMORY = Memory()


def _ensure_numpy(audio: Tuple[int, np.ndarray] | Tuple[int, bytes]) -> Tuple[int, np.ndarray]:
    sample_rate, payload = audio
    if isinstance(payload, (bytes, bytearray)):
        data, sample_rate = sf.read(io.BytesIO(payload))
        payload = np.asarray(data, dtype=np.float32)
    return sample_rate, np.asarray(payload, dtype=np.float32)


def agentic_plan(user_text: str) -> Tuple[bool, str, List[str]]:
    """Decide whether to answer directly or call the LLM."""
    lowered = user_text.lower()
    if any(token in lowered for token in ["solve", "calculate", "evaluate", "="]):
        return True, safe_calculate(user_text), []

    blocks = query(user_text, k=CFG["max_refs"])
    context = "\n---\n".join(block for block, _ in blocks)
    citations = cite_blocks(blocks, limit=CFG["max_refs"])
    history = MEMORY.window()
    prompt = format_prompt(user_text, context, " ".join(citations), history)
    return False, prompt, citations


def pipeline(audio: Optional[Tuple[int, np.ndarray]], history: List[Tuple[str, str]]) -> Generator:
    if audio is None:
        yield None, history, history
        return

    sample_rate, samples = _ensure_numpy(audio)
    transcript, _ = ASR_ENGINE.transcribe(samples, sample_rate)
    if not transcript:
        yield None, history + [("[error]", "Did not catch that, please try again.")], history
        return

    direct, payload, citations = agentic_plan(transcript)

    if direct:
        answer = payload
        wav_path = TTS_ENGINE.synth(answer)
        updated = history + [(transcript, answer)]
        MEMORY.add(transcript, answer)
        yield wav_path, updated, updated
        return

    collected = ""
    updated_history = history
    for chunk in generate(payload):
        collected += chunk
        updated_history = history + [(transcript, collected)]
        yield None, updated_history, updated_history

    answer = collected.strip()
    if not answer:
        answer = "I could not generate an answer right now."
    if citations:
        answer = f"{answer}\n\n{' '.join(citations)}"
    wav_path = TTS_ENGINE.synth(answer)
    MEMORY.add(transcript, answer)
    final_history = history + [(transcript, answer)]
    yield wav_path, final_history, final_history


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


def rebuild_index() -> None:
    """Convenience entry-point for local reindexing."""
    ingest()


if __name__ == "__main__":
    demo.queue(max_size=64, concurrency_count=2).launch()
