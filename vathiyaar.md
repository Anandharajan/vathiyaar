# **Vathiyaar** — Agentic AI Voice Student Assistant

### **Codex CLI → GitHub → Hugging Face Space (GPU) — Lean MVP with Fast Response**

> This is a **finetuned, end-to-end Codex CLI playbook** to build, version, and deploy the voice-first, agentic student assistant **Vathiyaar** on **Hugging Face Spaces with a GPU**.
> It preserves your earlier plan (ASR → RAG → LLM → TTS, citations, memory, guardrails) and adapts it for Spaces (no Ollama dependency; uses **Transformers** on GPU).
> UI: clean Gradio app with streaming text + audio. Agent tools: RAG retriever + calculator.

---

## 0) TL;DR (what you’ll get)

* 🎙️ **Voice in** (Faster-Whisper, GPU-accelerated where available)
* 🔎 **RAG** over PDFs (Chroma + SentenceTransformers) with **citations** `[Doc p.X]`
* 🧠 **LLM**: small, fast **Transformers** instruct model on GPU (defaults to **Phi-3-mini** or **Qwen2.5-1.5B-Instruct**)
* 🗣️ **Voice out** via Coqui TTS
* 🧩 **Agent tools**: calculator + retriever
* ⚡ **Fast**: streaming tokens, light model, minimal prompt, caching
* 🌐 **Runs anywhere** via browser; **deployed** to Hugging Face Space with **GPU**
* 🧭 Name: **vathiyaar**

---

## 1) Create project with Codex CLI

```bash
codex sh "mkdir -p vathiyaar && cd vathiyaar && git init -b main"
codex sh "mkdir -p app data/raw data/chroma tests .github/workflows"
codex sh "touch README.md requirements.txt .env.example Dockerfile"
```

---

## 2) Space metadata in README (Hugging Face front-matter)

```bash
codex file write README.md <<'MD'
---
title: vathiyaar
emoji: 🎓
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: 3.11
app_file: app/main.py
pinned: true
license: apache-2.0
---

# Vathiyaar — Agentic AI Voice Student Assistant (GPU)

Voice in → ASR (Whisper) → RAG over your notes → LLM (small, fast) → TTS reply, with citations and guardrails.  
This Space targets **low latency**: streaming tokens, compact models, and GPU inference.

## Quick Start
- Upload a few PDFs in the app, or mount a dataset repo.  
- Ask syllabus questions; Vathiyaar cites `[Doc p.X]` when using your notes.  
- Toggle models in Settings for best speed on your hardware.

## Notes
- For **GPU**: choose **Hardware → GPU (T4 or A10G)** in Space Settings.  
- Data is not persisted unless you enable **persistent storage** on the Space.
MD
```

---

## 3) Configuration

```bash
codex file write .env.example <<'ENV'
LLM_ID=microsoft/Phi-3-mini-4k-instruct   # alt: Qwen/Qwen2.5-1.5B-Instruct
EMBEDDINGS=sentence-transformers/all-MiniLM-L6-v2
DEVICE=auto                 # cuda | mps | cpu
ASR_MODEL=small             # tiny|base|small|medium|large-v2
VAD=on
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
DOCS_DIR=./data/raw
CHROMA_DIR=./data/chroma
MAX_REFERENCES=4
MAX_NEW_TOKENS=384
TEMP=0.2
TOP_P=0.9
TOP_K=50
STREAM_TOKENS=1
ENV
```

---

## 4) Dependencies (GPU-friendly)

```bash
codex file write requirements.txt <<'REQ'
gradio==4.44.0
transformers>=4.44.0
accelerate>=0.34.0
torch  # use Space's CUDA build
bitsandbytes>=0.43.0    # 4-bit loading where available (T4/A10G)
faster-whisper==1.0.3
chromadb==0.5.5
sentence-transformers==3.1.1
pypdf==4.3.1
numpy==2.1.2
onnxruntime-gpu; platform_system != 'Darwin'
TTS==0.22.0
soundfile==0.12.1
python-dotenv==1.0.1
pytest==8.3.2
REQ
```

> **Tip:** On Spaces GPU, `torch` is preinstalled; keeping it unpinned avoids conflicts.

---

## 5) Prompts

```bash
codex file write app/prompts.py <<'PY'
SYSTEM_PROMPT = (
  "You are 'Vathiyaar', a precise, friendly student assistant.\n"
  "Rules:\n"
  "1) Prefer short, stepwise answers.\n"
  "2) If the query is factual/syllabus-based, retrieve from notes; cite as [Doc p.X].\n"
  "3) If unsure, say so and suggest how to verify.\n"
  "4) For math: show steps, check units, verify result.\n"
)

RAG_PROMPT = (
  "Question: {question}\n"
  "Context (top-{k} chunks):\n{contexts}\n"
  "Answer concisely with steps when useful; include citations to the most relevant chunks."
)
PY
```

---

## 6) Config & Tools

```bash
codex file write app/config.py <<'PY'
import os
from dotenv import load_dotenv
load_dotenv()

CFG = {
  "llm_id": os.getenv("LLM_ID","microsoft/Phi-3-mini-4k-instruct"),
  "embeddings": os.getenv("EMBEDDINGS","sentence-transformers/all-MiniLM-L6-v2"),
  "device": os.getenv("DEVICE","auto"),
  "asr_model": os.getenv("ASR_MODEL","small"),
  "vad": os.getenv("VAD","on")=="on",
  "tts_model": os.getenv("TTS_MODEL","tts_models/en/ljspeech/tacotron2-DDC"),
  "docs_dir": os.getenv("DOCS_DIR","./data/raw"),
  "chroma_dir": os.getenv("CHROMA_DIR","./data/chroma"),
  "max_refs": int(os.getenv("MAX_REFERENCES",4)),
  "gen": {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS",384)),
    "temperature": float(os.getenv("TEMP",0.2)),
    "top_p": float(os.getenv("TOP_P",0.9)),
    "top_k": int(os.getenv("TOP_K",50)),
    "stream": os.getenv("STREAM_TOKENS","1")=="1",
  },
}
PY
```

```bash
codex file write app/tools.py <<'PY'
import math
def safe_calculate(expr: str):
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"__builtins__": {}})
    try:
        return str(eval(expr, allowed, {}))
    except Exception as e:
        return f"calc_error: {e}"

def cite_blocks(blocks):
    cites = []
    for _, m in blocks:
        cites.append(f"[{m['source']} p.{m['page']}]")
    return sorted(set(cites))[:4]
PY
```

```bash
codex file write app/memory.py <<'PY'
from collections import deque
class Memory:
    def __init__(self, max_turns=10): self.buf = deque(maxlen=max_turns)
    def add(self, user, assistant): self.buf.append((user, assistant))
    def window(self): return "\n".join([f"U{i}: {u}\nA{i}: {a}" for i,(u,a) in enumerate(self.buf)])
PY
```

---

## 7) ASR, RAG, and TTS modules

```bash
codex file write app/asr.py <<'PY'
from faster_whisper import WhisperModel
import numpy as np

class ASR:
    def __init__(self, model_name: str = "small", device: str = "auto"):
        self.model = WhisperModel(model_name, device=device, compute_type="int8")
    def transcribe(self, audio_np: np.ndarray, sample_rate: int):
        segments, _ = self.model.transcribe(audio_np, beam_size=1)
        text = " ".join([seg.text for seg in segments])
        return text.strip()
PY
```

```bash
codex file write app/rag.py <<'PY'
import argparse, os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from app.config import CFG

EMB = SentenceTransformer(CFG["embeddings"])
client = chromadb.PersistentClient(path=CFG["chroma_dir"])
COL = client.get_or_create_collection("study_notes")

def split_text(t, chunk=800, overlap=120):
    t = " ".join(t.split()); out=[]; i=0
    while i < len(t): out.append(t[i:i+chunk]); i += chunk-overlap
    return out

def embed(texts): return EMB.encode(texts, normalize_embeddings=True).tolist()

def ingest():
    ids, docs, metas = [], [], []
    for fn in os.listdir(CFG["docs_dir"]):
        if not fn.lower().endswith(".pdf"): continue
        path = os.path.join(CFG["docs_dir"], fn)
        pdf = PdfReader(path)
        for pi, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            for ci, chunk in enumerate(split_text(raw)):
                ids.append(f"{fn}-{pi}-{ci}")
                docs.append(chunk)
                metas.append({"source": fn, "page": pi+1})
    if ids:
        COL.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
        print(f"Indexed {len(ids)} chunks.")

def query(q, k=4):
    res = COL.query(query_texts=[q], n_results=k, include=["documents","metadatas","distances"])
    docs = res.get("documents",[[]])[0]
    metas = res.get("metadatas",[[]])[0]
    return [(d,m) for d,m in zip(docs,metas)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        try: client.delete_collection("study_notes")
        except: pass
        ingest()
PY
```

```bash
codex file write app/tts.py <<'PY'
import soundfile as sf
from TTS.api import TTS
import uuid

class Speech:
    def __init__(self, model="tts_models/en/ljspeech/tacotron2-DDC"):
        self.tts = TTS(model)
    def synth(self, text, voice=None):
        path = f"/tmp/{uuid.uuid4()}.wav"
        wav = self.tts.tts(text=text)
        sf.write(path, wav, 22050)
        return path
PY
```

---

## 8) Lightweight, fast LLM wrapper (Transformers, GPU/4-bit)

```bash
codex file write app/llm.py <<'PY'
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from app.config import CFG

_model = None
_tok = None

def _load():
    global _model, _tok
    if _model is None:
        llm_id = CFG["llm_id"]
        device = 0 if torch.cuda.is_available() else "cpu"
        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        try:
            # Try 4-bit if available (faster/smaller on T4/A10G)
            import bitsandbytes as bnb  # noqa
            kwargs.update(dict(load_in_4bit=True))
        except Exception:
            pass
        _tok = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(llm_id, **kwargs)
    return _tok, _model

def generate(prompt: str):
    tok, model = _load()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = CFG["gen"]
    if gen_cfg["stream"]:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        for text in streamer:
            yield text
    else:
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=True,
        )
        yield tok.decode(out[0], skip_special_tokens=True)
PY
```

---

## 9) The Gradio app (agentic + streaming + lean UI)

```bash
codex file write app/main.py <<'PY'
import gradio as gr, numpy as np
from app.asr import ASR
from app.rag import query
from app.tts import Speech
from app.tools import safe_calculate, cite_blocks
from app.memory import Memory
from app.config import CFG
from app.prompts import SYSTEM_PROMPT, RAG_PROMPT
from app.llm import generate

ASR_ENGINE = ASR(CFG["asr_model"], CFG["device"])
TTS_ENGINE = Speech(CFG["tts_model"])
MEM = Memory()

def agentic_answer(user_text: str):
    # quick routing
    if any(tok in user_text for tok in ["solve","calculate","evaluate","="]):
        return f"Computation: {safe_calculate(user_text.replace('solve',''))}", None
    blocks = query(user_text, k=CFG["max_refs"]) or []
    ctx = "\n---\n".join([b[0] for b in blocks])
    cites = " ".join(cite_blocks(blocks))
    prompt = (
        f"{SYSTEM_PROMPT}\n\nUser: {user_text}\n\nContext:\n{ctx}\n\n"
        f"Answer with steps when useful and include citations like {cites} if used."
    )
    return prompt, cites

def pipeline(audio, history):
    if audio is None: return None, history
    sr, y = audio
    if isinstance(y, (bytes, bytearray)):
        import soundfile as sf, io
        y, sr = sf.read(io.BytesIO(y))
    text = ASR_ENGINE.transcribe(np.array(y, dtype=np.float32), sr)

    prompt_or_answer, cites = agentic_answer(text)
    if cites is None:
        answer = prompt_or_answer
        wavpath = TTS_ENGINE.synth(answer)
        MEM.add(text, answer)
        history = history + [(text, answer)]
        return (wavpath, answer), history

    # streaming tokens
    stream = generate(prompt_or_answer)
    collected = ""
    for chunk in stream:
        collected += chunk
        yield None, history + [(text, collected)]

    answer = collected.strip()
    wavpath = TTS_ENGINE.synth(answer)
    MEM.add(text, answer)
    history = history + [(text, answer)]
    yield (wavpath, answer), history

with gr.Blocks(css=".wrap {max-width: 920px !important;}") as demo:
    gr.Markdown("## 🎓 Vathiyaar — Agentic Voice Doubt Solver")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="numpy", streaming=False, label="Hold & Ask")
        out_audio = gr.Audio(label="Vathiyaar", autoplay=True)
    chat = gr.Chatbot(label="Transcript", height=360, avatar_images=(None, None))
    ask = gr.Button("Answer")
    state = gr.State(value=[])

    ask.click(pipeline, inputs=[mic, state], outputs=[out_audio, chat], api_name="ask")

if __name__ == "__main__":
    demo.queue(max_size=64, concurrency_count=2).launch()
PY
```

---

## 10) Smoke test

```bash
codex file write tests/smoke_test.py <<'PY'
def test_imports():
    import app.asr, app.rag, app.tts, app.tools, app.memory, app.config, app.prompts, app.llm, app.main
    assert True
PY
codex sh "pytest -q"
```

---

## 11) Optimize latency (Space-ready tweaks)

* **Model choice:** default `microsoft/Phi-3-mini-4k-instruct` (fast); try `Qwen/Qwen2.5-1.5B-Instruct` if similar speed.
* **Quantization:** `bitsandbytes` 4-bit auto-enabled if available.
* **Streaming:** enabled; first token fast.
* **Batching:** Gradio `.queue(concurrency_count=2)`; tune to GPU.
* **ASR:** `ASR_MODEL=small` (or `base`) for speed.
* **TTS:** tacotron2 is light.
* **GPU:** choose **T4 Small** or **A10G** in Space settings (see below).

---

## 12) Dockerfile (optional — Spaces can use default Python)

```bash
codex file write Dockerfile <<'DOCKER'
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y ffmpeg git python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python","-m","app.main"]
DOCKER
```

> If you select **Docker** Space type, set **App file** to `Dockerfile`. Otherwise keep default (Python) with `app_file: app/main.py`.

---

## 13) Commit & push to GitHub

```bash
codex sh 'git add .'
codex sh 'git commit -m "feat(vathiyaar): agentic voice assistant (ASR+RAG+LLM+TTS), Spaces-ready"'
codex sh 'git branch -M main'
codex sh 'git remote add origin <YOUR_GITHUB_REPO_URL>'
codex sh 'git push -u origin main'
```

---

## 14) Create the Hugging Face Space (GPU) from GitHub

> **Spaces → New Space → From Repo**
>
> * **Repository**: link to your GitHub repo
> * **Space SDK**: **Gradio** (or **Docker** if using Dockerfile)
> * **App file**: `app/main.py`
> * **Hardware**: **GPU (T4 Small)** or **A10G** for lower latency
> * (Optional) **Persistent storage** if you want to keep the Chroma index
> * **Secrets**: add provider keys only if you later swap to cloud LLM/TTS

> Hardware selection is a UI step; it cannot be enforced by repo files. Choose GPU to enable CUDA.

---

## 15) Populate notes & build index on Space

* Either **upload PDFs** via the UI component you add (or mount a dataset).
* On first run, trigger:

```bash
# locally; on Space you can add a small "Rebuild Index" button (recommended) that calls:
# python -m app.rag --rebuild
codex sh "python -m app.rag --rebuild"
```

*(If you want a button in the Space, we can add a Gradio `Button` that calls `ingest()` and refreshes.)*

---

## 16) Daily dev loop (Codex CLI)

```bash
# Edit code
codex file edit app/main.py

# Test
codex sh "pytest -q"

# Run locally
codex sh "python -m app.main"

# Reindex after adding PDFs
codex sh "python -m app.rag --rebuild"

# Commit
codex sh 'git add . && git commit -m "perf: faster streaming + smaller ASR" && git push'
```

---

## 17) Troubleshooting & speed notes

* **Cold start:** first load downloads models → subsequent runs are faster (cached).
* **OOM:** switch to `Phi-3-mini` (default) or reduce `MAX_NEW_TOKENS`.
* **CPU fallback:** if no GPU, it still runs; slower.
* **ASR lag:** set `ASR_MODEL=base` or `tiny`.
* **TTS glitch:** ensure `ffmpeg` present (added in Dockerfile; on Python Space, Spaces include it).
* **RAG empty:** ensure PDFs uploaded and reindex.

---

## 18) Optional: nicer landing UI polish (lean)

* Add brand name **Vathiyaar** in header, subtitle “Ask Me Anything from Your Syllabus.”
* Add small **Settings** accordion for model choice & speed/quality knobs.
* Add **Upload PDFs** widget + **Rebuild Index** button.
* Add **Latency panel** (p50/p90) with simple timestamps around ASR/LLM/TTS.

*(Say the word and I’ll drop the extra UI code + buttons into `app/main.py`.)*

---

### You’re ready to ship **Vathiyaar** 🚀

Push to GitHub → create Space from repo → choose **GPU** → run.
You’ll have an agentic, voice-first student assistant with citations and streaming, optimized for **minimum response time** on Hugging Face Spaces.
