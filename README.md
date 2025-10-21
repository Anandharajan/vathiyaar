---
title: vathiyaar
emoji: "🎙️"
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: 3.11
app_file: app/main.py
pinned: true
license: apache-2.0
---

# Vathiyaar - Agentic AI Voice Student Assistant (GPU)

Voice in + ASR (Faster-Whisper) + RAG over your notes + compact LLM + TTS reply, with citations and guardrails.
This Space targets low latency with streaming tokens, lightweight models, and GPU inference.

## Quick Start
- Upload a few PDFs in the app or mount a dataset repo.
- Ask syllabus questions; Vathiyaar cites chunks as `[Doc p.X]` when it uses your notes.
- Toggle models in Settings to balance speed and quality for your hardware.

## Notes
- For GPU acceleration choose **Hardware → GPU (T4 or A10G)** in Space Settings.
- Data is not persisted unless you enable persistent storage for the Space.
