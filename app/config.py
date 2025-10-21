import os

from dotenv import load_dotenv

load_dotenv()

CFG = {
    "llm_id": os.getenv("LLM_ID", "microsoft/Phi-3-mini-4k-instruct"),
    "embeddings": os.getenv("EMBEDDINGS", "sentence-transformers/all-MiniLM-L6-v2"),
    "device": os.getenv("DEVICE", "auto"),
    "asr_model": os.getenv("ASR_MODEL", "small"),
    "vad": os.getenv("VAD", "on").lower() == "on",
    "tts_model": os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"),
    "docs_dir": os.getenv("DOCS_DIR", "./data/raw"),
    "chroma_dir": os.getenv("CHROMA_DIR", "./data/chroma"),
    "max_refs": int(os.getenv("MAX_REFERENCES", 4)),
    "gen": {
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 384)),
        "temperature": float(os.getenv("TEMP", 0.2)),
        "top_p": float(os.getenv("TOP_P", 0.9)),
        "top_k": int(os.getenv("TOP_K", 50)),
        "stream": os.getenv("STREAM_TOKENS", "1") == "1",
    },
}
