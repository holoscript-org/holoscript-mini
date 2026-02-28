import time
import numpy as np
import whisper

_MODEL = whisper.load_model("tiny.en")


def transcribe(audio: np.ndarray) -> str:
    start = time.perf_counter()
    result = _MODEL.transcribe(audio, fp16=False)
    elapsed = time.perf_counter() - start
    print(f"[transcriber] inference: {elapsed * 1000:.1f}ms")
    return result["text"].strip()
