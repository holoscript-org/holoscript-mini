import time
import numpy as np
import whisper
from core.utils.logger import get_logger

logger = get_logger("transcriber")

_MODEL = whisper.load_model("tiny.en")


def transcribe(audio: np.ndarray) -> str:
    start = time.perf_counter()
    result = _MODEL.transcribe(audio, fp16=False)
    elapsed = time.perf_counter() - start
    logger.info("inference: %.1fms", elapsed * 1000)
    return result["text"].strip()
