"""
Whisper-based speech transcription with preprocessing.

Converts audio to text using OpenAI's Whisper model with audio cleaning.
"""

import time
import numpy as np
import whisper
from core.utils.logger import get_logger
from voice.audio_utils import normalize_audio, trim_silence, noise_gate

logger = get_logger("transcriber")

_MODEL = whisper.load_model("tiny.en")


def transcribe(
    audio: np.ndarray | None,
    language: str = "en",
) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio: numpy array of audio samples, or None
        language: Language code (default: "en" for English)

    Returns:
        Transcribed text, or empty string if transcription failed
    """
    if audio is None:
        logger.warning("Audio is None, returning empty string")
        return ""

    if len(audio) == 0:
        logger.warning("Audio array is empty, returning empty string")
        return ""

    # Preprocess audio
    try:
        audio = normalize_audio(audio)
        audio = trim_silence(audio)
        audio = noise_gate(audio)
    except Exception as e:
        logger.error("Audio preprocessing failed: %s", e)
        return ""

    # Transcribe with Whisper
    start = time.perf_counter()
    try:
        result = _MODEL.transcribe(
            audio,
            language=language,
            task="transcribe",
            fp16=False,
        )
        elapsed = time.perf_counter() - start
        logger.info("Whisper latency: %.1fms", elapsed * 1000)

        text = result.get("text", "").strip()

        # Reject very short transcriptions (likely noise)
        if len(text) < 3:
            logger.warning("Transcription too short (%d chars), returning empty", len(text))
            return ""

        return text

    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return ""
