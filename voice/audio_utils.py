"""
Audio preprocessing and cleaning utilities.

Provides normalization, silence trimming, noise gating, and optional filtering.
"""

import numpy as np
from scipy.signal import butter, lfilter


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range using peak normalization.

    Args:
        audio: numpy array of audio samples

    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: numpy array of audio samples
        threshold: Amplitude threshold below which is considered silence

    Returns:
        Audio with silence trimmed
    """
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    start = np.argmax(mask)
    end = len(audio) - np.argmax(mask[::-1])
    return audio[start:end]


def noise_gate(audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Apply noise gate - zero out samples below threshold.

    Args:
        audio: numpy array of audio samples
        threshold: Amplitude threshold below which samples are zeroed

    Returns:
        Audio with noise gate applied
    """
    return np.where(np.abs(audio) < threshold, 0, audio)


def bandpass_filter(audio: np.ndarray, low: int = 80, high: int = 3000, sr: int = 16000) -> np.ndarray:
    """
    Apply bandpass filter to isolate speech frequencies.

    Args:
        audio: numpy array of audio samples
        low: Low frequency cutoff (Hz)
        high: High frequency cutoff (Hz)
        sr: Sample rate (Hz)

    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    b, a = butter(5, [low / nyquist, high / nyquist], btype="band")
    return lfilter(b, a, audio)


def preprocess_audio(
    audio: np.ndarray,
    normalize: bool = True,
    trim: bool = True,
    gate: bool = True,
    filter_audio: bool = False,
    sr: int = 16000,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Args:
        audio: Input audio array
        normalize: Apply normalization
        trim: Trim silence
        gate: Apply noise gate
        filter_audio: Apply bandpass filter
        sr: Sample rate

    Returns:
        Preprocessed audio
    """
    result = audio.copy()

    if normalize:
        result = normalize_audio(result)

    if trim:
        result = trim_silence(result)

    if gate:
        result = noise_gate(result)

    if filter_audio:
        result = bandpass_filter(result, sr=sr)

    return result
