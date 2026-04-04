import sounddevice
import numpy as np


def record_audio(duration: int = 5, samplerate: int = 16000) -> np.ndarray:
    recording = sounddevice.rec(
        duration * samplerate,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sounddevice.wait()
    return recording.squeeze()
