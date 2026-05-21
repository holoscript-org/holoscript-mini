import numpy as np
import sounddevice

from voice.vad import VADSegmenter


def record_audio(
    duration: int = 5,
    samplerate: int = 16000,
    use_vad: bool = True,
    vad_settings: dict | None = None,
) -> np.ndarray:
    if use_vad:
        try:
            return record_audio_vad(
                max_duration=duration,
                samplerate=samplerate,
                vad_settings=vad_settings,
            )
        except Exception:
            pass

    recording = sounddevice.rec(
        duration * samplerate,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sounddevice.wait()
    return recording.squeeze()


def record_audio_vad(
    max_duration: int = 10,
    samplerate: int = 16000,
    vad_settings: dict | None = None,
) -> np.ndarray:
    settings = vad_settings or {}
    segmenter = VADSegmenter(**settings)
    return segmenter.listen(samplerate=samplerate, max_duration_s=max_duration)
