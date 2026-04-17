"""
Smart audio recorder using Voice Activity Detection.

Records only when speech is detected, handles pre-roll buffering,
and stops automatically on silence.
"""

import sounddevice as sd
import numpy as np
import collections
import time
from voice.vad import VoiceActivityDetector

DEBUG = False  # Set to True to enable debug output


def record_speech(
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    silence_duration_ms: int = 800,
    preroll_duration_ms: int = 300,
) -> np.ndarray | None:
    """
    Record speech using VAD-based detection.

    Does NOT record for a fixed duration. Instead:
    1. Buffers audio before speech starts (pre-roll)
    2. Starts recording when speech is detected
    3. Stops when silence > silence_duration_ms

    Args:
        sample_rate: Audio sample rate (Hz)
        frame_duration_ms: Frame duration for VAD (20ms recommended)
        silence_duration_ms: Silence threshold before stopping (800ms default)
        preroll_duration_ms: Pre-roll buffer duration (300ms default)

    Returns:
        numpy array of recorded audio, or None if no valid speech captured
    """
    vad = VoiceActivityDetector(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        aggressiveness=3,
    )

    frame_size = int(sample_rate * frame_duration_ms / 1000)

    # Pre-roll buffer: store audio before speech starts
    preroll_frames = int(preroll_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=preroll_frames)

    # Track recording state
    triggered = False
    voiced_frames = []
    silence_counter = 0
    silence_limit = int(silence_duration_ms / frame_duration_ms)

    start_time = time.time()

    if DEBUG:
        print(f"[Recorder] Starting VAD detection...")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Frame size: {frame_size} samples ({frame_duration_ms}ms)")
        print(f"  Pre-roll buffer: {preroll_frames} frames ({preroll_duration_ms}ms)")
        print(f"  Silence threshold: {silence_limit} frames ({silence_duration_ms}ms)")

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, silence_counter

        if status:
            print(f"[Recorder] Audio callback status: {status}")

        # Get audio chunk (mono, float32)
        audio_chunk = indata[:, 0].copy()

        # Convert float32 [-1, 1] to int16 for VAD
        audio_int16 = (np.clip(audio_chunk, -1, 1) * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Detect speech
        is_speech = vad.is_speech(audio_bytes)

        if not triggered:
            # Still in pre-roll phase
            ring_buffer.append(audio_chunk)

            if is_speech:
                # Speech detected! Start recording
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()

                if DEBUG:
                    elapsed = time.time() - start_time
                    print(f"[Recorder] Speech detected at {elapsed:.2f}s")

        else:
            # Recording phase
            voiced_frames.append(audio_chunk)

            if is_speech:
                silence_counter = 0
            else:
                silence_counter += 1

            if silence_counter > silence_limit:
                if DEBUG:
                    elapsed = time.time() - start_time
                    print(f"[Recorder] Silence detected at {elapsed:.2f}s, stopping...")
                raise sd.CallbackStop()

    # Record with streaming callback
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(100)

    except sd.CallbackStop:
        pass

    if DEBUG:
        elapsed = time.time() - start_time
        print(f"[Recorder] Recording stopped. Duration: {elapsed:.2f}s")

    # If no speech was captured
    if len(voiced_frames) == 0:
        if DEBUG:
            print("[Recorder] No speech frames captured, returning None")
        return None

    # Concatenate all frames
    audio = np.concatenate(voiced_frames)

    # Reject very short clips (less than 500ms)
    min_duration = sample_rate * 0.5
    if len(audio) < min_duration:
        if DEBUG:
            duration = len(audio) / sample_rate
            print(f"[Recorder] Clip too short ({duration:.2f}s < 0.5s), rejecting")
        return None

    if DEBUG:
        duration = len(audio) / sample_rate
        print(f"[Recorder] Captured {len(audio)} samples ({duration:.2f}s)")

    return audio


def record_speech_with_retries(
    max_retries: int = 3,
    sample_rate: int = 16000,
) -> np.ndarray | None:
    """
    Record speech with automatic retry on failure.

    Args:
        max_retries: Maximum number of recording attempts
        sample_rate: Audio sample rate (Hz)

    Returns:
        Recorded audio, or None if all retries failed
    """
    for attempt in range(1, max_retries + 1):
        if DEBUG:
            print(f"\n[Recorder] Attempt {attempt}/{max_retries}...")

        audio = record_speech(sample_rate=sample_rate)

        if audio is not None:
            if DEBUG:
                print(f"[Recorder] SUCCESS on attempt {attempt}")
            return audio

        if attempt < max_retries and DEBUG:
            print(f"[Recorder] Failed, retrying...")

    if DEBUG:
        print(f"\n[Recorder] All {max_retries} attempts failed")

    return None
