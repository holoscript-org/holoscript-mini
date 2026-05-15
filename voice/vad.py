import time
from collections import deque

import numpy as np
import sounddevice
import webrtcvad


class VADSegmenter:
    def __init__(
        self,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        padding_ms: int = 300,
        silence_timeout_ms: int = 800,
        min_speech_ms: int = 300,
        max_speech_ms: int = 8000,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.padding_ms = padding_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

    def listen(self, samplerate: int = 16000, max_duration_s: int = 10) -> np.ndarray:
        frame_size = int(samplerate * self.frame_ms / 1000)
        padding_frames = max(1, int(self.padding_ms / self.frame_ms))
        silence_frames = max(1, int(self.silence_timeout_ms / self.frame_ms))
        max_frames = int(min(max_duration_s * 1000, self.max_speech_ms) / self.frame_ms)

        ring: deque[tuple[np.ndarray, bool]] = deque(maxlen=padding_frames)
        frames: list[np.ndarray] = []
        triggered = False
        silence_count = 0

        start = time.perf_counter()
        with sounddevice.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
        ) as stream:
            while time.perf_counter() - start < max_duration_s:
                audio, _ = stream.read(frame_size)
                if audio is None or audio.size == 0:
                    continue
                mono = audio[:, 0] if audio.ndim > 1 else audio.squeeze()
                pcm16 = (mono * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
                is_speech = self.vad.is_speech(pcm16, samplerate)

                if not triggered:
                    ring.append((mono, is_speech))
                    if sum(1 for _, voiced in ring if voiced) > 0.9 * ring.maxlen:
                        triggered = True
                        frames.extend([frame for frame, _ in ring])
                        ring.clear()
                        silence_count = 0
                else:
                    frames.append(mono)
                    silence_count = 0 if is_speech else silence_count + 1

                    if len(frames) >= max_frames:
                        break
                    if silence_count >= silence_frames and _total_ms(frames, self.frame_ms) >= self.min_speech_ms:
                        break

        if not frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(frames).astype(np.float32)


def _total_ms(frames: list[np.ndarray], frame_ms: int) -> int:
    return len(frames) * frame_ms
