import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="webrtcvad",
)

import webrtcvad
import collections
import numpy as np


class VoiceActivityDetector:
    """
    WebRTC-based Voice Activity Detector for real-time speech detection.
    
    Uses frame-based analysis to classify audio chunks as speech or silence.
    """

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 20, aggressiveness: int = 3):
        """
        Initialize VAD detector.

        Args:
            sample_rate: Audio sample rate in Hz (must be 8000, 16000, or 32000)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            aggressiveness: Aggressiveness level (0-3). Higher = more aggressive filtering.
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame

    def frame_generator(self, audio):
        """
        Split audio array into fixed-size frames for processing.

        Args:
            audio: numpy array of float32 audio samples

        Yields:
            numpy arrays of fixed frame size
        """
        n = self.frame_size
        for i in range(0, len(audio), n):
            frame = audio[i : i + n]
            # Pad last frame if incomplete
            if len(frame) < n:
                frame = np.pad(frame, (0, n - len(frame)), mode="constant")
            yield frame

    def is_speech(self, frame_bytes: bytes) -> bool:
        """
        Detect if a frame contains speech.

        Args:
            frame_bytes: Audio frame as bytes (int16 format)

        Returns:
            True if speech detected, False otherwise
        """
        return self.vad.is_speech(frame_bytes, self.sample_rate)
