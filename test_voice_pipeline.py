#!/usr/bin/env python
"""
Test script for the production voice pipeline.

Tests individual components and the full pipeline.
"""

import sys
import numpy as np
from voice.vad import VoiceActivityDetector
from voice.audio_utils import normalize_audio, trim_silence, noise_gate, bandpass_filter
from voice.recorder import record_speech
from voice.transcriber import transcribe


def test_vad():
    """Test Voice Activity Detection."""
    print("\n[TEST] Voice Activity Detection...")
    vad = VoiceActivityDetector()
    print(f"  ✓ VAD initialized (sample_rate={vad.sample_rate}, frame_size={vad.frame_size})")

    # Test with silent frame
    silent_frame = np.zeros(vad.frame_size, dtype=np.float32)
    silent_int16 = (silent_frame * 32768).astype(np.int16)
    is_speech = vad.is_speech(silent_int16.tobytes())
    assert not is_speech, "Silent frame should not be detected as speech"
    print(f"  ✓ Silent frame correctly identified as non-speech")


def test_audio_utils():
    """Test audio preprocessing utilities."""
    print("\n[TEST] Audio Preprocessing Utilities...")

    # Create synthetic audio
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Sine wave at 440Hz
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add noise
    audio += 0.05 * np.random.randn(len(audio))

    # Add silence at start/end
    audio = np.concatenate([np.zeros(sr // 2), audio, np.zeros(sr // 2)])

    print(f"  Created synthetic audio: {len(audio)} samples ({len(audio)/sr:.1f}s)")

    # Test normalize
    norm = normalize_audio(audio)
    assert np.max(np.abs(norm)) <= 1.0
    print(f"  ✓ Normalization works (max: {np.max(np.abs(norm)):.3f})")

    # Test trim silence
    trimmed = trim_silence(norm)
    assert len(trimmed) < len(norm)
    print(f"  ✓ Silence trimming works ({len(norm)} → {len(trimmed)} samples)")

    # Test noise gate
    gated = noise_gate(norm)
    assert np.sum(gated == 0) > np.sum(norm == 0)  # More zeros after gating
    print(f"  ✓ Noise gate works (zeroed {np.sum(gated == 0)} samples)")

    # Test bandpass filter
    filtered = bandpass_filter(norm)
    assert len(filtered) == len(norm)
    print(f"  ✓ Bandpass filter works (shape preserved)")


def test_full_pipeline():
    """Test the full voice-to-text pipeline."""
    print("\n[TEST] Full Voice-to-Text Pipeline...")

    print("  Attempting to record speech...")
    print("  (Speak now or test will use timeout)")

    audio = record_speech()

    if audio is None:
        print("  ⚠ No speech captured (this is OK if using timeout)")
        return

    print(f"  ✓ Speech captured: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    print("  Transcribing...")
    text = transcribe(audio)

    if text:
        print(f"  ✓ Transcription successful: '{text}'")
    else:
        print("  ⚠ Transcription returned empty string (could be silence or noise)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PRODUCTION VOICE PIPELINE TEST SUITE")
    print("=" * 60)

    try:
        test_vad()
        test_audio_utils()
        test_full_pipeline()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
