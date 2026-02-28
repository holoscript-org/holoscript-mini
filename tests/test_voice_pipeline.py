import numpy as np
from voice.transcriber import transcribe
from voice.command_parser import classify_command


def test_transcriber_silent_audio():
    audio = np.zeros(16000 * 5, dtype="float32")
    text = transcribe(audio)
    assert isinstance(text, str)
    print(f"Transcript: {repr(text)}")


def test_command_parser():
    cases = [
        ("show me the solar system", "NEW_SCENE"),
        ("create a water molecule", "NEW_SCENE"),
        ("what is a neutron star", "NEW_SCENE"),
        ("make it bigger", "REFINE"),
        ("add a ring", "REFINE"),
        ("zoom out", "REFINE"),
        ("something random", "NEW_SCENE"),
    ]
    for text, expected_mode in cases:
        mode, cleaned = classify_command(text)
        assert mode == expected_mode, f"FAIL: '{text}' -> {mode}, expected {expected_mode}"
        assert cleaned == text.strip()
        print(f"OK: '{text}' -> {mode}")


if __name__ == "__main__":
    print("=== test_transcriber_silent_audio ===")
    test_transcriber_silent_audio()
    print("\n=== test_command_parser ===")
    test_command_parser()
    print("\nAll tests passed.")
