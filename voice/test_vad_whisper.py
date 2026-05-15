import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from voice.recorder import record_audio
from voice.transcriber import transcribe


def main() -> None:
    print("=== VAD + Whisper Smoke Test ===")
    print("Speak a short phrase. Recording will stop after silence.")
    audio = record_audio(duration=10, use_vad=True)
    if audio.size == 0:
        print("No speech detected.")
        return
    text = transcribe(audio)
    print(f"Transcript: {text!r}")


if __name__ == "__main__":
    main()
