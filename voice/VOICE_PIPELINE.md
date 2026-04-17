# 🎙️ Production Voice Pipeline

A **production-quality voice input system** for capturing clean human speech and generating accurate transcriptions using Whisper.

---

## ✨ Features

### Voice Activity Detection (VAD)
- **Real-time speech detection** using WebRTC VAD
- **No fixed duration recording** — stops automatically on silence detection
- **Frame-based processing** (20ms chunks) for responsive detection
- **Configurable aggressiveness** (0-3 levels)

### Smart Recording
- **Pre-roll buffering** (300ms) captures audio before speech starts
- **Silence detection** (800ms default threshold) — stops when silence exceeds threshold
- **Minimum duration validation** (500ms) — rejects clips that are too short
- **Automatic retries** (up to 3 attempts) on recording failure

### Audio Preprocessing
- **Normalization**: Peak-based normalization to [-1, 1] range
- **Silence trimming**: Remove leading/trailing silence
- **Noise gate**: Zero out sub-threshold samples
- **Optional bandpass filter**: 80-3000 Hz for speech emphasis

### Transcription
- **OpenAI Whisper** (tiny.en model) for accurate speech-to-text
- **Input validation**: Rejects transcriptions < 3 characters
- **Error handling**: Returns empty string gracefully on failure
- **Performance logging**: Tracks inference latency

---

## 📁 Architecture

```
voice/
├── vad.py                    # Voice Activity Detection wrapper
├── recorder.py               # Smart recording with VAD
├── audio_utils.py            # Audio preprocessing pipeline
├── transcriber.py            # Whisper transcription
└── __init__.py
```

### Core Components

#### **VoiceActivityDetector** (`vad.py`)
```python
from voice.vad import VoiceActivityDetector

vad = VoiceActivityDetector(
    sample_rate=16000,
    frame_duration_ms=20,
    aggressiveness=3  # 0=least, 3=most aggressive
)

# Check if audio frame contains speech
is_speech = vad.is_speech(audio_bytes)
```

#### **Smart Recording** (`recorder.py`)
```python
from voice.recorder import record_speech, record_speech_with_retries

# Single attempt
audio = record_speech()

# With retries
audio = record_speech_with_retries(max_retries=3)
```

#### **Audio Utilities** (`audio_utils.py`)
```python
from voice.audio_utils import (
    normalize_audio,
    trim_silence,
    noise_gate,
    bandpass_filter,
    preprocess_audio
)

# Full preprocessing
audio = preprocess_audio(
    audio,
    normalize=True,
    trim=True,
    gate=True,
    filter_audio=False
)
```

#### **Transcription** (`transcriber.py`)
```python
from voice.transcriber import transcribe

text = transcribe(audio, language="en")
```

---

## 🚀 Usage

### Basic Pipeline
```python
from voice.recorder import record_speech_with_retries
from voice.transcriber import transcribe

# Record and transcribe
audio = record_speech_with_retries(max_retries=3)
text = transcribe(audio)

print(f"You said: {text}")
```

### Full Example (`generate_live_scene.py`)
```bash
python generate_live_scene.py
```

The pipeline:
1. **Listen** for speech with VAD (no fixed duration)
2. **Transcribe** with Whisper + preprocessing
3. **Classify intent** (NEW_SCENE or REFINE)
4. **Generate scene** with LLM (GROQ + OLLAMA fallback)
5. **Save JSON** output

---

## 🔧 Configuration

Environment variables in `.env`:

```env
# Voice Recording
VOICE_DEBUG=false              # Enable debug output
VOICE_SILENCE_MS=800           # Silence threshold (ms)
VOICE_PREROLL_MS=300           # Pre-roll buffer (ms)

# Whisper
WHISPER_LANGUAGE=en            # Language for transcription
WHISPER_MIN_LENGTH=3           # Min chars for valid transcription
```

### Debug Mode
Enable detailed logging:
```python
# In recorder.py
DEBUG = True  # Shows speech detection, timing, retry attempts
```

Output:
```
[Recorder] Starting VAD detection...
  Sample rate: 16000 Hz
  Frame size: 320 samples (20ms)
  Pre-roll buffer: 15 frames (300ms)
  Silence threshold: 40 frames (800ms)
[Recorder] Speech detected at 0.42s
[Recorder] Silence detected at 3.21s, stopping...
[Recorder] Recording stopped. Duration: 3.21s
[Recorder] Captured 51360 samples (3.21s)
```

---

## 📊 Technical Details

### VAD Algorithm
- **Method**: WebRTC Voice Activity Detection
- **Frame size**: 320 samples @ 16kHz (20ms)
- **Aggressiveness**: 3 (balanced false negatives vs positives)
- **Latency**: < 1ms per frame

### Recording State Machine
```
┌─────────────────────────┐
│  PRE-ROLL PHASE         │
│  (Buffering only)       │
└────────────┬────────────┘
             │
    Speech detected?
             │
    ┌────────▼────────────┐
    │  RECORDING PHASE    │
    │  (Collecting audio) │
    └────────┬────────────┘
             │
    Silence > 800ms?
             │
    ┌────────▼────────────┐
    │  DONE               │
    │  Validate & return  │
    └─────────────────────┘
```

### Audio Preprocessing Pipeline
```
Input (float32)
    ↓
Normalize ([-1, 1])
    ↓
Trim silence (< 0.01 threshold)
    ↓
Noise gate (< 0.02 amplitude)
    ↓
[Optional: Bandpass filter 80-3000 Hz]
    ↓
Whisper transcription
    ↓
Output (text)
```

---

## 🧪 Testing

Run the test suite:
```bash
python test_voice_pipeline.py
```

Tests:
- Voice Activity Detection
- Audio preprocessing (normalize, trim, gate)
- Full recording pipeline
- Whisper transcription

---

## 📈 Performance

### Typical Metrics
| Metric | Value |
|--------|-------|
| **VAD Latency** | < 1ms per frame |
| **Recording Startup** | < 500ms (first speech) |
| **Whisper Latency** | 2-5s (tiny.en model) |
| **Total P2P Latency** | 3-7s (end-to-end) |
| **Accuracy** | 90%+ (clean speech) |

### Optimization Tips
1. Use **lower aggressiveness** (1-2) if sensitivity is needed
2. **Reduce silence threshold** (400-500ms) for responsive stopping
3. **Skip bandpass filter** for minor speed boost
4. Use **base.en model** instead of tiny.en for higher accuracy (slower)

---

## 🐛 Troubleshooting

### No Speech Captured
- Check microphone access: `sounddevice.query_devices()`
- Increase pre-roll buffer: `preroll_duration_ms=500`
- Lower VAD aggressiveness: `aggressiveness=1`

### Poor Transcription
- Enable bandpass filter: `preprocess_audio(..., filter_audio=True)`
- Lower noise gate threshold: `noise_gate(audio, threshold=0.01)`
- Use larger model: switch to `base.en` or `small.en`

### False Silence Detection
- Increase silence threshold: `silence_duration_ms=1200`
- Increase VAD aggressiveness: `aggressiveness=3`

---

## 📦 Dependencies

```
sounddevice==0.5.5
webrtcvad==2.0.10
scipy==1.13.0
numpy==2.4.2
openai-whisper==20250625
```

---

## 🎯 Key Design Principles

✅ **No Fixed Duration**: Recording adapts to natural speech pauses
✅ **Pre-roll Buffer**: Never misses the start of speech
✅ **VAD-Driven**: Efficient, responsive, low latency
✅ **Graceful Degradation**: Retries and validation prevent bad data
✅ **Production Ready**: Error handling, logging, timeout protection

---

## 📝 License

Part of the holoscript-mini project.
