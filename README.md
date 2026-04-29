# HoloScript Mini

A mixed Python + Next.js hologram rendering system. Generates 3D scenes from voice/LLM input and displays them as WebGL visualizations and cylindrical LED POV frames.

## Quick Start

### GUI (WebGL, no backend needed)
```bash
cd gui
npm install
npm run dev
# Open http://localhost:3000
```

### Backend + Python Renderer
```bash
pip install -r requirements.txt
uvicorn backend.api_server:app --reload --port 8000
python -m renderer.main.render_window
```

### Voice → Scene Pipeline
```bash
python -m voice.generate_live_scene
```

## Docs

| Doc | What it covers |
|-----|---------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System overview, data flow, folder structure |
| [docs/SCENE_SCHEMA.md](docs/SCENE_SCHEMA.md) | Strict scene JSON schema (Three.js renderer contract) |
| [docs/MEMBER1_PIPELINE.md](docs/MEMBER1_PIPELINE.md) | JSON generation pipeline for LLM output |
| [docs/API.md](docs/API.md) | Backend HTTP endpoints reference |

## System at a Glance

```
Voice/LLM ──► core/outputs/scene_grammar.json
                     │
                     ▼
          ┌──────────────────────┐
          │   Scene JSON         │  ← Member 1 generates this
          └──────────────────────┘
                 │          │
                 ▼          ▼
         Three.js GUI    Python OpenGL
         (WebGL view)    (cylindrical POV)
                              │
                              ▼
                       FastAPI backend
                       └► /frame /stream
```

## Key Files

- `backend/api_server.py` — FastAPI server
- `renderer/main/render_window.py` — OpenGL cylindrical POV renderer
- `gui/components/ThreeScene.tsx` — Three.js WebGL scene renderer
- `gui/lib/sceneFactory.ts` — Scene validation + TypeScript types
- `core/state/scene_state.py` — Shared thread-safe state blackboard
- `core/outputs/scene_grammar.json` — LLM output (gitignored, loaded at runtime)
- `gesture/gesture_engine.py` — MediaPipe hand tracking
- `voice/generate_live_scene.py` — Voice → LLM → scene pipeline entry point
