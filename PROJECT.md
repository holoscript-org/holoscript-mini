# HoloScript Mini — Project Documentation

## 1. What This Project Is

HoloScript Mini turns natural language — typed or spoken — into a live, animated 3D scene, and simultaneously computes what that scene would look like painted onto a spinning cylindrical LED display (a physical "volumetric hologram" rig, driven by persistence of vision). You say "show me the solar system" or "add a red pendulum," and within seconds a validated 3D scene graph exists, rendered live in the browser (WebGL) and, in parallel, projected into an `(360° × 18 LEDs)` frame buffer suitable for driving real POV hardware.

A second, independent input channel — webcam hand-gesture tracking — lets you rotate, zoom, freeze, and navigate the same scene without touching a keyboard, by pinching, pointing, or making a fist in front of the camera.

**The core idea**: separate *scene description* (a strict, LLM-generated JSON schema) from *scene consumption* (multiple renderers — Three.js in-browser, Python/OpenGL windowed, cylindrical LED-frame simulation — that all read the same JSON and never need to know how it was produced). Voice, gesture, and text are just different ways of writing to that shared scene state.

### Motto

*Describe it. See it. Touch it.* — collapse the distance between saying what you want and having a manipulable 3D object in front of you, whether on a screen or (eventually) spinning in physical space.

### Who/what each "member" builds (per repo convention)

The codebase is informally split into ownership zones, referenced in code comments and file headers:
- **Member 1** — the generative pipeline (`pipeline/`, `llm/`): transcript → validated scene JSON.
- **Member 2** — the physical/cylindrical renderer (`renderer/`): scene JSON → OpenGL window + LED POV frames.
- A third, less explicitly labeled zone covers the **GUI** (`gui/`), **gesture** (`gesture/`), **voice** (`voice/`), and **backend** (`backend/`) glue.

---

## 2. The Problem This Solves

Building a 3D scene by hand (mesh selection, positioning, materials, lighting, animation, camera framing) is slow and requires domain knowledge most people don't have. Physical volumetric/POV displays additionally require scene data in an unusual format (angular slices × vertical LED columns) that has no natural authoring tool.

HoloScript Mini removes both barriers:
- **Authoring** happens in plain English (typed or spoken), interpreted by an LLM that already knows 3D scene composition, lighting recipes, and basic physics.
- **Output format fan-out** happens automatically: the same generated JSON drives a WebGL preview, a debug OpenGL window, and a cylindrical LED frame buffer, with no format conversion required by the author.
- **Interaction** doesn't require a mouse/keyboard: gesture control offers a touchless way to inspect the result, which matters for a domain (holographic display) where the whole point is not looking at a flat screen.

---

## 3. High-Level Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │                  INPUT LAYER                   │
                    │  Voice (mic + Whisper)   Text (GUI command box)│
                    │  Gesture (webcam + MediaPipe, control-only)    │
                    └───────────────────┬──────────────────────────--┘
                                        │ transcript / command string
                                        ▼
                    ┌────────────────────────────────────────────────┐
                    │    GENERATIVE PIPELINE  (pipeline/, llm/)      │
                    │  prompt optimize → intent extract → semantic  │
                    │  parse → resolve → asset verify → live search │
                    │  → 3-pass LLM architect → critic/fixer loop    │
                    │  → validate → repair → cache                   │
                    │  every stage streams live progress over        │
                    │  WebSocket (backend/api_server.py: /ws/pipeline)│
                    └───────────────────┬──────────────────────────--┘
                                        │ validated scene JSON
                                        │ (gui/lib/sceneFactory.ts is the
                                        │  canonical schema; scene_validator.py
                                        │  is its Python mirror)
                    ┌───────────────────┴──────────────────────────--┐
                    │                                                │
                    ▼                                                ▼
        ┌──────────────────────┐                     ┌──────────────────────────┐
        │   Next.js GUI         │                     │  Python Renderer          │
        │   (Three.js WebGL,    │                     │  (OpenGL/pyglet window,   │
        │   fully client-side   │                     │   cylindrical POV frame   │
        │   rendering + gesture,│                     │   builder, standalone     │
        │   + live pipeline     │                     │   OS process)             │
        │   overlay UI)         │                     │                           │
        └───────────┬───────────┘                     └────────────┬─────────────┘
                    │ HTTP + WebSocket (only for the                │ file-based IPC
                    │ generative pipeline)                          │ (.runtime/*.json, *.npy)
                    ▼                                                ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                     FastAPI backend (backend/)                    │
        │  thin façade: triggers pipeline, streams progress over WebSocket, │
        │  proxies renderer state/frames, forwards GUI scene pushes +       │
        │  control actions                                                  │
        └────────────────────────────────────────────────────────────────--┘
```

**Key architectural fact**: the GUI and the Python renderer are two *independent* consumers of the same scene schema, not a pipeline where one feeds the other. The GUI does its own WebGL rendering and its own client-side cylindrical-POV *simulation* (`gui/hooks/webglPov.ts`); the Python renderer does real OpenGL rendering and the real cylindrical POV computation intended for physical hardware. They can run entirely independently — the GUI does not need the Python backend running except to trigger the LLM pipeline.

The backend and the Python renderer are **separate OS processes** (started independently: `uvicorn backend.api_server:app` and `python -m renderer.main.render_window`) that never share memory. They communicate exclusively through atomically-written files under `.runtime/` — there is no socket/RPC layer between them.

---

## 4. Features

- **Natural-language scene generation** — typed command or live microphone input becomes a full 3D scene (objects, materials, lights, camera, animation), with automatic asset lookup/download for named real-world objects.
- **Multi-pass generation pipeline** — the raw request is first clarified by an LLM prompt optimizer, then structured into a Scene Intent IR (objects, spatial relationships, dynamics, mood) by a dedicated extraction stage, before the scene architect builds it across three focused LLM passes (layout → per-object detail → lighting/camera) rather than one giant call — narrower, more supervisable steps at each stage (§5.2).
- **Iterative self-critique** — a critic/fixer loop reviews the generated scene for object-level technical issues (spatial/scale/lighting/animation/physics/camera) and fixes what it finds, re-checking up to 3 times until clean (§5.2 Stage 8).
- **Holistic intent verification with targeted re-generation** — after the technical critique, a separate agent compares the finished scene against the user's original request as a whole and, if it falls short (wrong object count, missing described relationships, wrong mood), identifies which specific generation stage is responsible and re-runs *only that stage* with corrective feedback — modifying and realigning the existing scene rather than discarding it or falling back to a generic placeholder (§5.2 Stage 8.5).
- **Live pipeline visualization** — every generation stage streams real-time progress over WebSocket to a full-screen overlay in the GUI, rendering each stage's actual output in a purpose-built visual form (before/after text diffs, animated concept chips, a live top-down placement preview, issue cards, a validation checklist) rather than a raw JSON dump (§11.6).
- **Physics-aware animation** — not just spin/orbit, but closed-form/integrated gravity, simple-harmonic-motion, pendulum, and projectile motion, with numeric parameters generated by the LLM and range-clamped for physical plausibility.
- **Educational framing** — every generated scene includes a 2–4 sentence `summary` and per-object `description` explaining the underlying concept (this is explicitly a teaching tool, not just a renderer).
- **Self-healing validation** — a generated scene that's slightly malformed (bad color format, missing required field, dangling mesh reference) is *repaired*, not rejected: structural fixers run before any hard failure, and a last-resort fallback scene guarantees the pipeline always produces something renderable.
- **Live 3D preview in-browser** — Three.js/React-Three-Fiber scene rendered client-side, with orbit camera controls, PBR materials, texture loading, GLB/OBJ mesh support, and a scene-graph parent/child hierarchy.
- **Cylindrical LED POV simulation** — both a real Python/NumPy implementation (for driving physical hardware) and a client-side JS approximation (for browser preview) that project the 3D scene into a `360 × 18` angle/height LED frame buffer.
- **Touchless gesture control** — one-hand webcam tracking recognizes 5 gestures (pinch, open palm, fist, point, V-sign) mapped to rotate / zoom / freeze / navigate / reset, with hysteresis, confirmation-window debouncing, and a graceful HSV-color-based fallback when the ML hand-tracking model is unavailable.
- **Voice input with VAD** — microphone capture auto-segments speech using WebRTC voice-activity detection (not fixed-duration recording), transcribed locally via Whisper (no cloud STT dependency).
- **Asset economy** — a MongoDB-backed concept knowledge base + Redis cache + on-demand Poly Pizza API search/download means the system can acquire new 3D meshes at runtime for concepts it doesn't already have, rather than being limited to a fixed asset library.
- **Undo-capable scene history** — the shared state blackboard keeps the last 20 scene versions.
- **Cross-process live control** — scene pushes and keyboard-equivalent control actions (rotate, zoom, explode, freeze, reset) can be sent from the GUI/backend to a separately-running Python renderer window via file-based IPC, with no shared process required.

---

## 5. The Generative Pipeline (`pipeline/`, `llm/`, `core/`)

This is the heart of the system — "Member 1's pipeline." It converts a transcript into a scene JSON that exactly matches the schema defined in [gui/lib/sceneFactory.ts](gui/lib/sceneFactory.ts) (the single source of truth; [pipeline/scene_validator.py](pipeline/scene_validator.py) is a hand-maintained Python mirror of it, cross-referenced by line number in its own docstring).

### 5.1 Scene JSON schema (canonical)

```ts
SceneDef {
  name?: string
  summary?: string                 // 2-4 sentence educational blurb
  objects: SceneObject[]
  lights: LightDef[]
  camera: CameraDef
}

SceneObject {
  id: string
  type: "primitive" | "mesh"
  parent?: string                  // scene-graph nesting
  geometry?: GeometryDef           // required if type === "primitive"
  model?: string                   // required if type === "mesh" (.glb/.gltf/.obj)
  position: [x, y, z]
  rotation?: [x, y, z]             // Euler, degrees
  scale?: [x, y, z]
  material: MaterialDef            // PBR: color, roughness, metalness, maps, emissive
  label?: string
  description?: string             // educational blurb for this object
  animation?: AnimationDef
}

GeometryDef.type: "sphere"|"box"|"cylinder"|"plane"|"ring"|"capsule"|"torus"

AnimationDef (discriminated union on `type`):
  { type: "none" }
  { type: "orbit", center?, center_ref?, axis?, speed?, phase? }
  { type: "spin",  axis?, speed? }
  { type: "physics", physics_type: "gravity"|"shm"|"pendulum"|"projectile", ... }

LightDef.type: "ambient"|"directional"|"point"|"spot"
CameraDef { position: [x,y,z], target: [x,y,z], fov? }
```

Validation is **fail-soft**: a malformed individual object is dropped with a logged error rather than invalidating the whole scene. Physics parameters are *clamped* into physically-sane ranges rather than rejected (e.g. gravity `g` clamped to `[0.1, 30.0]`).

### 5.2 Stage-by-stage pipeline (`pipeline/pipeline_runner.py: run_pipeline()`)

Every stage below can optionally emit progress events (`pipeline/events.py`'s `PipelineEvent`) via an `on_event` callback threaded through `run_pipeline(transcript, on_event=None, run_id=None)`. Both parameters default to `None`/auto-generated and are fully backward compatible — CLI usage and any caller that doesn't pass `on_event` behaves exactly as before, with zero event emission overhead. `backend/api_server.py`'s `/ws/pipeline` route (§10.3) is what actually consumes these events to drive the frontend's live pipeline overlay (§11.6).

```
transcript
  │
  ├─ [Redis scene-cache check — skipped for refinement commands like "add a red cube",
  │   keyed on the ORIGINAL transcript, not the Stage 2 optimizer's output — see §5.3]
  │
  ▼
Stage 1  Receive transcript
  │
Stage 2  Prompt Optimizer (LLM)            pipeline/prompt_optimizer.py
         Gemini 2.5 Flash (primary) / Groq fallback. Rewrites a possibly
         fragmentary/ASR-noisy transcript into one clear scene description
         without inventing content; refinement-style short commands pass
         through nearly verbatim. Fails open (returns transcript unchanged)
         on any error — never blocks the pipeline.
  │
Stage 3  Structured Intent Extraction (LLM) pipeline/intent_extractor.py
         Gemini 2.5 Pro (primary) / Groq fallback. Produces a Scene Intent
         IR — objects with counts/roles, spatial relationships, dynamics,
         mood/style, educational focus, explicit constraints — as advisory
         context for the architect. Never validated, never written to
         live_scene.json; a transient reasoning artifact only. Fails open
         with an all-empty IR on error.
  │
Stage 4  Semantic parse                    pipeline/semantic_parser.py
         sentence-transformer (all-MiniLM-L6-v2) embeddings + greedy phrase
         matching against a MongoDB concept corpus → {objects, structures,
         systems, effects, unresolved_tokens}. Runs on the OPTIMIZED prompt.
         Unchanged by this rebuild — see §5.3 for why intent extraction is a
         separate stage rather than a repurposing of this parser.
  │
Stage 5  Resolve intent                    pipeline/fallback_engine.py
         Mongo direct lookup → synonym lookup → compound-token lookup →
         embedding similarity search, in that priority order
  │
Stage 6  Verify assets on disk             pipeline/asset_registry.py
         only concepts whose GLB mesh file actually exists survive;
         disk-scan fallback re-registers found files back into Mongo
  │
Stage 6b Live asset search (conditional)   pipeline/live_search.py
         Poly Pizza API search+download for stale/unresolved concepts;
         invalidates the scene cache if new assets were fetched
  │
Stage 7  Scene Architect — 3 LLM passes    pipeline/scene_architect.py
  │   7a. Layout & Composition (_architect_layout, Gemini Pro/Groq)
  │       decides object ids, primitive-vs-mesh type, geometry family,
  │       coarse position, parent/child skeleton, labels — count/identity/
  │       spatial-skeleton decisions made in isolation, undiluted by
  │       simultaneously reasoning about colors/physics constants
  │   7b. Per-Object Detail (_architect_detail, Gemini Pro/Groq)
  │       fills in full geometry params, materials, animation (incl.
  │       physics), educational label/description text — explicitly
  │       forbidden from adding/removing/repositioning objects wholesale
  │   7c. Lighting/Camera/Polish (_architect_finish, Gemini Flash/Groq)
  │       decides lights, camera framing, scene name/summary — a bounded,
  │       mechanical decision given the complete object list, so a
  │       cheaper model suffices
  │   If 7b/7c fail individually, _default_lighting_camera() provides a
  │   deterministic (no-LLM) safety net rather than aborting the scene.
  │   └─ 7d (only if 7a itself produced no objects): legacy fallback path —
  │        pipeline/retrieval.py → scene_builder.py → scene_enhancer.py
  │        (deterministic placement + Groq llama-3.1-8b-instant refinement)
  │        — unchanged failure semantics from before this rebuild
  │
Stage 8  Critic ↔ Fixer loop, up to 3 iters pipeline/critic_agent.py
         RE-ENABLED as a genuine iterative loop (critique_and_fix_loop):
         Gemini 2.5 Flash reviews for intent/spatial/scale/lighting/
         animation/physics/camera issues → fixer applies only the listed
         fixes → re-critique → repeat, short-circuiting immediately once a
         pass finds no issues, capped at 3 iterations (matches
         repair_loop.py's own max_iterations=3 convention). Each iteration
         is a separately-numbered pipeline event ("critic_iteration_1",
         "critic_iteration_2", ...) for live progress visibility.
         OBJECT-LEVEL technical review only — see Stage 8.5 for the
         holistic "does this satisfy the request" check.
  │
Stage 8.5 Intent Verifier ↔ Realign loop, up to 2 rounds  pipeline/intent_verifier.py
         NEW. Where Stage 8 checks individual object defects, this stage
         asks a holistic question: does the FINISHED scene, taken as a
         whole, actually satisfy the user's ORIGINAL request and the
         Scene Intent IR (Stage 3)? (e.g. "all 8 planets" but only 3
         exist; "calm underwater mood" but lighting is harsh and white).
         Gemini 2.5 Flash compares scene vs. request → if it falls short,
         identifies which ONE of the three architect passes (layout /
         detail / finish) is responsible → calls
         scene_architect.regenerate_pass() to re-run ONLY that pass with
         the reviewer's feedback injected into its prompt → merges the
         result back into the scene → re-verifies. Short-circuits as soon
         as a round is satisfied, capped at 2 rounds. Critically, this
         NEVER falls back to a generic/primitive placeholder scene on a
         mismatch — it modifies and realigns the existing scene toward
         the user's intent. Fails open (keeps the pre-round scene) if a
         round's regeneration produces nothing usable.
  │
Stage 9  Validate                          pipeline/scene_validator.py
         schema conformance check (Python mirror of sceneFactory.ts)
  │
Stage 10 Repair                            pipeline/repair_loop.py
         non-LLM structural auto-fix, up to 3 iterations, triggered by
         string-matching the validator's error messages
  │
  ├─ [Redis scene-cache write — non-refinement commands only]
  │
Stage 11 Fallback if still empty, strip nulls, save
         pipeline/repair_loop.py: DEMO_FALLBACK (a single spinning yellow
         sphere) if the scene still has no objects; strip_none() removes
         all null values; write to core/outputs/live_scene.json
```

### 5.3 Supporting subsystems

- **Shared LLM client** (`llm/gemini_client.py`) — single Vertex AI (Gemini, ADC auth, no API key) + Groq REST client used by every LLM-calling stage above (`prompt_optimizer.py`, `intent_extractor.py`, `scene_architect.py`, `critic_agent.py`). Exposes `call_gemini()`, `call_groq()`, and `call_llm()` (try-Gemini-then-Groq, returns which provider actually answered — surfaced as pipeline-event metadata so the frontend can badge "Gemini 2.5 Pro" vs "Groq (fallback)"). This replaces two previously-duplicated Vertex AI client constructors (`scene_architect._make_vertex_client` and `critic_agent._get_gemini_client`) that were identical except for a try/except wrapper.
- **Why intent extraction is a new LLM stage, not a repurposed `semantic_parser.py`**: the semantic parser is an *asset-recall* tool — it embeds the transcript and scores it against a MongoDB corpus of known concept phrases to find candidate mesh/primitive matches. It has no ability to reason about spatial relationships ("the moon orbits the earth"), counts ("eight planets"), dynamics ("bouncing"), or mood ("dark and vast") — those are language-understanding tasks needing a real reasoning pass, not corpus similarity search. `pipeline/intent_extractor.py` runs entirely on the optimized prompt and doesn't depend on Mongo/asset lookups, so it's placed before semantic parsing in execution order even though asset resolution conceptually follows "what does the user want" in the pipeline's narrative.
- **Progress events** (`pipeline/events.py`) — defines `PipelineEvent` (run_id, stage, status, label, payload, timestamp, elapsed_ms, provider) and `make_emitter()`, a no-op-when-`on_event`-is-`None` closure factory. Zero knowledge of WebSockets/asyncio/FastAPI here — purely a transport-agnostic contract; the WebSocket bridge lives entirely in `backend/api_server.py` (§10.3). **MIRROR STATUS**: this dataclass's shape is mirrored in TypeScript at `gui/lib/pipelineTypes.ts` — same manual-sync obligation and risk as the `scene_validator.py`/`sceneFactory.ts` schema mirror; any field change must be made in both places.
- **Knowledge base** (`pipeline/knowledge_base/`) — MongoDB is the source of truth for "concept → asset" mappings (`mongo_client.py`); `embedder.py` uses the same sentence-transformer model as the semantic parser to do brute-force cosine-similarity search over concept descriptions (no vector index — fine at this corpus scale).
- **Caching** (`pipeline/cache.py`) — Redis, two namespaces: resolved asset paths (7-day TTL) and full generated scenes keyed by `md5(transcript)` (6-hour TTL), keyed on the **original** transcript, not the Stage 2 optimizer's output — the optimizer can vary slightly run-to-run at temperature > 0, which would fragment the cache if keyed on its output instead. Entirely optional — Redis being down silently degrades to "no cache," never breaks the pipeline.
- **Asset acquisition** — `pipeline/live_search.py` does on-demand Poly Pizza search/download at pipeline runtime; `pipeline/asset_ingester.py` is a separate one-time bulk-seeding script (Poly Pizza + Kenney asset packs + Khronos glTF samples + Poly Haven HDRIs), not part of the live request path.
- **`llm/groq_client.py`** — a smaller, independent Groq wrapper (model `llama-3.1-8b-instant`) used only by the *legacy* fallback path (`scene_enhancer.py`, `llm_bridge.py`), plus a `ContextManager` for bounded conversation history. Its own `generate_scene_groq()`/`generate_scene()` full-scene-generation functions appear to be **dead code**, superseded by `scene_architect.py`. Not touched by this rebuild — it belongs to the untouched legacy fallback path; a natural future candidate to fold into `llm/gemini_client.py`, flagged but not actioned here.
- **Critic (Stage 8) vs. Intent Verifier (Stage 8.5) — two different questions, deliberately not merged into one pass**: the critic (`pipeline/critic_agent.py`) checks the scene against a fixed checklist of *technical* defects (missing lights, invalid physics params, overlapping positions, a spin axis of `[0,0,0]`) and patches exactly the flagged fields. The intent verifier (`pipeline/intent_verifier.py`) instead re-reads the user's original request and asks a subjective, holistic question a checklist can't capture — "does this scene, as a whole, give the user what they asked for?" — and if not, decides which entire *architect pass* (not which field) is responsible and re-runs it with feedback via `scene_architect.regenerate_pass()`. Running the verifier after the critic means the critic has already cleaned up mechanical defects before the (more expensive, whole-scene) intent check runs, so the verifier's feedback is about composition/mood, not things the critic would have caught anyway.

### 5.4 Known current-state caveats (worth knowing before relying on this doc)

**Resolved as part of the pipeline rebuild** (see §5.2/§5.3/§10.3 above for detail):
- The critic/fixer quality pass is now a genuine re-enabled iterative loop (Stage 8), not disabled.
- Added a holistic intent-verification loop (Stage 8.5, `pipeline/intent_verifier.py`) on top of the critic: the critic alone only checked object-level technical defects and had no notion of "does this match what the user actually asked for" as a whole. The verifier compares the finished scene against the original request, and on a mismatch, targets and regenerates the specific architect pass responsible — it does not fall back to a generic/primitive scene the way an early version of the critic's mesh-sanitization safety net might suggest (that safety net, `_sanitize_mesh_paths`, is a narrower, unrelated mechanism — it only replaces an individual object whose mesh path isn't verified on disk, never the whole scene).
- The `GROQ_API_KEY` hard-gate bug (`scene_architect.py`'s old `generate_scene()` returned `None` immediately if `GROQ_API_KEY` was unset, regardless of whether Gemini/Vertex was configured) is fixed — the primary/fallback try-order is the only gate now.
- The two duplicated Vertex AI client constructors are consolidated into `llm/gemini_client.py`.
- `requirements.txt` now lists `google-genai`, `groq`, `openai`, `websockets`, `redis` — all were installed and imported by code but previously unlisted (`redis` in particular is a hard top-level import in `pipeline/cache.py`, not lazy/optional, even though the *server* being unreachable degrades gracefully).
- Stale module docstrings in `scene_architect.py`/`critic_agent.py` claiming `GEMINI_API_KEY (direct API)` were corrected — the actual code always used Vertex AI + ADC, no API key.

**Deliberately deferred, not fixed** (flagged so they aren't rediscovered and re-litigated):
- `pipeline/cache.py`'s `invalidate_all_scenes()` uses blocking `KEYS scene:*` (O(n) on Redis) — a `SCAN`-based rewrite would be better, but explicitly out of scope for this rebuild.
- `llm/groq_client.py`'s redundancy with `llm/gemini_client.py` — not merged, since it belongs to the untouched legacy fallback path.
- No persistence of `PipelineEvent` history beyond the live WebSocket stream (e.g. for later replay/analysis) — a natural extension point, not built.

**Still open from before this rebuild**:
- The README's Quick Start references `voice/generate_live_scene.py`, which **does not exist**. The real voice entry point is `pipeline/pipeline_runner.py`'s `run_with_voice()` (invoke via `python -m pipeline.pipeline_runner` with no args).
- `core/assets/examples/SCHEMA_GAPS.md` is a historical planning document; nearly all the gaps it lists have since been closed in the current schema — don't treat it as current.

---

## 6. Shared State & Cross-Process Communication (`core/`)

Everything in this project that isn't the pipeline itself hangs off one shared contract: **`core/state/scene_state.py`**.

### 6.1 `SceneState` — the in-process blackboard

A thread-safe (single `RLock`-guarded) singleton with an explicit field-ownership contract:

| Field | Owner (writer) | Notes |
|---|---|---|
| `scene_json`, `transcript`, `scene_history` | Voice/LLM thread | setting `scene_json` auto-pushes the previous value onto a 20-entry undo deque and bumps `scene_version` |
| `rotation_y`, `scale`, `explode`, `frozen`, `current_gesture`, `nav_event`, `nav_phase`, `preview_candidate`, `focused_object`, `hand_present` | Gesture thread | `scale` clamped `[0.3, 4.0]`, `explode` clamped `[0,1]` — redundantly enforced here even though the gesture engine already clamps, "so no code path can breach the contract" |
| `current_frame` | Renderer thread | strictly validated: must be `numpy.ndarray`, shape exactly `(360, 18, 3)`, dtype `uint8` — this is the authoritative definition of "360 angular positions × 18 LEDs" |
| `logs` | any thread | 100-entry ring buffer, UTC-timestamped |

All reads/writes go through property getters/setters; `get_render_params()` does one atomic lock acquisition to return `(rotation_y, scale, explode, frozen)` together, avoiding torn reads.

### 6.2 `core/state/ipc_store.py` — the cross-process bridge

Because the FastAPI backend and the Python OpenGL renderer are separate OS processes, `SceneState` alone can't connect them. `ipc_store.py` provides atomic file-based pub/sub under `.runtime/` (write to a temp file, then `os.replace` — readers never see a partial write):

| File | Writer | Reader | Content |
|---|---|---|---|
| `scene_snapshot.json` | renderer, ~0.35s cadence | backend | scene JSON + logs + status dict |
| `current_frame.npy` | renderer | backend | the live `(360,18,3)` uint8 POV frame |
| `scene_command.json` | backend (`POST /scene`, `POST /command`) | renderer, polled every render frame | `{id, scene}` — new scene to apply |
| `control_command.json` | backend (`POST /control`) | renderer, polled every render frame | `{id, action}` — keyboard-equivalent action |

Commands carry a monotonically increasing `id` (`time.time_ns()`) so each side only applies a given command once.

---

## 7. Voice Input (`voice/`)

Supplies capture + transcription primitives; the actual voice→scene orchestration lives in `pipeline/pipeline_runner.py::run_with_voice()`, not inside `voice/` itself.

```
mic audio
  → voice/vad.py: VADSegmenter.listen()
      WebRTC VAD (webrtcvad), triggered ring-buffer state machine:
      ~300ms pre-speech padding preserved, stops after 800ms silence
      (min 300ms / max 8s of captured speech)
  → voice/recorder.py: record_audio()
      wraps VAD capture; ANY failure silently falls back to a fixed-
      duration sounddevice.rec() so this call can never crash
  → voice/transcriber.py: transcribe()
      OpenAI Whisper, local inference, "tiny.en" model (loaded once at
      import time), fp16=False for CPU compatibility
  → interactive CLI confirmation ("Is this correct? y/n")
  → pipeline.pipeline_runner.run_pipeline(transcript)   [see §5.2]
```

STT is fully local — no cloud speech API dependency.

---

## 8. Gesture Control (`gesture/`)

Webcam-based, single-hand, control-only (it drives `rotation_y`/`scale`/`frozen`/navigation on an *existing* scene — it does not generate scenes). Two implementations exist side by side:

- **`gesture/classification/gesture_classifier.py` + `demo_live_gestures.py`** — a simple, standalone reference/demo implementation (fixed thresholds, no smoothing) used only for visual demonstration, not wired to `scene_state`.
- **`gesture/gesture_engine.py`** (the production implementation, 1359 lines) — a documented 9-stage pipeline, run in its own background daemon thread:

```
Stage 0  Frame timing        — real dt via time.monotonic(), not a fixed 33ms tick
Stage 1  MediaPipe extraction — 21 hand landmarks (HandLandmarker Tasks API)
Stage 2  EMA smoothing        — dt-normalized alpha per landmark
Stage 3  Palm normalization   — wrist-relative, palm-size scale invariant
Stage 4  Finger states        — hysteresis-banded open/closed booleans + pinch distance
Stage 5  Gesture classification — strict priority tree:
                                  PINCH → V_SIGN → POINT → OPEN_PALM → FIST → UNKNOWN
Stage 6  Confirmation window  — per-gesture debounce (80-300ms) before a
                                 candidate gesture becomes "active"
Stage 7  Velocity             — 5-point palm-center average, EMA-smoothed,
                                 dead-zone gated
Stage 8  Modal dispatch       — active gesture selects a MODE, hand velocity
                                 drives that mode's action
Stage 9  SceneState update    — written every frame
```

**Gesture → Mode mapping**: `PINCH → ROTATE`, `OPEN_PALM → ZOOM`, `FIST → FREEZE`, `V_SIGN → RESET`, `POINT → navigate/preview/focus (PSF sub-system)`.

The **POINT** mode drives a more elaborate sub-state-machine: directional-lock continuous navigation (IDLE → COLLECTING → NAVIGATING, locks onto one of RIGHT/LEFT/UP/DOWN once movement is unambiguous) that, on going idle, hands off to a **PREVIEW → FOCUS** layer — averaging the last 6 frames of the wrist→index-tip pointing ray into a candidate, and "snapping" to FOCUS if the hand holds still for 300ms. Concrete object hit-testing against that ray is left to the renderer (deliberate separation of concerns); FOCUS-mode gesture routing to the focused object is a documented but unimplemented extension point.

If the MediaPipe Tasks model or library is unavailable, the engine **falls back to pure OpenCV**: HSV skin-color masking + MOG2 background subtraction to find the largest hand-like contour, generates 21 mock landmarks arranged around its bounding box, and feeds them through the *same* Stages 3–8 — full functionality is preserved at reduced confidence, not lost.

`gesture/main.py` is a standalone OpenCV-drawn demo ("SimpleRenderer") that visualizes `SceneState` reacting to gesture input as a colored square — useful as a debugging harness and as the reference for how the real renderer should consume the same state.

---

## 9. Python Renderer (`renderer/`)

The "Member 2" component. A standalone process (`python -m renderer.main.render_window`) that does real OpenGL rendering *and* computes the actual cylindrical LED POV frame intended for physical hardware. Runs on legacy/fixed-function OpenGL (GL 2.1 compatibility context, `pyglet` for windowing only — all drawing is raw `PyOpenGL` immediate-mode).

### 9.1 Load → animate → draw → capture pipeline

```
Scene JSON (file, or live via IPC command, or SceneState.scene_json)
  → renderer/loader/scene_loader.py + renderer/scene_parser.py
      parse_scene(): JSON → list[SceneObject] dataclasses
      never raises — malformed objects are skipped with a warning
  ↓  (every render frame, in Renderer.on_draw)
  1. Poll IPC for new scene/control commands, apply if present
  2. Read (rotation_y, scale, explode, frozen) atomically from SceneState
  3. engine/animation.py Animator.update() — advances orbiting objects'
     world_position (angle = orbit_speed * sim_time * 0.5)
  4. Clear frame, rebuild camera matrices, apply global scale/rotation
     (engine/transforms.py TransformApplier)
  5. Per object: temporary explode offset → engine/primitives.py dispatch()
     draws the right primitive (sphere/cube/cylinder/ring/mesh) with
     type-specific surface styles (banded, earth, polar_caps,
     saturn_rings, emissive_glow) → restore true position
  6. 2D label/billboard overlay pass (gluProject + orthographic text)
  ↓  (throttled, only if scene has orbiting objects or just changed)
  7a. FrameExtractor.extract() — glReadPixels → downscaled JPEG →
      .runtime/render_preview.jpg  (this is the "3D preview" image;
      served by backend's /render-preview and /stream MJPEG endpoints)
  7b. engine/cylindrical/frame_builder.py build_frame_from_scene() —
      computed DIRECTLY from SceneObject geometry (NOT from rendered
      pixels): samples each object's surface in 3D, projects every
      sample point to (angle 0-359, LED 0-17) via engine/cylindrical/
      projector.py, composites with brightest-wins blending →
      (360, 18, 3) uint8 array → SceneState.current_frame +
      .runtime/current_frame.npy  (served by backend's /frame endpoint;
      this is the actual physical-hardware-facing format)
```

The raw-framebuffer JPEG preview and the cylindrical LED frame are **two independent outputs computed in parallel from the same scene state**, not a pipeline where one derives from the other.

### 9.2 The cylindrical projection, concretely

- `cartesian_to_angle_idx(x, z)` → `atan2(z, x)` normalized to one of 360 integer angular slots.
- `y_to_led_idx(y, y_min, y_max)` → world height linearly mapped to a *fractional* position in `[0, 17]` (18 LEDs), so a point between two LEDs contributes proportionally to both (vertical anti-aliasing).
- Only object *surfaces* are sampled (`sample_sphere_surface`/`sample_cube_surface`, sample count scales with object size, clamped `[50, 800]`), mimicking that a physical hologram shows only the visible boundary of a volume, not its interior.
- Cylinders and rings are approximated via sphere sampling (a documented simplification).
- Compositing is "brightest wins" (perceptual luminance comparison) — a cheap approximation of occlusion without real ray casting or z-buffering.

### 9.3 Constants worth knowing

| Constant | Value |
|---|---|
| POV frame shape | `(360, 18, 3)` uint8 — enforced by `SceneState.current_frame` setter |
| Orbit speed scale | `0.5` (visual, not physically literal) |
| Explode push distance | `8.0` units at `explode=1.0` |
| Scale clamp | `[0.3, 4.0]` |
| Default camera | fov 65°, eye `[13.5, 28, 65]`, target `[13.5, 0, 0]` |
| Perf budgets (per `README_INTEGRATION.md`) | frame extraction < 10ms, cylindrical build < 50ms, scene rebuild < 100ms, render ≥ 25fps |

Bundled demo scenes: `renderer/assets/solar_system.json` (sun + 8 planets, realistic relative orbital speeds, per-planet surface styles) and `renderer/assets/human_heart.json` (25-object anatomical model built entirely from spheres/cylinders/rings, demonstrating the renderer isn't astronomy-specific).

---

## 10. Backend (`backend/api_server.py`)

A thin FastAPI façade — it does not render anything itself. Docstring: "FastAPI bridge between the Python renderer (SceneState) and the React GUI."

### 10.1 Endpoint table

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/frame` | Cylindrical POV frame as nested lists — IPC snapshot first, in-process `SceneState` fallback |
| `GET` / `POST` | `/scene` | Read or push scene JSON. `POST` updates both `SceneState` and the cross-process `scene_command.json` |
| `GET` | `/logs` | Recent log lines |
| `GET` | `/status` | `{rotation_y, scale, explode, frozen, gesture, ts}` |
| `GET` | `/render-preview` | Raw OpenGL framebuffer JPEG (falls back to a static test PNG) |
| `GET` | `/stream` | MJPEG multipart stream of the same preview, ~30fps poll |
| `POST` | `/control` | Keyboard-equivalent control action (`space/left/right/up/down/e/r/j/k/h`) — "mirrors exactly what on_key_press does" in the renderer, forwarded via IPC |
| `POST` | `/command` | **The generative-pipeline trigger (legacy/back-compat path).** Runs `pipeline.pipeline_runner.run_pipeline()` in a background daemon thread; returns immediately with `"processing"`. Left byte-for-byte unchanged by the pipeline-visualization rebuild — still used by any non-browser caller (voice pipeline, scripts). |
| `GET` | `/command/status` | Poll target for the above — `{running, state, message, run_id}` (`run_id` added, purely additive) |
| `WS` | `/ws/pipeline` | **The live pipeline trigger + progress stream**, used by the GUI's pipeline overlay (§11.6). See §10.3. |

### 10.2 Threading model

- **Startup**: a `"preload-semantic"` daemon thread eagerly loads the sentence-transformer model + Mongo concept corpus (via FastAPI's `lifespan` context manager), so the first real request doesn't pay that cold-start cost.
- **Per `POST /command`**: a `"pipeline"` daemon thread runs the full generation pipeline; a plain `threading.Lock` ensures only one pipeline run happens at a time (a concurrent request gets `{"status": "busy"}` immediately rather than queueing). **Per `/ws/pipeline` command**: the same single-flight lock is reused — a `"pipeline-ws"` daemon thread runs the pipeline, and a second command while one is in flight gets a `run_finished`/`error` message rather than queueing (§10.3) — no new concurrency model, WS is a pure observer/trigger layered on the existing lock.
- No asyncio background tasks are used for the HTTP routes — concurrency is plain OS threads + explicit locks throughout, consistent with the rest of the codebase (gesture engine also uses a daemon thread). The WebSocket route is the one place asyncio is load-bearing (see §10.3's thread↔event-loop hand-off).
- CORS is fully open (`allow_origins=["*"]`) — this is a development configuration, not hardened for production exposure.

### 10.3 `/ws/pipeline` — live pipeline progress stream

Additive alongside `POST /command`/`GET /command/status`, which are left completely untouched for back-compat. This route is the new trigger + live progress path the GUI's pipeline overlay (§11.6) uses.

**Protocol**: client connects, sends `{"command": "..."}`. Server checks the same `_pipeline_lock`/`_pipeline_state` used by `/command`; if busy, sends one `run_finished`/`error` message and keeps the socket open (a transient "try again" condition, not a reason to force a reconnect). If free, generates a `run_id` (uuid4), sends `run_started`, and spawns the same kind of daemon thread `/command` does — but now passing an `on_event` callback into `run_pipeline()`. On completion, sends `run_finished` with the final scene JSON inlined on success (so the GUI can hydrate its 3D view without an extra `GET /scene` round trip). The connection is **not** closed after `run_finished` — it can be reused for a subsequent command in the same page session.

Wire message schema (mirrors `pipeline/events.py`'s `PipelineEvent` — see §5.3's MIRROR STATUS note):
```jsonc
{ "type": "run_started", "run_id": "...", "transcript": "..." }
{ "type": "pipeline_event", "run_id", "stage", "status": "started"|"output"|"completed"|"failed",
  "label", "payload", "provider": "gemini"|"groq"|null, "timestamp", "elapsed_ms" }
{ "type": "run_finished", "run_id", "status": "done"|"error", "scene": {...} (only on done), "message" }
```

**Thread-safe hand-off** (the core plumbing challenge): `run_pipeline()` executes inside a background `threading.Thread` (sync/blocking — same as `/command` today), but sending on a FastAPI `WebSocket` must happen from within the async event loop. The route captures `asyncio.get_running_loop()` at connection time; the `on_event` callback (called from the background pipeline thread) schedules a `websocket.send_json(...)` coroutine onto that loop via `asyncio.run_coroutine_threadsafe(...)`. This is the standard correct primitive for "schedule a coroutine on a specific loop from a different OS thread" — no new queue/polling infrastructure needed, since sends are naturally serialized onto the single event loop and stages are seconds apart (nowhere near a throughput/backpressure concern). If the client has already disconnected when an event fires, the scheduled send fails silently — the pipeline thread never observes or cares, matching the principle (already established for the renderer's file-based IPC in `core/state/ipc_store.py`) that the WebSocket is a pure observer layer, never load-bearing for pipeline correctness: the pipeline always writes to `core/outputs/live_scene.json` and `SceneState` regardless of whether anyone is listening.

**Out of scope for this route** (documented, not silently dropped): token-level/streaming LLM output within a single stage (each stage's payload arrives as one complete blob per `stage_output` event — see §11.6); run cancellation (a client can stop *watching* by disconnecting or via the overlay's "run in background" affordance, but the server-side run always completes); multiple simultaneous pipeline runs (still single-flight via `_pipeline_lock`, unchanged).

---

## 11. GUI (`gui/`)

Next.js 16 (App Router, Turbopack) + React 19 + TypeScript, using `@react-three/fiber`/`@react-three/drei` for WebGL and `@mediapipe/tasks-vision` for in-browser hand tracking. Despite talking to a Python backend, **this is primarily a self-sufficient client-side renderer**, not a thin view onto server-rendered frames.

### 11.1 What actually needs the backend, and what doesn't

| Capability | Backend required? | How |
|---|---|---|
| Browse/render example scenes | **No** | `app/api/scenes/route.ts` reads scene JSON directly off disk from `core/assets/examples/` and `renderer/assets/` (Next.js server route, not a Python call); `ThreeScene.tsx` renders it client-side |
| Load 3D mesh assets (GLB/OBJ/textures) | **No** | `app/assets/[...path]/route.ts` proxies static files from `core/assets/` |
| Cylindrical POV preview panel | **No** | `hooks/webglPov.ts` recomputes an approximate `(360,18,3)` frame client-side every ~66ms from the same scene JSON, independent of the Python renderer |
| Gesture control (rotate/zoom/freeze/reset) | **No** | 100% client-side: `getUserMedia` + MediaPipe WASM running in a Web Worker (`workers/gestureWorker.js`) + local EMA smoothing (`hooks/useGestureControl.ts`) |
| Generate a new scene from a voice/text command | **Yes** | `ws://localhost:8000/ws/pipeline` — send `{"command": "..."}`, receive live per-stage progress events, then a `run_finished` message carrying the final scene inline (see §11.6). `POST /command` + poll `/command/status` still exists unchanged for non-browser callers (voice pipeline, scripts). |

### 11.2 Scene rendering (`components/ThreeScene.tsx`, ~1250 lines)

A generic, schema-driven renderer with zero special-casing by object name — every visual behavior is derived purely from the validated `SceneDef`:
- Primitive geometries (sphere/box/cylinder/plane/ring/capsule/torus), PBR `meshStandardMaterial` with optional texture maps, all loaded via `Suspense`.
- GLB/GLTF meshes normalized to a fixed 2-unit bounding box on load (so scale behaves predictably regardless of how the source model was authored) and OBJ mesh support; per-object load failures are caught by an error boundary so one bad asset doesn't crash the whole scene.
- Full parent/child scene graph, recursively nested `<group>`s.
- Animation: orbit (including orbiting a *moving* object via `center_ref`), spin, and all four physics types (gravity/shm/pendulum/projectile) evaluated per-frame in `useFrame`.
- Interaction layer: raycast selection, scene-drag and object-drag (plane-intersection based), camera auto-fit after scene/model changes, camera reset, FPS meter, debug overlay.

### 11.3 Data flow / state (`hooks/useWebGLSceneData.ts`)

The hook actually wired into the dashboard (`app/page.tsx`): fetches the scene catalog and selected scene from the local `/api/scenes` route, runs `validateScene()` (the same schema module used by the test suite) client-side, and runs its own `requestAnimationFrame` loop to keep the POV simulation panel live. `refreshFromBackend()` is the *only* method that talks to the Python API — called after a voice/text command finishes processing.

### 11.4 Legacy/dead code present but not active

Worth knowing about so it isn't mistaken for current architecture: `hooks/useSceneData.ts` (an older hook that polls `/frame`, `/scene`, `/logs`, `/status` every 100ms — not imported by `page.tsx`), `components/render-window-panel.tsx` (an MJPEG-stream viewer + keyboard passthrough, vestige of an earlier design where the Python side rendered server-side and streamed frames to the browser), and most of the `components/ui/*` shadcn primitive set (scaffolding from the `v0.app`-generated starting point, not wired into the live dashboard).

### 11.5 Dev workflow

`npm run dev` (in `gui/`) does **not** run `next dev` directly — it runs `scripts/dev-orchestrator.js`, which spawns the Python backend (`uvicorn`) and the Next.js dev server as two tagged child processes, coordinating shutdown so neither survives the other's crash.

### 11.6 Live pipeline overlay (`components/pipeline-overlay/`)

A full-screen overlay that appears the moment a command is submitted and shows every generative-pipeline stage (§5.2) live — not just stage names, but each stage's actual output rendered in a purpose-built visual form, not a raw JSON dump.

**State/transport** (`hooks/usePipelineStream.ts`): owns a single lazily-connected `WebSocket` to `/ws/pipeline` (reused across multiple commands in one page session), reducing incoming messages into a `PipelineRunState` via `useReducer`. `page.tsx`'s `handleCommandSend` calls `pipeline.start(command)` instead of the old fetch+poll loop; a `useEffect` watching `status === "done"` still calls the existing `refreshFromBackend()` as a belt-and-suspenders re-fetch (the WS message's inlined scene is treated as a fast-path preview for the overlay's own success view, not a full replacement of the existing refresh call, since `refreshFromBackend()` may sync more dashboard state than raw scene JSON).

**Layout** (`PipelineOverlay.tsx`): a `StageTimeline.tsx` side rail (all ~15 stage rows, status dot + elapsed time, click to pin focus) next to one focused stage panel. Hand-rolled to match `HoloPanel`'s HUD-bracket visual language rather than using the installed-but-unused Radix `Dialog` — this is a progress broadcast, not a dismissible modal (non-cancellable mid-run by design; only a "run in background" affordance to stop watching, since the run can't actually be cancelled server-side). Auto-dismisses ~1.8s after `done` with a brief success beat.

**Per-stage views** (`components/pipeline-overlay/stages/`), each purpose-built rather than a generic payload dumper:
- `PromptOptimizerView` — before/after two-column text diff + clarification bullets.
- `IntentExtractionView` — extracted objects as animated pill chips (role shown via border style, count as a superscript badge), spatial relationships as a short text list.
- `AssetResolutionView` — combines the four fast/mechanical stages (semantic parse, resolve, asset registry, live search) into one panel: resolved-concept counters, verified-mesh checklist, live Poly Pizza download status.
- `ArchitectLayoutView` — a live top-down (X/Z) `<svg>` dot-plot of the Stage 7a object skeleton, with parent-child connector lines — deliberately not a second Three.js/WebGL context, just an abstract placement preview.
- `ArchitectDetailView` — per-object cards (color swatch, geometry type, animation badge) as Stage 7b fills them in.
- `ArchitectFinishView` — a lighting-rig icon/intensity-bar readout plus the scene's `name`/`summary` reveal, the first point in the flow the user sees the actual scene title.
- `CriticLoopView` — one collapsible card per critic iteration, issue cards color-coded by category with fix-instruction text, a "Scene passed review" / "Stopped after N iterations" terminal banner.
- `ValidationRepairView` — a humanized checklist derived from validator/repair actions (not raw error strings), with a final object-count summary.

**Streaming granularity, stated plainly**: each backend LLM call is one blocking request — a stage's payload arrives as one complete JSON blob per `stage_output` event, not token-by-token. The "live" feel comes from `stage_started` firing immediately (the timeline pulses before the result is known) plus CSS enter/stagger animations on arrival, not from incremental streaming within a stage. This is a deliberate scope boundary, not an accidental limitation.

**MIRROR STATUS**: `gui/lib/pipelineTypes.ts` mirrors `pipeline/events.py`'s `PipelineEvent` wire shape — see §5.3.

---

## 12. Data Flow: End-to-End Example ("add a spinning red cube")

1. **Input**: typed into the GUI command box, or spoken and transcribed locally via Whisper.
2. GUI opens (or reuses) a WebSocket to `ws://localhost:8000/ws/pipeline` and sends `{"command": "add a spinning red cube"}`. The [PipelineOverlay](gui/components/pipeline-overlay/PipelineOverlay.tsx) opens immediately.
3. Backend accepts the connection, checks the single-flight `_pipeline_lock` (unchanged from the `POST /command` era — only one pipeline run at a time by design), spawns a `"pipeline-ws"` daemon thread, and sends `run_started`.
4. Pipeline runs inside that thread, calling `run_pipeline(command, on_event=..., run_id=...)`: prompt optimizer clarifies the request → intent extractor produces a Scene Intent IR → semantic parse recognizes "cube" as a known primitive concept (no mesh lookup needed) → intent resolved → no live asset search needed → the 3-pass architect (layout → detail → finish) builds a red box object with `animation: {type: "spin", ...}` and appropriate lighting/camera → the critic/fixer loop reviews and either confirms it clean or fixes flagged issues, up to 3 iterations → validated → any remaining structural issues auto-repaired → cached in Redis (keyed on the original transcript) → written to `core/outputs/live_scene.json` and pushed to `SceneState.scene_json` + `scene_command.json` (IPC). Each stage transition and output is streamed back over the same WebSocket connection as a `pipeline_event` message (via `asyncio.run_coroutine_threadsafe`, since the pipeline thread is sync/blocking but the socket send must happen on the FastAPI event loop) and rendered live in the corresponding overlay panel.
5. On completion, the backend sends `run_finished` with the final scene JSON inlined; the GUI hydrates the 3D view from it and also calls `refreshFromBackend()` (§11.6) as a full-state re-sync; the overlay auto-dismisses.
6. If the Python renderer process is also running, it independently picks up the same scene via `scene_command.json`, rebuilds its own OpenGL scene graph, and starts computing both the JPEG preview and the `(360,18,3)` cylindrical LED frame every render loop.
7. The user can now gesture at their webcam (pinch to rotate the cube, fist to freeze it) — entirely client-side in the GUI, or (if running) via the separate `GestureEngine` writing into the shared `SceneState` that the Python renderer reads.

---

## 13. Directory Reference

```
holoscript-mini/
├── backend/            FastAPI façade — pipeline trigger (HTTP + WebSocket), IPC proxy, no rendering itself
├── core/
│   ├── state/           scene_state.py (in-process blackboard), ipc_store.py (cross-process files)
│   ├── utils/            logger.py, config.py
│   ├── outputs/          live_scene.json (gitignored, regenerated per pipeline run)
│   └── assets/            meshes/, hdri/, examples/*.json (17 hand-authored demo scenes)
├── pipeline/            the generative pipeline — prompt optimize → intent extract → semantic parse →
│   │                    resolve → asset verify → 3-pass LLM architect → critic loop → intent verify/
│   │                    realign loop → validate → repair (§5.2)
│   ├── events.py          PipelineEvent contract — mirrored by gui/lib/pipelineTypes.ts
│   ├── prompt_optimizer.py Stage 2 — LLM prompt clarification
│   ├── intent_extractor.py Stage 3 — LLM structured Scene Intent IR
│   ├── scene_architect.py  Stage 7 — 3-pass LLM scene builder (layout/detail/finish);
│   │                       also exposes regenerate_pass() for Stage 8.5's targeted re-runs
│   ├── critic_agent.py     Stage 8 — iterative critique/fix loop (object-level technical defects)
│   ├── intent_verifier.py  Stage 8.5 — holistic scene-vs-request comparison + targeted realignment
│   └── knowledge_base/   MongoDB client + sentence-transformer embedder
├── llm/
│   ├── gemini_client.py  shared Vertex AI (Gemini) + Groq client — used by every LLM-calling pipeline stage
│   └── groq_client.py    older, smaller Groq wrapper — used only by the legacy fallback path (dead-code note, §5.3)
├── voice/               VAD segmentation, Whisper transcription, recording fallback
├── gesture/             MediaPipe hand tracking, 9-stage gesture engine, OpenCV fallback
├── renderer/             standalone OpenGL/pyglet process — 3D rendering + cylindrical POV frame builder
│   ├── engine/            animation, transforms, primitives, mesh loading, cylindrical projection
│   ├── loader/            scene JSON → SceneObject
│   └── main/              render_window.py — the executable entry point
├── gui/                 Next.js 16 + React 19 — WebGL renderer, gesture control, command UI, live pipeline overlay
│   ├── app/               routes: page.tsx (dashboard), api/scenes, assets/[...path]
│   ├── components/        ThreeScene.tsx and supporting panels
│   │   └── pipeline-overlay/  PipelineOverlay.tsx, StageTimeline.tsx, stages/*View.tsx (§11.6)
│   ├── lib/                sceneFactory.ts — THE canonical scene schema + validator;
│   │                       pipelineTypes.ts — mirrors pipeline/events.py's wire contract
│   ├── hooks/              state management (useWebGLSceneData, usePipelineStream)
│   └── workers/            gestureWorker.js — off-main-thread MediaPipe inference
├── .runtime/            gitignored — file-based IPC channel between backend and renderer processes
├── requirements.txt     Python dependencies
└── README.md            quick-start (partially stale — see §5.4 caveats)
```

---

## 14. Environment Variables Reference

No `.env.example` exists in the repo; this table is assembled from source (`os.getenv` call sites) since it's the closest thing to one:

| Variable | Used by | Purpose | Default |
|---|---|---|---|
| `MONGODB_URI` | `pipeline/knowledge_base/mongo_client.py` | Concept knowledge base | none — raises if unset |
| `MONGODB_DB` | same | Database name | `holoscript` |
| `MONGODB_COLLECTION` | same | Collection name | `knowledge_base` |
| `REDIS_URL` | `pipeline/cache.py` | Asset/scene cache | `redis://localhost:6379/0` |
| `REDIS_ASSET_TTL` | same | Asset cache TTL (s) | `604800` (7d) |
| `REDIS_SCENE_TTL` | same | Scene cache TTL (s) | `21600` (6h) |
| `EMBEDDING_MODEL` | `pipeline/knowledge_base/embedder.py` | Sentence-transformer model | `all-MiniLM-L6-v2` |
| `EMBEDDING_TOP_K` | same | Search result count | `3` |
| `EMBEDDING_SIMILARITY_THRESHOLD` | same | Min cosine similarity | `0.45` |
| `GROQ_API_KEY` | `llm/gemini_client.py` (shared, used by every LLM stage), `llm/groq_client.py` | Groq LLM auth | none — Gemini/Vertex-only if unset, no longer a hard gate (§5.4) |
| `GEMINI_CRITIC_MODEL` | `critic_agent.py` | Critique/fix model (all 3 loop iterations) | `gemini-2.5-flash` |
| `GCP_PROJECT` | `llm/gemini_client.py` (shared) | Vertex AI project | `reportevaluator` |
| `GCP_LOCATION` | same | Vertex AI region | `us-central1` |
| `POLYPIZZA_API_KEY` / `POLY_PIZZA_API_KEY` | `live_search.py`, `asset_ingester.py` | Live 3D asset search | none — live search silently disabled |

Gemini/Vertex AI uses Application Default Credentials, not an API key.

Note: the new Stage 2/3/7 modules (`prompt_optimizer.py`, `intent_extractor.py`, `scene_architect.py`'s three passes) use hardcoded model constants (`gemini-2.5-flash` for cheap/fast tasks, `gemini-2.5-pro` for reasoning-heavy ones) rather than separate env-var overrides — only the critic's model remains independently configurable via `GEMINI_CRITIC_MODEL`, matching its pre-existing convention. `GEMINI_ARCHITECT_MODEL` (the old single-call architect's override) no longer applies now that the architect is 3 passes with different per-pass models.

---

## 15. Notable Design Decisions & Their Rationale

- **Fail-soft everywhere, not fail-fast**: malformed scene objects are dropped rather than rejecting the whole scene; VAD failures fall back to fixed-duration recording; Redis/Mongo unavailability degrades gracefully; the gesture engine falls back to OpenCV color detection if MediaPipe's model is missing. The system is built to always produce *something* rather than surface an error to the end user, since the failure modes (bad LLM output, flaky webcam, no network) are expected to be common in this domain.
- **Schema duplication over a shared library**: the scene schema is defined once in TypeScript (`sceneFactory.ts`) and hand-mirrored in Python (`scene_validator.py`), rather than sharing a single source (e.g. via JSON Schema codegen). This is a deliberate-looking tradeoff for simplicity at the current scale, but is a correctness risk if the two drift — the Python file's docstring explicitly cross-references TS line numbers to make manual sync-checking possible.
- **Repair before reject**: the pipeline treats "invalid scene" as a solvable problem (structural repair loop, physics clamping, mesh-path substitution) rather than a terminal error, because the upstream LLM call is expensive (seconds of latency) and unreliable enough that discarding its output on a minor error would be wasteful.
- **File-based IPC instead of sockets/RPC**: chosen so the OpenGL renderer (which needs to own the main thread for its event loop) and the FastAPI backend can run as fully independent processes with no network protocol between them — at the cost of polling latency (up to one frame, or ~0.35s for snapshot publishing).
- **Client-side scene rendering, server-side generation only**: the GUI doesn't depend on the Python backend for anything except the LLM call — this means scene browsing, gesture control, and the POV preview all work offline/without Python running, which matters for demoing or developing the frontend in isolation.
- **Multi-pass generation over one giant prompt**: the scene architect (Stage 7) was deliberately split into layout → detail → finish rather than kept as a single call, because a single call has to simultaneously juggle object composition, per-object numeric precision, and lighting/camera framing — and quality/consistency degrades as scene complexity grows. Each pass maps onto a natural seam the original single prompt's constants already had (`_RULES`' primitive/mesh section → layout, `_SCHEMA`/`_EDUCATION`/`_PHYSICS` → detail, `_RULES`' lighting/camera section → finish), so the split cost no new prompt-engineering surface, only orchestration.
- **A transient reasoning IR, not a schema extension**: the Scene Intent IR (Stage 3) deliberately isn't part of the canonical scene schema and is never validated or persisted — adding it to `sceneFactory.ts`/`scene_validator.py` would have created a second dual-maintenance obligation for something that's purely advisory context for the architect prompts, not a structural requirement of the final scene.
- **Callback-based event emission, not a message queue**: pipeline progress events are threaded through `run_pipeline()` as a plain `on_event` callback rather than a queue or global event bus, specifically so `pipeline/events.py` and every stage module stay transport-agnostic — they have zero knowledge of WebSockets or asyncio. The WebSocket-specific plumbing (`asyncio.run_coroutine_threadsafe` to hand events from the background pipeline thread to the event loop) lives entirely in `backend/api_server.py`, where transport concerns belong.
- **A capped, self-terminating critic loop**: the critic/fixer (Stage 8) loops at most 3 times and stops immediately once a pass finds no issues, rather than looping until perfect or looping unboundedly. This bounds worst-case added latency to roughly 3 pairs of Flash-model calls (acceptable given "go all in" on quality was an explicit requirement) while still converging quickly on the common case where the first critique already passes.
