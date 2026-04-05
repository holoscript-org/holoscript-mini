# Renderer Module — Integration Guide

Member 2 owns `renderer/` at the repo root. This document explains how other members interact with the Renderer through the shared `SceneState` blackboard (`core/state/scene_state.py`).

---

## How Member 1 triggers a scene update

Member 1 (Voice/LLM) simply writes a new scene dict to the blackboard:

```python
from core.state.scene_state import scene_state

scene_state.scene_json = {
    "objects": [
        {"id": "sun", "type": "sphere", "position": [0,0,0],
         "color": [1.0, 0.84, 0.0], "animation": "none",
         "orbit_center": [0,0,0], "orbit_speed": 0.0},
        ...
    ]
}
```

The Renderer detects the change automatically on its next frame by comparing `scene_state.scene_version` against its cached value. No function call, no signal, no callback is needed. The rebuild happens within one frame (~16 ms). If the new JSON is malformed or produces no valid objects, the Renderer logs a warning and keeps the previous scene — it never crashes.

---

## How Member 4 reads the POV frame

`scene_state.current_frame` is updated every rendered frame. It is always safe to read from any thread:

```python
from core.state.scene_state import scene_state
import numpy as np

frame = scene_state.current_frame  # None until first frame is ready
if frame is not None:
    # frame.shape == (360, 18, 3)
    # frame.dtype == np.uint8
    # frame[angle_idx, led_idx] = (R, G, B) values 0-255
    pass
```

- **Shape:** `(360, 18, 3)` — 360 angular slots × 18 LEDs × 3 RGB channels
- **Dtype:** `uint8`
- **Update rate:** Every rendered frame (target ≥ 25 FPS)
- **Thread safety:** Reads are protected by `SceneState._lock` (RLock)

The frame represents what a spinning 18-LED arm should display at each of 360 angular positions to holographically reconstruct the current scene.

---

## SceneState fields written by Renderer

| Field | Shape / Type | Update frequency | Notes |
|---|---|---|---|
| `current_frame` | `(360, 18, 3)` uint8 | Every frame | POV cylindrical projection |

---

## SceneState fields read by Renderer

| Field | Type | Expected range | Purpose |
|---|---|---|---|
| `scene_json` | `dict` or `None` | Any valid scene dict | Parsed into SceneObject list on version change |
| `scene_version` | `int` | 0 → ∞, monotone | Change detection; compared each frame |
| `rotation_y` | `float` | Any degrees | Global Y-axis rotation of entire scene |
| `scale` | `float` | 0.2 – 3.0 (clamped by keyboard handler) | Uniform scale applied before rotation |
| `explode` | `float` | 0.0 – 1.0 | Pushes objects outward from origin (exploded view) |
| `frozen` | `bool` | `True` / `False` | Pauses orbital animation when `True` |

All six are read atomically via `get_render_params()` (single lock acquisition) for `rotation_y / scale / explode / frozen`. `scene_json` and `scene_version` are read together under a single lock at the top of each frame.

---

## Keyboard controls (demo / testing)

| Key | Action |
|---|---|
| `SPACE` | Toggle animation freeze (pauses orbital motion) |
| `LEFT` | Rotate scene left 5° (decrements `rotation_y`) |
| `RIGHT` | Rotate scene right 5° (increments `rotation_y`) |
| `UP` | Zoom in — increases `scale` by 0.1 (max 3.0) |
| `DOWN` | Zoom out — decreases `scale` by 0.1 (min 0.2) |
| `E` | Increase explode factor by 0.1 (max 1.0) |
| `R` | Reset all transforms to defaults (rotation=0, scale=1, explode=0) |
| `S` | Save current raw OpenGL frame to `renderer/assets/test_frame.png` |
| `V` | Save cylindrical POV visualization to `renderer/assets/pov_frame.png` |
| `J` | Inject a 3-object test scene (sun + 2 planets) into `scene_state.scene_json` |
| `K` | Reload the solar system JSON from `renderer/assets/solar_system.json` |

---

## Performance targets

| Operation | Target |
|---|---|
| Render FPS | ≥ 25 FPS |
| Frame extraction (`glReadPixels`) | < 10 ms per frame |
| Cylindrical engine (`build_frame_from_scene`) | < 50 ms per frame |
| Scene rebuild (`parse_scene` on JSON change) | < 100 ms (triggered only on `scene_version` change) |

Performance is logged to `scene_state.logs` every 25 frames as:
```
[Renderer] FPS=XX.X objects=N
```
