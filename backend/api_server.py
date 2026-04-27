# -*- coding: utf-8 -*-
"""backend/api_server.py
FastAPI bridge between the Python renderer (SceneState) and the React GUI.

Run with:
    uvicorn backend.api_server:app --reload --port 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any, Generator

from core.state.scene_state import scene_state
from core.state.ipc_store import (
    read_renderer_snapshot,
    publish_scene_command,
    publish_control_command,
)

app = FastAPI(title="HoloScript API", version="1.0.0")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORE_DIR = _PROJECT_ROOT / "core"
_RENDER_PREVIEW_PATH = _PROJECT_ROOT / ".runtime" / "render_preview.jpg"
_RENDER_PREVIEW_FALLBACK = _PROJECT_ROOT / "renderer" / "assets" / "test_frame.png"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /frame  — cylindrical POV frame (360 × 18 × 3, uint8)
# ---------------------------------------------------------------------------

@app.get("/frame")
def get_frame() -> dict[str, Any]:
    snap = read_renderer_snapshot()
    frame = snap.get("frame")
    if frame is not None:
        return {"frame": frame.tolist()}

    frame = scene_state.current_frame
    if frame is None:
        return {"frame": None}
    return {"frame": frame.tolist()}


# ---------------------------------------------------------------------------
# GET /scene  — current scene JSON or load by name from core/assets/
# ---------------------------------------------------------------------------

@app.get("/scene")
def get_scene(name: str | None = None) -> dict[str, Any]:
    if name is not None:
        grammar_path = _CORE_DIR / "outputs" / "scene_grammar.json"
        if grammar_path.exists():
            try:
                return json.loads(grammar_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        example_path = _CORE_DIR / "assets" / "examples" / f"{name}.json"
        if example_path.exists():
            try:
                return json.loads(example_path.read_text(encoding="utf-8"))
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to read scene: {name}")
        raise HTTPException(status_code=404, detail=f"Scene not found: {name}")

    snap = read_renderer_snapshot()
    scene = snap.get("scene")
    if isinstance(scene, dict):
        return scene
    return scene_state.scene_json or {}


# ---------------------------------------------------------------------------
# POST /scene  — push a new scene JSON from the GUI
# ---------------------------------------------------------------------------

class ScenePayload(BaseModel):
    scene: dict[str, Any]

@app.post("/scene")
def set_scene(payload: ScenePayload) -> dict[str, str]:
    try:
        scene_state.scene_json = payload.scene
        publish_scene_command(payload.scene)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /logs  — latest renderer / system log entries (newest last)
# ---------------------------------------------------------------------------

@app.get("/logs")
def get_logs() -> dict[str, list[str]]:
    snap = read_renderer_snapshot()
    logs = snap.get("logs")
    if isinstance(logs, list):
        return {"logs": [str(item) for item in logs]}
    return {"logs": scene_state.logs}


# ---------------------------------------------------------------------------
# GET /status  — live render-transform parameters + gesture
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status() -> dict[str, Any]:
    snap = read_renderer_snapshot()
    status = snap.get("status")
    if isinstance(status, dict):
        return status

    rotation_y, scale, explode, frozen = scene_state.get_render_params()
    return {
        "rotation_y": rotation_y,
        "scale": scale,
        "explode": explode,
        "frozen": frozen,
        "gesture": scene_state.current_gesture,
        "transcript": scene_state.transcript,
    }


# ---------------------------------------------------------------------------
# GET /render-preview  — single latest snapshot (used as fallback / first load)
# ---------------------------------------------------------------------------

@app.get("/render-preview")
def get_render_preview() -> FileResponse:
    if _RENDER_PREVIEW_PATH.exists():
        return FileResponse(_RENDER_PREVIEW_PATH, media_type="image/jpeg")
    if _RENDER_PREVIEW_FALLBACK.exists():
        return FileResponse(_RENDER_PREVIEW_FALLBACK, media_type="image/png")
    raise HTTPException(status_code=404, detail="No renderer preview available")


# ---------------------------------------------------------------------------
# GET /stream  — MJPEG live stream (multipart/x-mixed-replace)
#
# The browser opens one persistent connection; the server continuously pushes
# JPEG frames separated by MIME boundaries.  The <img> tag displays each
# frame as it arrives with no polling or src-swap latency.
# ---------------------------------------------------------------------------

_MJPEG_INTERVAL = 1 / 30   # target ~30 fps push rate


def _mjpeg_generator() -> Generator[bytes, None, None]:
    last_mtime: float = 0.0
    last_data: bytes | None = None

    while True:
        path = _RENDER_PREVIEW_PATH if _RENDER_PREVIEW_PATH.exists() else (
            _RENDER_PREVIEW_FALLBACK if _RENDER_PREVIEW_FALLBACK.exists() else None
        )

        if path is not None:
            try:
                mtime = path.stat().st_mtime
                if mtime != last_mtime or last_data is None:
                    last_data = path.read_bytes()
                    last_mtime = mtime
            except OSError:
                pass

        if last_data:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + last_data
                + b"\r\n"
            )

        time.sleep(_MJPEG_INTERVAL)


@app.get("/stream")
def stream_preview() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# POST /control  — keyboard shortcut bridge (GUI → renderer via SceneState)
#
# Mirrors exactly what on_key_press does in renderer/render_window.py so the
# React GUI can trigger the same transforms without a physical key event on
# the Pyglet window.
# ---------------------------------------------------------------------------



_VALID_ACTIONS = {"space", "left", "right", "up", "down", "e", "r", "j", "k", "h"}


class ControlPayload(BaseModel):
    action: str


@app.post("/control")
def post_control(payload: ControlPayload) -> dict[str, str]:
    action = payload.action.lower()
    if action not in _VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    # Write to the IPC file — the renderer process reads and applies it on its
    # next on_draw tick, so scene_state mutations happen in the correct process.
    publish_control_command(action)
    return {"status": "ok", "action": action}
