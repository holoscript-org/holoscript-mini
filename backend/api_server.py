# -*- coding: utf-8 -*-
"""backend/api_server.py
FastAPI bridge between the Python renderer (SceneState) and the React GUI.

Run with:
    uvicorn backend.api_server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
from core.utils.logger import get_logger
from pipeline.events import PipelineEvent, new_run_id

logger = get_logger("api_server")

# ---------------------------------------------------------------------------
# Pipeline state — shared between /command, /command/status, and /ws/pipeline
# ---------------------------------------------------------------------------

_pipeline_lock = threading.Lock()
_pipeline_state: dict[str, Any] = {
    "running": False, "state": "idle", "message": "", "run_id": None,
}


# ---------------------------------------------------------------------------
# Startup lifespan — preload semantic model so first command is instant
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    def _preload():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from pipeline.semantic_parser import get_parser
            get_parser()
            logger.info("Semantic parser preloaded and ready")
        except Exception as exc:
            logger.warning("Semantic parser preload failed: %s", exc)

    threading.Thread(target=_preload, daemon=True, name="preload-semantic").start()
    yield


app = FastAPI(title="HoloScript API", version="1.0.0", lifespan=lifespan)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORE_DIR = _PROJECT_ROOT / "core"
_LIVE_SCENE_PATH = _CORE_DIR / "outputs" / "live_scene.json"
_RENDER_PREVIEW_PATH = _PROJECT_ROOT / ".runtime" / "render_preview.jpg"
_RENDER_PREVIEW_FALLBACK = _PROJECT_ROOT / "renderer" / "assets" / "test_frame.png"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def log_unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled exception at %s %s", request.method, request.url.path)
    return {"detail": "Internal server error"}


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

    if scene_state.scene_json:
        return scene_state.scene_json

    if _LIVE_SCENE_PATH.exists():
        try:
            return json.loads(_LIVE_SCENE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {}


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
        # Normalize snapshot fields — snapshot may be up to _snapshot_interval_sec stale.
        # ts reflects when this response was assembled, not when the snapshot was written.
        return {
            "rotation_y": float(status.get("rotation_y", 0.0)),
            "scale":      float(status.get("scale", 1.0)),
            "explode":    float(status.get("explode", 0.0)),
            "frozen":     bool(status.get("frozen", False)),
            "gesture":    str(status.get("gesture", "NONE")),
            "ts":         time.monotonic(),
        }

    # Fallback: read directly from SceneState (gesture engine in same process).
    rotation_y, scale, explode, frozen = scene_state.get_render_params()
    return {
        "rotation_y": rotation_y,
        "scale":      scale,
        "explode":    explode,
        "frozen":     frozen,
        "gesture":    scene_state.current_gesture,
        "ts":         time.monotonic(),
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



# ---------------------------------------------------------------------------
# POST /command  — run the Member 1 pipeline from a text/voice transcript
# GET  /command/status — poll pipeline state (running / done / error)
# ---------------------------------------------------------------------------

class CommandPayload(BaseModel):
    command: str


@app.post("/command")
def post_command(payload: CommandPayload) -> dict[str, str]:
    with _pipeline_lock:
        if _pipeline_state["running"]:
            return {"status": "busy", "message": "Pipeline already running"}
        _pipeline_state.update({"running": True, "state": "running", "message": payload.command})

    def _run() -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from pipeline.pipeline_runner import run_pipeline
            scene = run_pipeline(payload.command)
            scene_state.scene_json = scene
            publish_scene_command(scene)
            with _pipeline_lock:
                _pipeline_state.update({"running": False, "state": "done", "message": "Scene ready"})
        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
            with _pipeline_lock:
                _pipeline_state.update({"running": False, "state": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True, name="pipeline").start()
    return {"status": "processing", "message": payload.command}


@app.get("/command/status")
def get_command_status() -> dict[str, Any]:
    with _pipeline_lock:
        return dict(_pipeline_state)


# ---------------------------------------------------------------------------
# WS /ws/pipeline — live pipeline progress stream
#
# Additive alongside POST /command and GET /command/status above, which are
# left completely unchanged for any existing caller (voice pipeline, scripts,
# curl). This route is the new trigger + live progress path used by the
# frontend's pipeline overlay: client sends {"command": "..."}, server runs
# the same run_pipeline() the /command handler uses — but now passing an
# on_event callback that streams a PipelineEvent for every stage transition
# and stage output back over the same socket.
#
# Wire message schema:
#   {"type": "run_started",  "run_id": "...", "transcript": "..."}
#   {"type": "pipeline_event", "run_id", "stage", "status", "label",
#    "payload", "provider", "timestamp", "elapsed_ms"}
#   {"type": "run_finished", "run_id", "status": "done"|"error",
#    "scene": {...} (only on done), "message": "..."}
#
# The connection is not closed after run_finished — it can be reused for a
# subsequent command in the same page session. A client disconnecting
# mid-run does not stop the pipeline: it already writes to
# core/outputs/live_scene.json and scene_state regardless of any listener,
# so the WebSocket is a pure observer layer, never load-bearing for pipeline
# correctness (same principle as the existing IPC bridge to the renderer
# process in core/state/ipc_store.py).
# ---------------------------------------------------------------------------


def _event_to_wire(event: PipelineEvent) -> dict[str, Any]:
    return {"type": "pipeline_event", **asdict(event)}


@app.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            raw = await websocket.receive_json()
            command = str(raw.get("command", "")).strip()
            if not command:
                continue

            with _pipeline_lock:
                if _pipeline_state["running"]:
                    await websocket.send_json({
                        "type": "run_finished",
                        "run_id": _pipeline_state.get("run_id"),
                        "status": "error",
                        "message": "Pipeline already running",
                    })
                    continue
                run_id = new_run_id()
                _pipeline_state.update({
                    "running": True, "state": "running",
                    "message": command, "run_id": run_id,
                })

            await websocket.send_json({
                "type": "run_started", "run_id": run_id, "transcript": command,
            })

            def on_event(event: PipelineEvent) -> None:
                # Called from the background pipeline thread — hand off to
                # the event loop that owns this WebSocket. If the client has
                # already disconnected, sending will fail; that's fine, the
                # pipeline thread never observes or cares about this outcome.
                async def _send() -> None:
                    try:
                        await websocket.send_json(_event_to_wire(event))
                    except Exception:
                        pass

                asyncio.run_coroutine_threadsafe(_send(), loop)

            def _run(command: str = command, run_id: str = run_id) -> None:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    from pipeline.pipeline_runner import run_pipeline
                    scene = run_pipeline(command, on_event=on_event, run_id=run_id)
                    scene_state.scene_json = scene
                    publish_scene_command(scene)
                    with _pipeline_lock:
                        _pipeline_state.update({"running": False, "state": "done", "message": "Scene ready"})
                    finished = {
                        "type": "run_finished", "run_id": run_id,
                        "status": "done", "scene": scene, "message": "Scene ready",
                    }
                except Exception as exc:
                    logger.error("Pipeline error: %s", exc)
                    with _pipeline_lock:
                        _pipeline_state.update({"running": False, "state": "error", "message": str(exc)})
                    finished = {
                        "type": "run_finished", "run_id": run_id,
                        "status": "error", "message": str(exc),
                    }

                async def _send_finished() -> None:
                    try:
                        await websocket.send_json(finished)
                    except Exception:
                        pass

                asyncio.run_coroutine_threadsafe(_send_finished(), loop)

            threading.Thread(target=_run, daemon=True, name="pipeline-ws").start()

    except WebSocketDisconnect:
        logger.info("WS /ws/pipeline: client disconnected")


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
