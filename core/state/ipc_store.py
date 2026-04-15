"""Lightweight file-based IPC bridge between renderer and FastAPI.

The renderer publishes live state snapshots for backend consumers.
The backend publishes scene-change commands for the renderer.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


_ROOT_DIR = Path(__file__).resolve().parents[2]
_RUNTIME_DIR = _ROOT_DIR / ".runtime"
_SNAPSHOT_PATH = _RUNTIME_DIR / "scene_snapshot.json"
_FRAME_PATH = _RUNTIME_DIR / "current_frame.npy"
_COMMAND_PATH = _RUNTIME_DIR / "scene_command.json"
_CONTROL_PATH = _RUNTIME_DIR / "control_command.json"


def _ensure_runtime_dir() -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> bool:
    tmp_path = path.with_name(f"{path.stem}.{time.time_ns()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp_path, path)
        return True
    except PermissionError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _atomic_write_npy(path: Path, frame: np.ndarray) -> bool:
    tmp_path = path.with_name(f"{path.stem}.{time.time_ns()}.tmp.npy")
    with tmp_path.open("wb") as f:
        np.save(f, frame, allow_pickle=False)
    try:
        os.replace(tmp_path, path)
        return True
    except PermissionError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def publish_renderer_snapshot(
    *,
    scene: dict[str, Any] | None,
    logs: list[str],
    status: dict[str, Any],
    frame: np.ndarray | None,
    frame_updated: bool,
) -> None:
    """Publish the renderer state for API consumers in other processes."""
    _ensure_runtime_dir()

    frame_written = False
    if frame_updated and frame is not None:
        frame_written = _atomic_write_npy(_FRAME_PATH, frame)

    payload = {
        "updated_at": time.time(),
        "scene": scene,
        "logs": logs,
        "status": status,
        "has_frame": bool(frame_written or frame is not None or _FRAME_PATH.exists()),
    }
    _atomic_write_text(_SNAPSHOT_PATH, json.dumps(payload, ensure_ascii=True))


def read_renderer_snapshot() -> dict[str, Any]:
    """Read the latest renderer snapshot and optional frame array."""
    if not _SNAPSHOT_PATH.exists():
        return {}

    try:
        payload = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
    except (OSError, json.JSONDecodeError):
        return {}

    if payload.get("has_frame") and _FRAME_PATH.exists():
        try:
            with _FRAME_PATH.open("rb") as f:
                payload["frame"] = np.load(f, allow_pickle=False)
        except (OSError, ValueError):
            payload["frame"] = None
    else:
        payload["frame"] = None

    return payload


def publish_scene_command(scene: dict[str, Any]) -> int:
    """Publish a scene-change command for the renderer process."""
    _ensure_runtime_dir()
    command_id = time.time_ns()
    payload = {"id": command_id, "scene": scene}
    _atomic_write_text(_COMMAND_PATH, json.dumps(payload, ensure_ascii=True))
    return command_id


def read_scene_command() -> dict[str, Any] | None:
    """Read the latest scene command written by backend or GUI."""
    if not _COMMAND_PATH.exists():
        return None

    try:
        payload = json.loads(_COMMAND_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("id"), int):
        return None
    if not isinstance(payload.get("scene"), dict):
        return None

    return payload


def publish_control_command(action: str) -> int:
    """Publish a keyboard-control action for the renderer process.

    The renderer polls this file in on_draw and applies it exactly once
    (tracked by command id), mirroring what on_key_press does locally.
    """
    _ensure_runtime_dir()
    command_id = time.time_ns()
    payload = {"id": command_id, "action": action}
    _atomic_write_text(_CONTROL_PATH, json.dumps(payload, ensure_ascii=True))
    return command_id


def read_control_command() -> dict[str, Any] | None:
    """Read the latest control command written by the backend."""
    if not _CONTROL_PATH.exists():
        return None
    try:
        payload = json.loads(_CONTROL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("id"), int):
        return None
    if not isinstance(payload.get("action"), str):
        return None
    return payload
