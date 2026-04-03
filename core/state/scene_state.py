"""
core/state/scene_state.py
=========================
Central blackboard for the HoloScript system.

Ownership contract
------------------
- scene_json      → written by Voice/LLM thread
- transcript      → written by Voice/LLM thread
- rotation_y      → written by Gesture thread
- scale           → written by Gesture thread
- explode         → written by Gesture thread
- frozen          → written by Gesture thread
- current_gesture → written by Gesture thread
- current_frame   → written by Renderer thread
- logs            → appended by any thread
- scene_history   → appended by Voice/LLM thread

All threads may read any field; all access goes through the
thread-safe getters / setters below.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRAME_SHAPE: tuple[int, int, int] = (360, 18, 3)
_FRAME_DTYPE = np.uint8


# ---------------------------------------------------------------------------
# SceneState
# ---------------------------------------------------------------------------


class SceneState:
    """Thread-safe blackboard shared by all subsystem threads.

    All public attributes are accessed through explicit getters and setters
    that acquire the internal :class:`threading.RLock`, making every read
    and write atomic with respect to the lock.

    The lock is reentrant (``RLock``) so that a single thread may call
    multiple getters / setters within the same logical operation without
    deadlocking.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # Voice / LLM
        self._scene_json: Dict[str, Any] | None = None
        self._transcript: str = ""

        # Gesture
        self._rotation_y: float = 0.0
        self._scale: float = 1.0
        self._explode: float = 0.0
        self._frozen: bool = False
        self._current_gesture: str = "NONE"
        self._hand_present: bool = False
        self._nav_event: str = ""        # written by gesture engine, consumed by renderer

        # NEW ── PREVIEW / SNAP / FOCUS layer ──────────────────────────────────
        self._nav_phase: str = "NAV"          # NAV / PREVIEW / FOCUS
        self._preview_candidate: str = ""     # object highlighted in PREVIEW
        self._focused_object: str = ""        # object locked in FOCUS

        # Renderer output
        self._current_frame: np.ndarray | None = None

        # Audit / history
        self._logs: Deque[str] = deque(maxlen=100)
        self._scene_history: Deque[Dict[str, Any]] = deque(maxlen=20)

    # ------------------------------------------------------------------
    # scene_json
    # ------------------------------------------------------------------

    @property
    def scene_json(self) -> Dict[str, Any] | None:
        """Most-recently parsed scene description from the LLM, or ``None``."""
        with self._lock:
            return self._scene_json

    @scene_json.setter
    def scene_json(self, value: Dict[str, Any] | None) -> None:
        """Set a new scene description.

        When a non-``None`` value is supplied the previous value (if any) is
        automatically pushed onto :attr:`scene_history` for undo support.

        Args:
            value: Parsed scene dict produced by the Voice/LLM thread, or
                ``None`` to clear the current scene.

        Raises:
            TypeError: If *value* is neither a :class:`dict` nor ``None``.
        """
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"scene_json must be a dict or None, got {type(value).__name__!r}")
        with self._lock:
            if self._scene_json is not None:
                self._scene_history.append(self._scene_json)
            self._scene_json = value

    # ------------------------------------------------------------------
    # transcript
    # ------------------------------------------------------------------

    @property
    def transcript(self) -> str:
        """Latest raw speech transcript from the Voice thread."""
        with self._lock:
            return self._transcript

    @transcript.setter
    def transcript(self, value: str) -> None:
        """Update the transcript.

        Args:
            value: Recognised speech string.

        Raises:
            TypeError: If *value* is not a :class:`str`.
        """
        if not isinstance(value, str):
            raise TypeError(f"transcript must be a str, got {type(value).__name__!r}")
        with self._lock:
            self._transcript = value

    # ------------------------------------------------------------------
    # rotation_y
    # ------------------------------------------------------------------

    @property
    def rotation_y(self) -> float:
        """Y-axis rotation in degrees, driven by the Gesture thread."""
        with self._lock:
            return self._rotation_y

    @rotation_y.setter
    def rotation_y(self, value: float) -> None:
        """Set the Y-axis rotation.

        Args:
            value: Rotation angle in degrees (converted to ``float``).
        """
        with self._lock:
            self._rotation_y = float(value)

    # ------------------------------------------------------------------
    # scale
    # ------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """Uniform scale factor, driven by the Gesture thread."""
        with self._lock:
            return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        """Set the uniform scale factor.

        Args:
            value: Positive scale multiplier (converted to ``float``).

        Raises:
            ValueError: If *value* is not positive.
        """
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"scale must be positive, got {value}")
        with self._lock:
            self._scale = value

    # ------------------------------------------------------------------
    # explode
    # ------------------------------------------------------------------

    @property
    def explode(self) -> float:
        """Explode factor in [0, 1] for exploded-view rendering."""
        with self._lock:
            return self._explode

    @explode.setter
    def explode(self, value: float) -> None:
        """Set the explode factor.

        Args:
            value: Float in [0.0, 1.0] (clamped automatically).
        """
        with self._lock:
            self._explode = max(0.0, min(1.0, float(value)))

    # ------------------------------------------------------------------
    # frozen
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        """Whether animation is frozen; set by the Gesture thread."""
        with self._lock:
            return self._frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        """Pause or resume animation.

        Args:
            value: ``True`` to freeze, ``False`` to resume.
        """
        with self._lock:
            self._frozen = bool(value)

    # ------------------------------------------------------------------
    # hand_present
    # ------------------------------------------------------------------

    @property
    def hand_present(self) -> bool:
        """Whether a hand is currently detected by the Gesture thread."""
        with self._lock:
            return self._hand_present

    @hand_present.setter
    def hand_present(self, value: bool) -> None:
        """Set hand detection status.

        Args:
            value: ``True`` if hand detected, ``False`` otherwise.
        """
        with self._lock:
            self._hand_present = bool(value)

    # NEW ----------------------------------------------------------------
    # nav_event  — write-once / consume-once navigation step from POINT mode
    # -------------------------------------------------------------------

    @property
    def nav_event(self) -> str:
        """Pending navigation event string set by the gesture engine.

        One of: ``"NEXT"``, ``"PREV"``, ``"UP"``, ``"DOWN"``, or ``""`` (none).
        Prefer :meth:`consume_nav_event` in the renderer to avoid re-triggering.
        """
        with self._lock:
            return self._nav_event

    @nav_event.setter
    def nav_event(self, value: str) -> None:
        """Write a navigation step event (called by gesture engine only)."""
        with self._lock:
            self._nav_event = str(value)

    def consume_nav_event(self) -> str:
        """Atomically read and clear the pending navigation event.

        The renderer should call this once per frame.  Returns the event string
        (``"NEXT"``, ``"PREV"``, ``"UP"``, ``"DOWN"``) or ``""`` if none pending.
        """
        with self._lock:
            evt = self._nav_event
            self._nav_event = ""
            return evt

    # NEW ── nav_phase / preview_candidate / focused_object ──────────────────

    @property
    def nav_phase(self) -> str:
        """Interaction phase: ``\"NAV\"``, ``\"PREVIEW\"``, or ``\"FOCUS\"``."""
        with self._lock:
            return self._nav_phase

    @nav_phase.setter
    def nav_phase(self, value: str) -> None:
        with self._lock:
            self._nav_phase = str(value)

    @property
    def preview_candidate(self) -> str:
        """Object ID currently highlighted in PREVIEW (empty = none)."""
        with self._lock:
            return self._preview_candidate

    @preview_candidate.setter
    def preview_candidate(self, value: str) -> None:
        with self._lock:
            self._preview_candidate = str(value)

    @property
    def focused_object(self) -> str:
        """Object ID locked in FOCUS (empty = none)."""
        with self._lock:
            return self._focused_object

    @focused_object.setter
    def focused_object(self, value: str) -> None:
        with self._lock:
            self._focused_object = str(value)

    # ------------------------------------------------------------------
    # current_gesture
    # ------------------------------------------------------------------

    @property
    def current_gesture(self) -> str:
        """Most-recently recognised gesture label (e.g. ``"ROTATE"``).

        Defaults to ``"NONE"`` when no gesture is active.
        """
        with self._lock:
            return self._current_gesture

    @current_gesture.setter
    def current_gesture(self, value: str) -> None:
        """Update the active gesture label.

        Args:
            value: Gesture name string.

        Raises:
            TypeError: If *value* is not a :class:`str`.
        """
        if not isinstance(value, str):
            raise TypeError(f"current_gesture must be a str, got {type(value).__name__!r}")
        with self._lock:
            self._current_gesture = value

    # ------------------------------------------------------------------
    # current_frame
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> np.ndarray | None:
        """Latest rendered frame as a ``(360, 18, 3)`` uint8 array, or ``None``."""
        with self._lock:
            return self._current_frame

    @current_frame.setter
    def current_frame(self, value: np.ndarray | None) -> None:
        """Replace the current frame.

        Only the Renderer thread should call this setter.

        Args:
            value: A NumPy array with shape ``(360, 18, 3)`` and dtype
                ``uint8``, or ``None`` to indicate that no frame is
                available yet.

        Raises:
            TypeError: If *value* is not a :class:`numpy.ndarray` or ``None``.
            ValueError: If the array shape or dtype does not match the
                required ``(360, 18, 3) uint8`` specification.
        """
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"current_frame must be a numpy.ndarray or None, got {type(value).__name__!r}"
                )
            if value.shape != _FRAME_SHAPE:
                raise ValueError(
                    f"current_frame must have shape {_FRAME_SHAPE}, got {value.shape}"
                )
            if value.dtype != _FRAME_DTYPE:
                raise ValueError(
                    f"current_frame must have dtype {_FRAME_DTYPE}, got {value.dtype}"
                )
        with self._lock:
            self._current_frame = value

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------

    @property
    def logs(self) -> list[str]:
        """Snapshot of the log deque as a plain list (oldest → newest)."""
        with self._lock:
            return list(self._logs)

    def append_log(self, message: str) -> None:
        """Append a timestamped log message to the ring buffer.

        Safe to call from any thread.

        Args:
            message: Human-readable log string.  A UTC timestamp is
                prepended automatically.
        """
        import datetime  # local import — stdlib, zero cost after first call

        if not isinstance(message, str):
            message = str(message)

        timestamp = datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self._logs.append(entry)

    # ------------------------------------------------------------------
    # scene_history
    # ------------------------------------------------------------------

    @property
    def scene_history(self) -> list[Dict[str, Any]]:
        """Snapshot of the scene-history deque as a plain list (oldest → newest)."""
        with self._lock:
            return list(self._scene_history)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_render_params(self) -> tuple[float, float, float, bool]:
        """Return all renderer-relevant transform parameters in one atomic read.

        Returns:
            ``(rotation_y, scale, explode, frozen)`` captured under a single
            lock acquisition to guarantee a consistent snapshot.
        """
        with self._lock:
            return self._rotation_y, self._scale, self._explode, self._frozen

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            frame_info = (
                f"shape={self._current_frame.shape}"
                if self._current_frame is not None
                else "None"
            )
            return (
                f"SceneState("
                f"gesture={self._current_gesture!r}, "
                f"rotation_y={self._rotation_y:.1f}, "
                f"scale={self._scale:.2f}, "
                f"explode={self._explode:.2f}, "
                f"frozen={self._frozen}, "
                f"frame={frame_info})"
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared blackboard instance imported by all subsystem modules.
scene_state = SceneState()
