"""
renderer/scene_state.py
=======================
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
- current_frame   → written by Renderer thread  (shape (360,18,3) uint8)
- logs            → appended by any thread
- scene_history   → appended by Voice/LLM thread

All threads may read any field.  Every read and write is serialised through
a private :class:`threading.RLock`, making individual attribute accesses
atomic.  The lock is reentrant so a single thread may perform multiple
getter/setter calls within one logical operation without deadlocking.
"""

from __future__ import annotations

import datetime
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

    All public attributes are exposed as Python *properties* whose getters
    and setters acquire the internal :class:`threading.RLock`, ensuring that
    every individual read or write is atomic with respect to cross-thread
    visibility.

    Use :meth:`get_render_params` when the Renderer thread needs a consistent
    snapshot of all transform parameters in a single lock acquisition.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        # Voice / LLM -------------------------------------------------
        self._scene_json: Dict[str, Any] | None = None
        self._transcript: str = ""

        # Gesture -----------------------------------------------------
        self._rotation_y: float = 0.0
        self._scale: float = 1.0
        self._explode: float = 0.0
        self._frozen: bool = False
        self._current_gesture: str = "NONE"

        # Renderer output ---------------------------------------------
        self._current_frame: np.ndarray | None = None

        # Audit / history ---------------------------------------------
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
            raise TypeError(
                f"scene_json must be a dict or None, got {type(value).__name__!r}"
            )
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
            raise TypeError(
                f"transcript must be a str, got {type(value).__name__!r}"
            )
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
            value: Rotation angle in degrees (coerced to ``float``).
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
            value: Positive scale multiplier (coerced to ``float``).

        Raises:
            ValueError: If *value* is not strictly positive.
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
        """Explode factor in ``[0.0, 1.0]`` for exploded-view rendering."""
        with self._lock:
            return self._explode

    @explode.setter
    def explode(self, value: float) -> None:
        """Set the explode factor.

        Values outside ``[0.0, 1.0]`` are clamped silently.

        Args:
            value: Desired explode magnitude (coerced to ``float``).
        """
        with self._lock:
            self._explode = max(0.0, min(1.0, float(value)))

    # ------------------------------------------------------------------
    # frozen
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        """Whether animation is paused; set by the Gesture thread."""
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
            raise TypeError(
                f"current_gesture must be a str, got {type(value).__name__!r}"
            )
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
        """Replace the current rendered frame.

        Only the Renderer thread should call this setter.

        Args:
            value: A :class:`numpy.ndarray` with shape ``(360, 18, 3)`` and
                dtype ``uint8``, or ``None`` to indicate no frame is available.

        Raises:
            TypeError: If *value* is not a :class:`numpy.ndarray` or ``None``.
            ValueError: If the array shape or dtype does not match
                ``(360, 18, 3) uint8``.
        """
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"current_frame must be a numpy.ndarray or None, "
                    f"got {type(value).__name__!r}"
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
    # logs  (read-only property + append method)
    # ------------------------------------------------------------------

    @property
    def logs(self) -> list[str]:
        """Snapshot of the log ring-buffer as a plain list (oldest → newest)."""
        with self._lock:
            return list(self._logs)

    def append_log(self, message: str) -> None:
        """Append a timestamped message to the log ring-buffer.

        Safe to call from any thread.  The UTC wall-clock time is prepended
        automatically in ``HH:MM:SS.mmm`` format.

        Args:
            message: Human-readable log string.  Non-string values are
                coerced via :func:`str`.
        """
        if not isinstance(message, str):
            message = str(message)
        timestamp = datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self._logs.append(entry)

    # ------------------------------------------------------------------
    # scene_history  (read-only)
    # ------------------------------------------------------------------

    @property
    def scene_history(self) -> list[Dict[str, Any]]:
        """Snapshot of the scene-history ring-buffer (oldest → newest)."""
        with self._lock:
            return list(self._scene_history)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_render_params(self) -> tuple[float, float, float, bool]:
        """Return all renderer transform parameters in one atomic read.

        Acquires the lock once and returns a consistent snapshot of the four
        transform fields so the Renderer thread cannot observe a torn state.

        Returns:
            ``(rotation_y, scale, explode, frozen)``
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
scene_state: SceneState = SceneState()
