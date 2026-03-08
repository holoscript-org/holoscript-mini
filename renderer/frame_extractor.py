"""
renderer/frame_extractor.py
============================
Efficient extraction of the active OpenGL framebuffer into a NumPy array.

Pipeline position
-----------------
::

    OpenGL back-buffer  →  FrameExtractor.extract()  →  (H, W, 3) uint8
                                                              ↓
                                              cylindrical_projection (next stage)

Design
------
The implementation minimises memory allocations and data copies:

1. A persistent ``numpy.ndarray`` of shape ``(H × W × 3,)`` is kept as an
   instance attribute and reused every frame.  It is only reallocated when
   the requested dimensions change.
2. ``glReadPixels`` writes directly into the NumPy buffer (PyOpenGL accepts an
   ``ndarray`` as the pixel-data argument), so **no intermediate ctypes copy**
   is performed.
3. Vertical flip is implemented with ``numpy.flipud``, which returns a *view*,
   followed by a single ``numpy.ascontiguousarray`` call that produces the
   final owned, C-contiguous output array.

Total cost per frame: **one GL read + one NumPy copy** (for the flip).

Module-level helper
-------------------
A convenience function :func:`extract_frame` is provided for callers that do
not need the buffer-reuse optimisation::

    frame = extract_frame(800, 800)   # (800, 800, 3) uint8, origin top-left

Class-based interface
---------------------
Use :class:`FrameExtractor` when you call from the same thread every frame and
want to avoid repeated allocations::

    extractor = FrameExtractor()
    frame = extractor.extract(800, 800)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from OpenGL.GL import (
    GL_RGB,
    GL_UNSIGNED_BYTE,
    glReadPixels,
)


# ---------------------------------------------------------------------------
# FrameExtractor
# ---------------------------------------------------------------------------


class FrameExtractor:
    """Stateful framebuffer extractor with a persistent reusable read buffer.

    The internal buffer is allocated lazily on the first call to
    :meth:`extract` and reallocated only when the requested ``(width, height)``
    dimensions differ from the previous call.

    This class is **not thread-safe**: it should be owned and called
    exclusively by the renderer thread.
    """

    def __init__(self) -> None:
        # Flat buffer reused every frame to avoid per-frame heap allocation.
        self._buf: Optional[np.ndarray] = None
        # Dimensions the current buffer was sized for.
        self._buf_dims: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------

    def extract(self, width: int, height: int) -> np.ndarray:
        """Read the current OpenGL framebuffer into a ``(H, W, 3) uint8`` array.

        The OpenGL origin is at the **bottom-left** corner; this method
        corrects that by flipping the row order so that row ``0`` of the
        returned array corresponds to the **top** of the rendered image,
        matching the convention used by NumPy, Pillow, and most image formats.

        Args:
            width:  Framebuffer width in pixels.  Must be a positive integer.
            height: Framebuffer height in pixels.  Must be a positive integer.

        Returns:
            A C-contiguous ``numpy.ndarray`` of shape ``(height, width, 3)``
            and dtype ``uint8`` containing the RGB pixel data, top-row first.

        Raises:
            ValueError: If *width* or *height* is not a positive integer.
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive integers, "
                f"got width={width}, height={height}"
            )

        # ── Step 1: ensure buffer is the right size ───────────────────────
        if self._buf is None or self._buf_dims != (width, height):
            self._buf = np.empty(height * width * 3, dtype=np.uint8)
            self._buf_dims = (width, height)

        # ── Step 2: read framebuffer directly into the NumPy buffer ──────
        # PyOpenGL accepts an ndarray as the pixel-data argument and writes
        # OpenGL's raw byte stream into it without an intermediate ctypes copy.
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, self._buf)

        # ── Step 3: reshape to (H, W, 3) — zero-copy view ────────────────
        shaped: np.ndarray = self._buf.reshape(height, width, 3)

        # ── Step 4: flip vertically (GL bottom-left → top-left) ──────────
        # np.flipud returns a view; ascontiguousarray makes one owned copy.
        return np.ascontiguousarray(np.flipud(shaped))


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def extract_frame(width: int, height: int) -> np.ndarray:
    """Read the current OpenGL framebuffer into a ``(H, W, 3) uint8`` array.

    This is a stateless convenience wrapper around :class:`FrameExtractor`.
    It allocates a fresh buffer on every call.  For render-loop use where
    efficiency matters, create a single :class:`FrameExtractor` instance
    instead.

    Args:
        width:  Framebuffer width in pixels.
        height: Framebuffer height in pixels.

    Returns:
        C-contiguous ``ndarray`` of shape ``(height, width, 3)``, dtype
        ``uint8``, top-row first (OpenGL origin corrected).

    Raises:
        ValueError: If *width* or *height* is not positive.
    """
    return FrameExtractor().extract(width, height)
