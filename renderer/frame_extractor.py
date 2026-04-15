"""renderer/frame_extractor.py
Reads the OpenGL framebuffer into a numpy array each frame.
No pyglet. No SceneState. Pure GL + numpy + PIL.
"""

from __future__ import annotations

import os

import numpy as np
from OpenGL.GL import glReadPixels, glReadBuffer, GL_BACK, GL_RGB, GL_UNSIGNED_BYTE
from PIL import Image


class FrameExtractor:

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._saved_test_frame: bool = False
        self._test_output_path: str = "renderer/assets/test_frame.png"

    def extract(self) -> np.ndarray:
        """Read the current OpenGL framebuffer into a (height, width, 3) uint8 array.

        Steps:
          1. glReadPixels → raw bytes
          2. np.frombuffer + reshape to (height, width, 3)
          3. np.flipud to correct OpenGL's bottom-left origin
          4. np.ascontiguousarray to ensure the flip produces a real copy

        Returns a zero-filled array of the correct shape/dtype on any GL error.
        """
        try:
            # In a double-buffered context, on_draw renders into the back buffer.
            # Read explicitly from GL_BACK to avoid intermittent black captures.
            glReadBuffer(GL_BACK)
            raw = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(raw, dtype=np.uint8)
            image = image.reshape(self.height, self.width, 3)
            image = np.flipud(image)
            image = np.ascontiguousarray(image)
            return image
        except Exception as e:
            print(f"[FrameExtractor] WARNING: glReadPixels failed ({type(e).__name__}: {e}). "
                  f"Returning blank frame.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def save_test_frame(self, image: np.ndarray) -> None:
        """Save image to disk as PNG exactly once.

        Subsequent calls are silently ignored once _saved_test_frame is True.
        Reset _saved_test_frame to False externally to force a re-save.
        """
        if self._saved_test_frame:
            return

        try:
            os.makedirs(os.path.dirname(self._test_output_path), exist_ok=True)
            Image.fromarray(image).save(self._test_output_path)
            print(f"[FrameExtractor] Test frame saved to {self._test_output_path}")
            self._saved_test_frame = True
        except Exception as e:
            print(f"[FrameExtractor] WARNING: could not save test frame: {e}")
