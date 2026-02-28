from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class SceneState:
    # === Voice / LLM ===
    transcript: str = ""
    intent: str = ""
    scene_json: Dict[str, Any] = field(default_factory=dict)

    # === Gesture ===
    current_gesture: str = ""
    gesture_confidence: float = 0.0

    # === Rendering ===
    objects: Dict[str, Any] = field(default_factory=dict)
    rotation_y: float = 0.0
    scale: float = 1.0
    frozen: bool = False

    # === Projection Output ===
    current_frame: np.ndarray | None = None


# Global blackboard instance
scene_state = SceneState()
