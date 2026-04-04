import json
from collections import deque

HISTORY_MAXLEN = 5
FULL_DETAIL_COUNT = 3


class ContextManager:
    def __init__(self) -> None:
        self._history: deque[tuple[str, dict]] = deque(maxlen=HISTORY_MAXLEN)

    def add(self, command: str, scene: dict) -> None:
        self._history.append((command, scene))

    def build_refinement_prompt(self, new_command: str) -> str:
        history = list(self._history)
        compress_up_to = max(0, len(history) - FULL_DETAIL_COUNT)

        lines = ["Previous conversation history:"]

        for i, (cmd, scene) in enumerate(history):
            if i < compress_up_to:
                object_count = len(scene.get("objects", []))
                lines.append(f"  [{i + 1}] Command: \"{cmd}\" -> scene with {object_count} object(s) (compressed)")
            else:
                lines.append(f"  [{i + 1}] Command: \"{cmd}\"")
                lines.append(f"       Scene: {json.dumps(scene)}")

        lines.append(f"\nNew user command: {new_command}")
        lines.append("\nApply the new command to the most recent scene. Output ONLY the modified scene as raw JSON. No markdown. No explanations.")

        return "\n".join(lines)

    def last_scene(self) -> dict | None:
        if not self._history:
            return None
        return self._history[-1][1]
