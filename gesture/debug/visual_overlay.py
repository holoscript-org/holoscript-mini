"""Visual overlay for debugging gesture tracking."""


class VisualOverlay:
    """Provides visual debugging overlays on camera feed."""
    
    def __init__(self):
        """Initialize visual overlay."""
        pass
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        pass
    
    def draw_gesture_info(self, frame, gesture_name, confidence):
        """Draw gesture information on frame."""
        pass
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter on frame."""
        pass
