"""Debug logger for gesture system."""
import logging
from datetime import datetime


class DebugLogger:
    """Handles logging for debugging gesture recognition."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize debug logger."""
        self.logger = logging.getLogger("GestureDebug")
        self.logger.setLevel(log_level)
    
    def log_landmarks(self, landmarks):
        """Log landmark data."""
        pass
    
    def log_gesture(self, gesture_name, confidence):
        """Log detected gesture."""
        pass
