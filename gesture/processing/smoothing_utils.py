"""Smoothing utilities for landmark data."""


class LandmarkSmoother:
    """Applies smoothing to landmark data to reduce jitter."""
    
    def __init__(self, window_size=5):
        """Initialize smoother with window size."""
        self.window_size = window_size
        self.history = []
    
    def smooth(self, landmarks):
        """Apply smoothing to landmarks."""
        pass
    
    def reset(self):
        """Reset smoothing history."""
        self.history = []
