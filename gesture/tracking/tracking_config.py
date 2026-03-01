"""Tracking configuration settings."""


class TrackingConfig:
    """Configuration for hand tracking parameters."""
    
    # Tracking settings
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    MODEL_COMPLEXITY = 1
