"""Debounce manager for gesture triggering.

Prevents rapid repeated triggering of the same gesture by enforcing
a minimum time interval between triggers of identical gestures.
"""

import time
from typing import Optional


class DebounceManager:
    """Manages gesture debouncing to prevent rapid triggering.
    
    Allows immediate triggering when gesture changes, but enforces
    a cooldown period for repeated identical gestures.
    """
    
    def __init__(self, debounce_time: float = 0.5):
        """Initialize debounce manager.
        
        Args:
            debounce_time: Minimum seconds between triggers of same gesture
        """
        self.debounce_time = debounce_time
        self.last_gesture: Optional[str] = None
        self.last_trigger_time: float = 0.0
    
    def should_trigger(self, gesture: str) -> bool:
        """Check if gesture should trigger based on debounce rules.
        
        Rules:
        - UNKNOWN gestures never trigger
        - New gesture (different from last) triggers immediately
        - Same gesture only triggers after debounce_time has elapsed
        
        Args:
            gesture: Current gesture name
            
        Returns:
            True if gesture should trigger, False otherwise
        """
        # Never trigger UNKNOWN
        if gesture == "UNKNOWN":
            return False
        
        current_time = time.time()
        
        # Gesture changed - trigger immediately
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.last_trigger_time = current_time
            return True
        
        # Same gesture - check if enough time has passed
        time_since_last = current_time - self.last_trigger_time
        
        if time_since_last > self.debounce_time:
            self.last_trigger_time = current_time
            return True
        
        # Too soon - don't trigger
        return False
