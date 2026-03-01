"""Gesture classifier for MediaPipe hand landmarks.

Provides robust gesture classification using proper landmark comparisons.
Detects: OPEN_PALM, FIST, PINCH, POINT, V_SIGN, UNKNOWN.
"""

import math
from typing import Tuple, List, Any


class GestureClassifier:
    """Classifies hand gestures from MediaPipe landmarks.
    
    Uses proper finger state detection based on TIP vs PIP comparisons
    for fingers and TIP vs IP x-coordinate for thumb.
    """
    
    def __init__(self):
        """Initialize gesture classifier."""
        # MediaPipe landmark indices
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.RING_TIP = 16
        self.RING_PIP = 14
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        
        # Detection thresholds
        self.PINCH_THRESHOLD = 0.05
    
    def distance(self, p1: Any, p2: Any) -> float:
        """Calculate Euclidean distance between two landmarks.
        
        Args:
            p1: First landmark with .x, .y, .z attributes
            p2: Second landmark with .x, .y, .z attributes
            
        Returns:
            Euclidean distance in normalized coordinates
        """
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = getattr(p1, 'z', 0.0) - getattr(p2, 'z', 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def is_finger_open(self, landmarks: List[Any], tip_index: int, pip_index: int) -> bool:
        """Check if a finger is open (extended).
        
        A finger is considered open if its tip y-coordinate is less than
        its PIP y-coordinate (tip is above PIP in image space).
        
        Args:
            landmarks: List of 21 hand landmarks
            tip_index: Index of fingertip landmark
            pip_index: Index of PIP joint landmark
            
        Returns:
            True if finger is open/extended, False if curled
        """
        tip = landmarks[tip_index]
        pip = landmarks[pip_index]
        # In image coordinates, y increases downward
        # Finger is open if tip is above (lower y value) PIP
        return tip.y < pip.y
    
    def is_thumb_open(self, landmarks: List[Any]) -> bool:
        """Check if thumb is open (extended).
        
        Assumes right hand. Thumb is open if tip x-coordinate is greater
        than IP x-coordinate (tip is to the right of IP).
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            True if thumb is open/extended, False if curled
        """
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        # For right hand, thumb open means tip.x > ip.x
        return thumb_tip.x > thumb_ip.x
    
    def classify(self, landmarks: List[Any]) -> Tuple[str, float]:
        """Classify hand gesture from landmarks.
        
        Detection priority: PINCH -> V_SIGN -> POINT -> OPEN_PALM -> FIST -> UNKNOWN
        
        Args:
            landmarks: List of 21 MediaPipe hand landmarks with .x, .y, .z attributes
            
        Returns:
            Tuple of (gesture_name, confidence)
            gesture_name: One of OPEN_PALM, FIST, PINCH, POINT, V_SIGN, UNKNOWN
            confidence: Float between 0.0 and 1.0
        """
        if not landmarks or len(landmarks) < 21:
            return ("UNKNOWN", 0.0)
        
        # Get finger states
        thumb_open = self.is_thumb_open(landmarks)
        index_open = self.is_finger_open(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_open = self.is_finger_open(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_open = self.is_finger_open(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_open = self.is_finger_open(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        # Count open fingers
        open_count = sum([index_open, middle_open, ring_open, pinky_open])
        
        # Priority 1: PINCH
        # Thumb tip close to index tip
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        pinch_distance = self.distance(thumb_tip, index_tip)
        
        if pinch_distance < self.PINCH_THRESHOLD:
            return ("PINCH", 0.9)
        
        # Priority 2: V_SIGN
        # Index and middle open, ring and pinky closed
        if index_open and middle_open and not ring_open and not pinky_open:
            return ("V_SIGN", 0.9)
        
        # Priority 3: POINT
        # Only index finger open
        if index_open and not middle_open and not ring_open and not pinky_open:
            return ("POINT", 0.9)
        
        # Priority 4: OPEN_PALM
        # All fingers open
        if open_count >= 4:
            return ("OPEN_PALM", 0.9)
        
        # Priority 5: FIST
        # All fingers closed
        if open_count == 0:
            return ("FIST", 0.9)
        
        # Priority 6: UNKNOWN
        # Partial configurations not matching specific gestures
        return ("UNKNOWN", 0.0)
