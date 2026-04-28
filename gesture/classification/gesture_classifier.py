
"""
Gesture classifier for MediaPipe hand landmarks.

Uses simple finger-state logic to detect gestures.

Supported gestures:
OPEN_PALM
FIST
PINCH
POINT
V_SIGN
UNKNOWN
"""

import math


class GestureClassifier:
    """Classifies hand gestures from MediaPipe landmarks."""

    def __init__(self):

        # -----------------------------
        # MediaPipe landmark indices
        # -----------------------------
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

        # -----------------------------
        # PINCH detection threshold
        # -----------------------------
        self.PINCH_THRESHOLD = 0.05


    # ---------------------------------------------------------
    # Distance between two landmarks
    # ---------------------------------------------------------
    def distance(self, p1, p2):

        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = getattr(p1, "z", 0.0) - getattr(p2, "z", 0.0)

        return math.sqrt(dx * dx + dy * dy + dz * dz)


    # ---------------------------------------------------------
    # Check if finger is open
    # ---------------------------------------------------------
    def is_finger_open(self, landmarks, tip_index, pip_index):

        tip = landmarks[tip_index]
        pip = landmarks[pip_index]

        # Finger is open if tip is above PIP joint
        return tip.y < pip.y


    # ---------------------------------------------------------
    # Check if thumb is open
    # ---------------------------------------------------------
    def is_thumb_open(self, landmarks):

        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]

        # Works for right hand
        return thumb_tip.x > thumb_ip.x


    # ---------------------------------------------------------
    # Main gesture classification
    # ---------------------------------------------------------
    def classify(self, landmarks):

        if not landmarks or len(landmarks) < 21:
            return ("UNKNOWN", 0.0)

        # -----------------------------
        # Detect finger states
        # -----------------------------
        thumb_open = self.is_thumb_open(landmarks)

        index_open = self.is_finger_open(
            landmarks,
            self.INDEX_TIP,
            self.INDEX_PIP,
        )

        middle_open = self.is_finger_open(
            landmarks,
            self.MIDDLE_TIP,
            self.MIDDLE_PIP,
        )

        ring_open = self.is_finger_open(
            landmarks,
            self.RING_TIP,
            self.RING_PIP,
        )

        pinky_open = self.is_finger_open(
            landmarks,
            self.PINKY_TIP,
            self.PINKY_PIP,
        )

        open_count = sum(
            [
                index_open,
                middle_open,
                ring_open,
                pinky_open,
            ]
        )

        # -----------------------------
        # PINCH detection
        # -----------------------------
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]

        pinch_distance = self.distance(
            thumb_tip,
            index_tip,
        )

        if pinch_distance < self.PINCH_THRESHOLD:
            return ("PINCH", 0.9)

        # -----------------------------
        # V SIGN
        # -----------------------------
        if (
            index_open
            and middle_open
            and not ring_open
            and not pinky_open
        ):
            return ("V_SIGN", 0.9)

        # -----------------------------
        # POINT gesture
        # -----------------------------
        if (
            index_open
            and not middle_open
            and not ring_open
            and not pinky_open
        ):
            return ("POINT", 0.9)

        # -----------------------------
        # OPEN PALM
        # -----------------------------
        if open_count >= 4:
            return ("OPEN_PALM", 0.9)

        # -----------------------------
        # FIST
        # -----------------------------
        if open_count == 0:
            return ("FIST", 0.9)

        # -----------------------------
        # UNKNOWN
        # -----------------------------
        return ("UNKNOWN", 0.0)

