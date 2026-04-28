
"""
Live Gesture Detection Demo 

Pipeline:
Camera → MediaPipe → Landmarks → GestureClassifier → Debounce → UI Display

This script demonstrates real-time gesture recognition using:
• MediaPipe Hand Landmarker
• Custom GestureClassifier
• Debounce stability logic (prevents gesture flickering)

Gestures detected:
PINCH | V_SIGN | POINT | OPEN_PALM | FIST
"""

import cv2
import sys
import time
from pathlib import Path


# -------------------------------------------------------------
# Add project folders to Python path so imports work correctly
# -------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


# -------------------------------------------------------------
# Try importing MediaPipe Tasks API
# -------------------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not installed. Run: pip install mediapipe")


# -------------------------------------------------------------
# Import project modules
# -------------------------------------------------------------
from camera.camera_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from gesture_classifier import GestureClassifier


# =============================================================
# GestureDemoLive
# Main class running the live gesture demo
# =============================================================
class GestureDemoLive:

    def __init__(self):
        """
        Initialize gesture demo system.
        """

        # -----------------------------------------------------
        # Gesture classifier (detects gesture from landmarks)
        # -----------------------------------------------------
        self.classifier = GestureClassifier()

        # -----------------------------------------------------
        # Debounce variables (gesture stability control)
        # Prevents gesture flickering every frame
        # -----------------------------------------------------
        self.last_gesture = None           # gesture detected last frame
        self.gesture_start_time = 0        # when current gesture started
        self.stable_gesture = None         # confirmed gesture
        self.debounce_time = 0.5           # gesture must remain stable for 0.5 seconds

        # -----------------------------------------------------
        # Current display state
        # -----------------------------------------------------
        self.current_gesture = "NONE"
        self.current_confidence = 0.0
        self.last_triggered_gesture = "NONE"

        # -----------------------------------------------------
        # MediaPipe Hand Landmarker setup
        # -----------------------------------------------------
        self.hand_landmarker = None
        self.use_mediapipe = False

        if MEDIAPIPE_AVAILABLE:

            # Try locating the hand_landmarker.task model file
            model_path = self._find_model()

            if model_path:

                # Configure MediaPipe model
                base_options = mp_tasks.BaseOptions(
                    model_asset_path=model_path,
                    delegate=mp_tasks.BaseOptions.Delegate.CPU
                )

                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )

                # Create hand detection engine
                self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                self.use_mediapipe = True

                print("✓ MediaPipe initialized")

        # -----------------------------------------------------
        # Camera and frame timing
        # -----------------------------------------------------
        self.cap = None
        self.frame_timestamp_ms = 0


    # =============================================================
    # Locate MediaPipe model file
    # =============================================================
    def _find_model(self):

        possible_paths = [
            Path(__file__).parent.parent.parent / "hand_landmarker.task",
            Path(__file__).parent.parent / "hand_landmarker.task",
            Path(__file__).parent / "hand_landmarker.task",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None


    # =============================================================
    # Initialize webcam
    # =============================================================
    def _initialize_camera(self):

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("✓ Camera initialized")


    # =============================================================
    # Process a single frame
    # =============================================================
    def _process_frame(self, frame):

        # If MediaPipe not available → skip detection
        if not self.use_mediapipe:
            return frame

        # Convert BGR → RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Increase timestamp (~30 FPS)
        self.frame_timestamp_ms += 33

        try:

            # -------------------------------------------------
            # Run MediaPipe hand detection
            # -------------------------------------------------
            result = self.hand_landmarker.detect_for_video(
                mp_image, self.frame_timestamp_ms
            )

            # -------------------------------------------------
            # If hand detected
            # -------------------------------------------------
            if result and result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                # Draw landmarks on frame
                self._draw_landmarks(frame, landmarks)

                # -------------------------------------------------
                # Classify gesture using our GestureClassifier
                # -------------------------------------------------
                gesture, confidence = self.classifier.classify(landmarks)

                self.current_gesture = gesture
                self.current_confidence = confidence

                # -------------------------------------------------
                # Debounce logic (gesture stability)
                # Gesture must remain same for 0.5 seconds
                # -------------------------------------------------
                current_time = time.time()

                if gesture != self.last_gesture:

                    # New gesture detected → reset timer
                    self.last_gesture = gesture
                    self.gesture_start_time = current_time

                else:

                    # Same gesture continuing
                    if current_time - self.gesture_start_time > self.debounce_time:

                        # Confirm gesture if stable
                        if self.stable_gesture != gesture:

                            self.stable_gesture = gesture
                            self.last_triggered_gesture = gesture

                            print("\nGesture confirmed:", gesture)
                            print("Confidence:", round(confidence, 2))

            else:

                # No hand detected
                self.current_gesture = "NONE"
                self.current_confidence = 0.0

        except Exception as e:

            print("Detection error:", e)

        return frame


    # =============================================================
    # Draw hand landmarks
    # =============================================================
    def _draw_landmarks(self, frame, landmarks):

        h, w, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


    # =============================================================
    # Draw UI overlay
    # =============================================================
    def _draw_ui(self, frame):

        h, w, _ = frame.shape

        # Show confirmed gesture
        display = self.stable_gesture if self.stable_gesture else "STABILIZING"

        cv2.putText(
            frame,
            f"Gesture: {display}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {self.current_confidence:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Last Triggered: {self.last_triggered_gesture}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press q to quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


    # =============================================================
    # Main program loop
    # =============================================================
    def run(self):

        print("\nLive Gesture Detection Demo\n")

        if not self.use_mediapipe:
            print("MediaPipe not available")
            return

        self._initialize_camera()

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            # Mirror image for natural interaction
            frame = cv2.flip(frame, 1)

            # Run gesture detection pipeline
            frame = self._process_frame(frame)

            # Draw UI information
            self._draw_ui(frame)

            # Show window
            cv2.imshow("Gesture Detection", frame)

            # Exit when 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup resources
        self.cap.release()
        cv2.destroyAllWindows()


# =============================================================
# Program entry point
# =============================================================
if __name__ == "__main__":

    demo = GestureDemoLive()
    demo.run()
