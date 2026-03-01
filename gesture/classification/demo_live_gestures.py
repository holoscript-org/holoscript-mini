"""Live gesture detection demo.

Opens camera and detects gestures in real-time using MediaPipe and
the new GestureClassifier.
"""

import cv2
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Install with: pip install mediapipe")

from camera.camera_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from gesture_classifier import GestureClassifier
from debounce_manager import DebounceManager


class GestureDemoLive:
    """Live gesture detection demo."""
    
    def __init__(self):
        """Initialize demo."""
        self.classifier = GestureClassifier()
        self.debounce = DebounceManager(debounce_time=0.3)
        
        # MediaPipe hand landmarker
        self.hand_landmarker = None
        self.use_mediapipe = False
        
        if MEDIAPIPE_AVAILABLE:
            # Look for model file
            model_path = self._find_model()
            if model_path:
                try:
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
                    self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                    self.use_mediapipe = True
                    print(f"✓ MediaPipe initialized with model: {model_path}")
                except Exception as e:
                    print(f"✗ Failed to initialize MediaPipe: {e}")
            else:
                print("✗ hand_landmarker.task not found")
        
        # Camera
        self.cap = None
        self.frame_timestamp_ms = 0
        
        # Current gesture display
        self.current_gesture = "NONE"
        self.current_confidence = 0.0
        self.last_triggered_gesture = "NONE"
    
    def _find_model(self):
        """Find hand_landmarker.task model file."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "hand_landmarker.task",
            Path(__file__).parent.parent / "hand_landmarker.task",
            Path(__file__).parent / "hand_landmarker.task",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        return None
    
    def _initialize_camera(self):
        """Initialize camera."""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
    
    def _process_frame(self, frame):
        """Process frame and detect gestures."""
        if not self.use_mediapipe or not self.hand_landmarker:
            # Draw message if MediaPipe not available
            cv2.putText(frame, "MediaPipe not available", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.frame_timestamp_ms += 33
        
        try:
            # Detect hands
            result = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            
            if result and result.hand_landmarks:
                # Get first hand
                landmarks = result.hand_landmarks[0]
                
                # Draw landmarks
                self._draw_landmarks(frame, landmarks)
                
                # Classify gesture
                gesture, confidence = self.classifier.classify(landmarks)
                self.current_gesture = gesture
                self.current_confidence = confidence
                
                # Check if should trigger
                if self.debounce.should_trigger(gesture):
                    self.last_triggered_gesture = gesture
                    print(f"🎯 TRIGGERED: {gesture} (confidence: {confidence:.2f})")
            else:
                self.current_gesture = "NONE"
                self.current_confidence = 0.0
        
        except Exception as e:
            cv2.putText(frame, f"Error: {str(e)[:40]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        h, w, _ = frame.shape
        
        # Draw landmark points
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)  # Palm
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
    
    def _draw_ui(self, frame):
        """Draw UI overlay with gesture info."""
        h, w, _ = frame.shape
        
        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current gesture
        color = (0, 255, 0) if self.current_gesture != "NONE" and self.current_gesture != "UNKNOWN" else (100, 100, 100)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Confidence: {self.current_confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Last triggered
        cv2.putText(frame, f"Last Triggered: {self.last_triggered_gesture}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions at bottom
        cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gesture guide
        guide_text = "PINCH | V_SIGN | POINT | OPEN_PALM | FIST"
        cv2.putText(frame, guide_text, (w - 520, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run the demo."""
        print("=" * 60)
        print("Live Gesture Detection Demo")
        print("=" * 60)
        
        if not self.use_mediapipe:
            print("\n⚠️  MediaPipe not available!")
            print("Install with: pip install mediapipe")
            print("Download model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            return
        
        try:
            self._initialize_camera()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return
        
        print("\n✓ Starting gesture detection...")
        print("  Try these gestures:")
        print("    - PINCH: Thumb and index finger together")
        print("    - V_SIGN: Index and middle finger up")
        print("    - POINT: Only index finger up")
        print("    - OPEN_PALM: All fingers extended")
        print("    - FIST: All fingers closed")
        print("\n  Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self._process_frame(frame)
                
                # Draw UI
                self._draw_ui(frame)
                
                # Display
                cv2.imshow("Gesture Detection", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hand_landmarker:
            self.hand_landmarker.close()
        print("\n✓ Demo stopped")


if __name__ == "__main__":
    demo = GestureDemoLive()
    demo.run()
