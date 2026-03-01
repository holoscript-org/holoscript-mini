"""Gesture Engine for AI hologram system.

Provides real-time hand pose tracking using MediaPipe Tasks Vision API.
Feature extraction layer: palm-size normalisation, exponentially-smoothed
pinch strength, per-finger curl scores, stable hand centre with motion
deltas, and wrist orientation vector — all continuous, no state machine.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from .camera.camera_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError:
    MEDIAPIPE_TASKS_AVAILABLE = False

# Resolve model path relative to this file, then project root as fallback
_MODEL_FILENAME = "hand_landmarker.task"
_MODULE_DIR = Path(__file__).parent
_MODEL_PATH = next(
    (str(p) for p in [_MODULE_DIR / _MODEL_FILENAME, _MODULE_DIR.parent / _MODEL_FILENAME]
     if p.exists()),
    None
)


class GestureEngine:
    """Real-time gesture recognition engine using MediaPipe Tasks Vision API."""
    
    def __init__(self):
        """Initialize the gesture engine."""
        # MediaPipe setup - use new Tasks API
        self.hand_landmarker = None
        self.use_tasks_api = False
        
        if MEDIAPIPE_TASKS_AVAILABLE and _MODEL_PATH:
            try:
                base_options = mp_tasks.BaseOptions(
                    model_asset_path=_MODEL_PATH,
                    delegate=mp_tasks.BaseOptions.Delegate.CPU
                )
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api = True
                print(f"Using MediaPipe Tasks Vision API  [{_MODEL_PATH}]")
            except Exception as e:
                print(f"Failed to initialize MediaPipe Tasks API: {e}")
                self.use_tasks_api = False
        elif not _MODEL_PATH:
            print(f"Model '{_MODEL_FILENAME}' not found — using fallback detection")
        
        if not self.use_tasks_api:
            print("Using fallback hand detection")
        
        # Camera setup
        self.cap = None
        self.running = False
        self.thread = None
        self.thread_lock = threading.Lock()
        
        # Gesture detection (kept for update_state / get_current_gesture compat)
        self.current_gesture = None
        self.gesture_confidence = 0.0

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()

        # Frame timestamp for Tasks API
        self.frame_timestamp_ms = 0

        # Landmark-level exponential smoothing
        self.prev_landmarks = None
        self.smoothing_alpha = 0.7
        self._current_pose = None

        # Per-feature smoothing state (reset on hand loss)
        self._prev_pinch: Optional[float] = None
        self._prev_avg_curl: Optional[float] = None
        self._prev_center: Optional[Tuple] = None
    
    def _convert_mediapipe_landmarks(self, landmarks_list) -> List:
        """Convert MediaPipe Tasks landmarks to legacy format for compatibility."""
        if not landmarks_list or not landmarks_list.hand_landmarks:
            return None
        
        # Get first hand landmarks
        hand_landmarks = landmarks_list.hand_landmarks[0]
        
        # Convert to legacy format (simple objects with x, y, z attributes)
        converted_landmarks = []
        for landmark in hand_landmarks:
            class LegacyLandmark:
                def __init__(self, x, y, z=0.0):
                    self.x = x
                    self.y = y
                    self.z = z
            converted_landmarks.append(LegacyLandmark(landmark.x, landmark.y, getattr(landmark, 'z', 0.0)))
        
        return converted_landmarks
    
    def _smooth_landmarks(self, current_landmarks):
        """Apply exponential smoothing across all 21 landmarks.

        Reduces jitter while preserving real-time responsiveness (~30 FPS).
        alpha=0.7 weights current frame heavily; lower values = more smoothing.
        """
        if (self.prev_landmarks is None
                or len(self.prev_landmarks) != len(current_landmarks)):
            self.prev_landmarks = current_landmarks
            return current_landmarks

        alpha = self.smoothing_alpha
        smoothed = []
        for curr, prev in zip(current_landmarks, self.prev_landmarks):
            class _SmoothedLM:
                pass
            lm = _SmoothedLM()
            lm.x = alpha * curr.x + (1.0 - alpha) * prev.x
            lm.y = alpha * curr.y + (1.0 - alpha) * prev.y
            lm.z = alpha * getattr(curr, 'z', 0.0) + (1.0 - alpha) * getattr(prev, 'z', 0.0)
            smoothed.append(lm)

        self.prev_landmarks = smoothed
        return smoothed

    def _initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialized: {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def _normalize_landmarks(self, landmarks):
        """Translate landmarks to wrist origin and scale by palm size.

        Removes position and scale variation so features are hand-size invariant.
        Wrist (landmark 0) becomes (0, 0, 0) after translation.
        Palm size = distance from wrist to middle MCP (landmark 9).
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size = np.sqrt(
            (middle_mcp.x - wrist.x) ** 2
            + (middle_mcp.y - wrist.y) ** 2
            + (getattr(middle_mcp, 'z', 0.0) - getattr(wrist, 'z', 0.0)) ** 2
        )
        if palm_size < 1e-6:
            palm_size = 1e-6  # guard against degenerate frames

        norm = []
        for lm in landmarks:
            class _NormLM:
                pass
            n = _NormLM()
            n.x = (lm.x - wrist.x) / palm_size
            n.y = (lm.y - wrist.y) / palm_size
            n.z = (getattr(lm, 'z', 0.0) - getattr(wrist, 'z', 0.0)) / palm_size
            norm.append(n)
        return norm

    def _extract_features(self, norm_landmarks) -> Dict[str, float]:
        """Compute continuously-smoothed hand features from palm-normalised landmarks.

        In normalised space the wrist is at origin, so magnitude of a tip
        vector equals its distance from the wrist.

        Pinch smoothing:    pinch  = 0.7 * prev  + 0.3 * current
        Curl  smoothing:    curl   = 0.7 * prev  + 0.3 * current
        pinch_active        pinch_strength > 0.75
        fist_active         avg_curl > 0.85
        """
        def _mag(lm):
            return np.sqrt(lm.x ** 2 + lm.y ** 2 + lm.z ** 2)

        def _dist(a, b):
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

        # --- Pinch ---
        raw_dist    = _dist(norm_landmarks[4], norm_landmarks[8])
        pinch_curr  = 1.0 - float(np.clip(raw_dist, 0.0, 1.0))
        if self._prev_pinch is None:
            self._prev_pinch = pinch_curr
        pinch_strength    = 0.7 * self._prev_pinch + 0.3 * pinch_curr
        self._prev_pinch  = pinch_strength
        pinch_active      = pinch_strength > 0.75

        # --- Per-finger curl (tip magnitude from wrist origin) ---
        index_curl  = float(np.clip(1.0 - _mag(norm_landmarks[8]),  0.0, 1.0))
        middle_curl = float(np.clip(1.0 - _mag(norm_landmarks[12]), 0.0, 1.0))
        ring_curl   = float(np.clip(1.0 - _mag(norm_landmarks[16]), 0.0, 1.0))
        pinky_curl  = float(np.clip(1.0 - _mag(norm_landmarks[20]), 0.0, 1.0))

        avg_curl_curr = (index_curl + middle_curl + ring_curl + pinky_curl) / 4.0
        if self._prev_avg_curl is None:
            self._prev_avg_curl = avg_curl_curr
        avg_curl           = 0.7 * self._prev_avg_curl + 0.3 * avg_curl_curr
        self._prev_avg_curl = avg_curl
        fist_active        = avg_curl > 0.85

        return {
            "pinch_strength": pinch_strength,
            "pinch_active":   pinch_active,
            "avg_curl":       avg_curl,
            "fist_active":    fist_active,
            "index_curl":     index_curl,
            "middle_curl":    middle_curl,
            "ring_curl":      ring_curl,
            "pinky_curl":     pinky_curl,
        }
    
    def _compute_and_store_pose(
        self,
        smoothed_landmarks,
        features: Dict[str, float],
        confidence: float,
    ):
        """Compute stable hand centre, motion deltas, palm orientation, then store.

        Centre is averaged over the 5 palm-base landmarks (wrist + 4 MCPs) in
        image space so it is robust to individual finger movement.
        Smoothing:  center = 0.6 * prev_center + 0.4 * current
        Deltas are the frame-to-frame displacement of the smoothed centre.
        """
        if len(smoothed_landmarks) < 21:
            return

        # --- Stable centre (palm base landmarks in image space) ---
        _PALM = [0, 5, 9, 13, 17]
        cx_curr = sum(smoothed_landmarks[i].x for i in _PALM) / len(_PALM)
        cy_curr = sum(smoothed_landmarks[i].y for i in _PALM) / len(_PALM)
        cz_curr = sum(getattr(smoothed_landmarks[i], 'z', 0.0) for i in _PALM) / len(_PALM)

        if self._prev_center is None:
            self._prev_center = (cx_curr, cy_curr, cz_curr)

        cx = 0.6 * self._prev_center[0] + 0.4 * cx_curr
        cy = 0.6 * self._prev_center[1] + 0.4 * cy_curr
        cz = 0.6 * self._prev_center[2] + 0.4 * cz_curr

        dx = cx - self._prev_center[0]
        dy = cy - self._prev_center[1]
        dz = cz - self._prev_center[2]
        self._prev_center = (cx, cy, cz)

        # --- Palm direction: wrist → middle MCP, normalised ---
        wrist      = smoothed_landmarks[0]
        middle_mcp = smoothed_landmarks[9]
        vx = middle_mcp.x - wrist.x
        vy = middle_mcp.y - wrist.y
        vz = getattr(middle_mcp, 'z', 0.0) - getattr(wrist, 'z', 0.0)
        vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
        if vmag > 1e-6:
            vx, vy, vz = vx / vmag, vy / vmag, vz / vmag

        pose = {
            "center":         (cx, cy, cz),
            "dx":             dx,
            "dy":             dy,
            "dz":             dz,
            "pinch_strength": features["pinch_strength"],
            "pinch_active":   features["pinch_active"],
            "avg_curl":       features["avg_curl"],
            "fist_active":    features["fist_active"],
            "palm_direction": (vx, vy, vz),
            "confidence":     confidence,
        }

        label = (
            "FIST"     if features["fist_active"]  else
            "GRABBING" if features["pinch_active"] else
            "OPEN_HAND"
        )

        with self.thread_lock:
            self._current_pose     = pose
            self.current_gesture   = label
            self.gesture_confidence = confidence
    
    def get_current_gesture(self) -> Dict[str, Any]:
        """Get current stable gesture.
        
        Returns:
            Dict with 'gesture' and 'confidence' keys
        """
        with self.thread_lock:
            return {
                "gesture": self.current_gesture or "NONE",
                "confidence": self.gesture_confidence
            }

    def get_hand_pose(self) -> Optional[Dict[str, Any]]:
        """Return continuously-updated hand pose data for 3D manipulation.

        Updated every frame (~30 FPS) with no debounce or hold delay.
        Thread-safe; may be called from any thread while the engine is running.

        Returns:
            dict::

                {
                    "center":         (x, y, z)  smoothed palm-base centre (0–1 image space),
                    "dx":             float — frame-to-frame x displacement of centre,
                    "dy":             float — frame-to-frame y displacement,
                    "dz":             float — frame-to-frame z displacement,
                    "pinch_strength": float [0, 1]  — 1.0 = full pinch (smoothed),
                    "pinch_active":   bool  — pinch_strength > 0.75,
                    "avg_curl":       float [0, 1]  — mean 4-finger curl (smoothed),
                    "fist_active":    bool  — avg_curl > 0.85,
                    "palm_direction": (vx, vy, vz)  unit vector wrist → middle MCP,
                    "confidence":     float — detection confidence,
                }

            None if no hand is currently detected.
        """
        with self.thread_lock:
            return self._current_pose
    
    def update_state(self, state_object):
        """Update external state object with current gesture data.
        
        Args:
            state_object: Object with current_gesture and gesture_confidence attributes
        """
        with self.thread_lock:
            if hasattr(state_object, 'current_gesture'):
                state_object.current_gesture = self.current_gesture
            if hasattr(state_object, 'gesture_confidence'):
                state_object.gesture_confidence = self.gesture_confidence
    
    def _process_frame(self, frame):
        """Process a single frame for gesture detection."""
        if self.use_tasks_api and self.hand_landmarker:
            return self._process_frame_tasks_api(frame)
        else:
            return self._process_frame_fallback(frame)
    
    def _process_frame_tasks_api(self, frame):
        """Process frame using MediaPipe Tasks Vision API (VIDEO mode)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.frame_timestamp_ms += 33  # monotonic ~30 FPS increment
        
        try:
            hand_result = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            
            if hand_result and hand_result.hand_landmarks:
                self._draw_landmarks_on_frame(frame, hand_result)
                landmarks = self._convert_mediapipe_landmarks(hand_result)
                if landmarks:
                    # 1. Exponential landmark smoothing
                    smoothed = self._smooth_landmarks(landmarks)
                    # 2. Palm-size normalisation (wrist origin, scale-invariant)
                    norm = self._normalize_landmarks(smoothed)
                    # 3. Smoothed pinch + curl features
                    features = self._extract_features(norm)
                    # 4. Centre, deltas, orientation → store pose
                    self._compute_and_store_pose(smoothed, features, 1.0)
            else:
                # No hand visible — reset all smoothing state
                with self.thread_lock:
                    self._current_pose = None
                    self.prev_landmarks = None
                self._prev_pinch    = None
                self._prev_avg_curl = None
                self._prev_center   = None
        except Exception as e:
            print(f"Hand detection error: {e}")
        
        return frame
    
    def _draw_landmarks_on_frame(self, frame, hand_result):
        """Draw hand landmarks on the frame."""
        if not hand_result.hand_landmarks:
            return
            
        # Draw landmarks for first hand
        landmarks = hand_result.hand_landmarks[0]
        
        # Draw landmark points
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Draw connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    def _process_frame_fallback(self, frame):
        """Process frame using fallback detection (when Tasks API unavailable)."""
        # Simple hand detection using color/motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple blob detection for demonstration
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Minimum size threshold
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Create mock landmarks for gesture classification
                hand_center = (x + w // 2, y + h // 2)
                hand_size = max(w, h)
                mock_landmarks = self._create_mock_landmarks(hand_center, hand_size)
                
                # Draw simple landmarks
                for i, landmark in enumerate(mock_landmarks):
                    px = int(landmark.x * FRAME_WIDTH)
                    py = int(landmark.y * FRAME_HEIGHT)
                    cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)
                
                # Extract features and update state with mock landmarks (fallback)
                norm = self._normalize_landmarks(mock_landmarks)
                features = self._extract_features(norm)
                self._compute_and_store_pose(mock_landmarks, features, 0.5)
                
                # Add text overlay
                cv2.putText(frame, f"Hand Detected (Fallback Mode)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _create_mock_landmarks(self, hand_center, hand_size):
        """Create mock landmarks for fallback mode."""
        landmarks = []
        
        # Create 21 mock landmarks in standard MediaPipe format
        for i in range(21):
            angle = (i / 21) * 2 * np.pi
            x = hand_center[0] + hand_size * 0.5 * np.cos(angle)
            y = hand_center[1] + hand_size * 0.5 * np.sin(angle)
            
            class MockLandmark:
                def __init__(self, x, y):
                    self.x = x / FRAME_WIDTH  # Normalize to 0-1
                    self.y = y / FRAME_HEIGHT
                
            landmarks.append(MockLandmark(x, y))
        
        return landmarks
    
    def _camera_loop(self):
        """Main camera processing loop (runs in thread)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self._process_frame(frame)
            
            # Display frame
            cv2.imshow("Gesture Engine", frame)
            
            # FPS tracking
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                if elapsed > 0:
                    fps = 30 / elapsed
                    print(f"FPS: {fps:.1f}")
                self.fps_start_time = current_time
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def start(self):
        """Start gesture recognition (blocking mode - DAY 1)."""
        if not self._initialize_camera():
            return
        
        self.running = True
        print("Gesture Engine started. Press 'q' to quit.")
        
        try:
            self._camera_loop()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
    
    def start_thread(self):
        """Start gesture recognition in background thread (DAY 4)."""
        if self.thread and self.thread.is_alive():
            print("Gesture engine already running")
            return
        
        if not self._initialize_camera():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        print("Gesture Engine started in background thread")
    
    def stop(self):
        """Stop gesture recognition and clean up resources."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Clean up MediaPipe Tasks resources
        if self.use_tasks_api and self.hand_landmarker:
            try:
                self.hand_landmarker.close()
                self.hand_landmarker = None
            except Exception as e:
                print(f"Error closing hand landmarker: {e}")
        
        cv2.destroyAllWindows()
        print("Gesture Engine stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop()