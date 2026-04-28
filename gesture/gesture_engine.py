"""
gesture/gesture_engine.py
=========================
9-Stage Gesture Engine for the HoloScript hologram control system.

Pipeline:
  Stage 0  Frame timing         — real dt (replaces fixed 33 ms increment)
  Stage 1  MediaPipe extraction — 21 hand landmarks (unchanged)
  Stage 2  EMA smoothing        — dt-normalised alpha per-landmark
  Stage 3  Palm normalisation   — wrist origin, palm-size scale invariance
  Stage 4  Finger states        — TIP/PIP hysteresis booleans + pinch distance
  Stage 5  Gesture candidate    — strict priority tree (PINCH→V_SIGN→POINT→OPEN_PALM→FIST)
  Stage 6  Confirmation window  — per-gesture time debounce (80–300 ms)
  Stage 7  Velocity             — dt-normalised, EMA-smoothed, dead-zone gated
  Stage 8  Modal dispatch       — Gesture=Mode, Movement=Action
  Stage 9  SceneState update    — written every frame, never stale

Threading:
  GestureEngine runs in a background daemon thread.
  SceneState (RLock) is the shared blackboard written by this thread.
  SimpleRenderer reads SceneState on the main thread.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from .camera.camera_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from core.state.scene_state import scene_state

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError:
    MEDIAPIPE_TASKS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
_MODEL_FILENAME = "hand_landmarker.task"
_MODULE_DIR     = Path(__file__).parent
_MODEL_PATH     = next(
    (str(p) for p in [_MODULE_DIR / _MODEL_FILENAME,
                       _MODULE_DIR.parent / _MODEL_FILENAME]
     if p.exists()),
    None,
)


# ===========================================================================
# GestureEngine
# ===========================================================================

class GestureEngine:
    """Real-time 9-stage gesture engine using MediaPipe Tasks Vision API."""

    # ── EMA base alphas (calibrated at 30 FPS, dt-normalised at runtime) ──
    BASE_ALPHA     = 0.7   # landmark position smoothing
    CENTER_ALPHA   = 0.4   # palm-center position smoothing (lower = stabler)
    VELOCITY_ALPHA = 0.5   # velocity smoothing (derivatives are noisier)

    # ── Dead zone (normalised image units / second) ─────────────────────────
    DEAD_ZONE = 0.03

    # ── Control sensitivities ───────────────────────────────────────────────
    ROTATION_SENSITIVITY = 90.0  # degrees per image-unit / second
    ZOOM_SENSITIVITY     = 1.5   # scale per image-unit / second

    # ── Scale bounds ────────────────────────────────────────────────────────
    SCALE_MIN = 0.3
    SCALE_MAX = 4.0

    # ── Hand-loss grace period (seconds) ───────────────────────────────────
    HAND_LOSS_GRACE = 0.300

    # ── Confirmation windows per gesture (seconds) ─────────────────────────
    CONFIRMATION_WINDOWS: Dict[str, float] = {
        "PINCH":     0.080,
        "OPEN_PALM": 0.100,
        "V_SIGN":    0.150,
        "POINT":     0.150,
        "FIST":      0.200,
        "UNKNOWN":   0.300,
    }

    # ── Finger hysteresis bands (normalised image coords) ───────────────────
    # gap = PIP.y - TIP.y  (positive → finger extended above PIP)
    FINGER_OPEN_THRESH  =  0.015  # gap must exceed  +0.015 to open
    FINGER_CLOSE_THRESH = -0.015  # gap must fall below -0.015 to close

    # ── Pinch distance thresholds (normalised palm space) ───────────────────
    PINCH_ACTIVATE = 0.35   # distance below which pinch activates
    PINCH_RELEASE  = 0.50   # distance above which pinch releases

    # ── Minimum confidence to process a detection result ────────────────────
    MIN_CONFIDENCE = 0.60

    # MODIFIED ── Directional-lock continuous navigation constants ──────────────
    NAV_ENTRY_BUFFER    = 0.200  # seconds to ignore movement after POINT confirms
    NAV_COLLECT_WINDOW  = 0.100  # seconds to accumulate direction signal
    NAV_LOCK_THRESHOLD  = 0.012  # normalised total displacement to lock direction
    NAV_AXIS_RATIO      = 1.8    # axis dominance ratio (prevents diagonals)
    NAV_IDLE_THRESHOLD  = 0.004  # per-frame displacement below which = idle
    NAV_IDLE_TIMEOUT    = 0.200  # seconds of idle before soft-stop / PREVIEW entry
    NAV_RELOCK_FACTOR   = 1.5    # relock threshold = NAV_LOCK_THRESHOLD × this
    NAV_OPPOSITE_DAMP   = 0.2    # damping for opposite-direction movement

    # NEW ── PREVIEW / SNAP / FOCUS constants ───────────────────────────────
    PSF_RAY_FRAMES       = 6      # frames to average for pointing ray
    PSF_CANCEL_FACTOR    = 1.5    # cancel threshold = NAV_LOCK_THRESHOLD * this
    PSF_SNAP_TIMEOUT     = 0.300  # seconds of PREVIEW before auto-FOCUS
    PSF_NAV_SUPPRESS     = 0.150  # seconds to suppress nav after exiting FOCUS
    PSF_DEADZONE_ANGLE   = 0.15   # radians: min angular diff between candidates

    # ── Palm base landmarks for stable center computation ───────────────────
    _PALM_BASE = [0, 5, 9, 13, 17]   # wrist + 4 MCP joints

    # ── (name, TIP_index, PIP_index) for 4 fingers ──────────────────────────
    _FINGER_DEFS: List[Tuple[str, int, int]] = [
        ("index",  8,  6),
        ("middle", 12, 10),
        ("ring",   16, 14),
        ("pinky",  20, 18),
    ]

    # ── Gesture → Mode lookup ────────────────────────────────────────────────
    _GESTURE_MODE: Dict[str, str] = {
        "PINCH":     "ROTATE",
        "OPEN_PALM": "ZOOM",
        "FIST":      "FREEZE",
        "V_SIGN":    "RESET",
        "POINT":     "POINT",
        "UNKNOWN":   "NONE",
    }

    # ==========================================================================
    # Construction
    # ==========================================================================

    def __init__(self) -> None:
        # ── MediaPipe setup ──────────────────────────────────────────────────
        self.hand_landmarker = None
        self.use_tasks_api   = False

        if MEDIAPIPE_TASKS_AVAILABLE and _MODEL_PATH:
            try:
                base_options = mp_tasks.BaseOptions(
                    model_asset_path=_MODEL_PATH,
                    delegate=mp_tasks.BaseOptions.Delegate.CPU,
                )
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api   = True
                print(f"Using MediaPipe Tasks Vision API  [{_MODEL_PATH}]")
            except Exception as exc:
                print(f"Failed to initialise MediaPipe Tasks API: {exc}")
        elif not _MODEL_PATH:
            print(f"Model '{_MODEL_FILENAME}' not found — using fallback detection")

        if not self.use_tasks_api:
            print("Using fallback hand detection")

        # ── Camera / threading ───────────────────────────────────────────────
        self.cap          = None
        self.running      = False
        self.thread       = None
        self.thread_lock  = threading.Lock()

        # ── Backward-compat public interface (used by external callers) ──────
        self.current_gesture:   str   = "NONE"
        self.gesture_confidence: float = 0.0
        self._current_pose: Optional[Dict[str, Any]] = None

        # ── Frame timing (Stage 0) ───────────────────────────────────────────
        self.frame_timestamp_ms: int             = 0
        self._prev_time:         Optional[float] = None
        self._dt:                float            = 1.0 / 30.0

        # ── Landmark EMA state (Stage 2) ────────────────────────────────────
        self.prev_landmarks = None

        # ── Per-finger hysteresis states (Stage 4) ───────────────────────────
        # True = OPEN, False = CLOSED — initialise all closed
        self._finger_states: Dict[str, bool] = {
            "thumb": False, "index": False, "middle": False,
            "ring": False,  "pinky": False,
        }
        self._pinch_active: bool = False

        # ── Confirmation window state (Stage 6) ──────────────────────────────
        self._confirmed_candidate:  str            = "UNKNOWN"
        self._candidate_start_time: Optional[float] = None
        self._active_gesture:       str            = "UNKNOWN"

        # ── Modal control state (Stage 8) ────────────────────────────────────
        self._active_mode:     str   = "NONE"
        self._freeze_snapshot: float = 0.0

        # ── Velocity state (Stage 7) ─────────────────────────────────────────
        self._prev_center: Optional[Tuple[float, float, float]] = None
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._vz: float = 0.0

        # ── Hand-loss state ───────────────────────────────────────────────────
        self._hand_loss_time: Optional[float] = None

        # ── Performance tracking ─────────────────────────────────────────────
        self.fps_counter:    int   = 0
        self.fps_start_time: float = time.time()

        # MODIFIED ── Directional-lock continuous nav state (POINT mode only) ───
        self._nav_state:         str            = "IDLE"  # IDLE/COLLECTING/NAVIGATING
        self._nav_entry_time:    Optional[float] = None   # when POINT mode entered
        self._nav_anchor_x:      Optional[float] = None   # sliding frame-reference x
        self._nav_anchor_y:      Optional[float] = None   # sliding frame-reference y
        self._nav_collect_start: Optional[float] = None   # collection window start
        self._nav_dx_sum:        float           = 0.0    # accumulated x displacement
        self._nav_dy_sum:        float           = 0.0    # accumulated y displacement
        self._nav_direction:     Optional[str]   = None   # RIGHT/LEFT/UP/DOWN
        self._nav_idle_start:    Optional[float] = None   # when idle was first seen
        self._nav_is_relocking:  bool            = False  # stronger threshold after pause
        self._nav_prev_x:        Optional[float] = None   # last frame palm x
        self._nav_prev_y:        Optional[float] = None   # last frame palm y

        # NEW ── PREVIEW / SNAP / FOCUS state ────────────────────────────────
        # Exactly one of nav_mode / preview / focus is active at a time.
        # PSF state never overlaps with the nav state machine.
        self._psf_phase:          str            = "NAV"   # NAV/PREVIEW/FOCUS
        self._psf_preview_start:  Optional[float] = None   # when PREVIEW was entered
        self._psf_ray_buf:        List           = []      # ring buffer of (rx, ry) rays
        self._psf_cancel_anchor_x: Optional[float] = None  # anchor at PREVIEW entry
        self._psf_cancel_anchor_y: Optional[float] = None
        self._psf_nav_suppress_until: Optional[float] = None  # nav suppression after FOCUS
        self._latest_smoothed_lm = None                    # updated each pipeline frame

    # ==========================================================================
    # Stage 0 — Frame Timing
    # ==========================================================================

    def _update_dt(self) -> float:
        """Compute real elapsed time since last frame.

        Returns dt in seconds.  Guard against zero/negative to prevent
        division-by-zero in velocity and dt-normalised EMA.
        """
        now = time.monotonic()
        if self._prev_time is None:
            self._prev_time = now
            return self._dt   # first frame: return default 33 ms

        dt = now - self._prev_time
        self._prev_time = now
        self._dt = max(dt, 0.001)   # floor at 1 ms
        return self._dt

    # ==========================================================================
    # Stage 1 — Landmark Conversion  (unchanged from original)
    # ==========================================================================

    def _convert_mediapipe_landmarks(self, landmarks_list) -> Optional[List]:
        """Convert MediaPipe Tasks landmarks to simple objects with x, y, z."""
        if not landmarks_list or not landmarks_list.hand_landmarks:
            return None
        hand_landmarks = landmarks_list.hand_landmarks[0]
        converted = []
        for lm in hand_landmarks:
            class _LM:
                def __init__(self, x, y, z):
                    self.x = x; self.y = y; self.z = z
            converted.append(_LM(lm.x, lm.y, getattr(lm, "z", 0.0)))
        return converted

    # ==========================================================================
    # Stage 2 — DT-Normalised EMA Landmark Smoothing
    # ==========================================================================

    def _smooth_landmarks(self, current_landmarks, dt: float):
        """Apply dt-normalised exponential moving average to all 21 landmarks.

        alpha = BASE_ALPHA ^ (dt * 30)
        At 30 FPS: alpha = 0.7   (standard behaviour)
        At 15 FPS: alpha ≈ 0.49  (less smoothing — compensates for longer dt)
        At 60 FPS: alpha ≈ 0.84  (more smoothing — compensates for shorter dt)
        """
        alpha = self.BASE_ALPHA ** (dt * 30.0)

        if (self.prev_landmarks is None
                or len(self.prev_landmarks) != len(current_landmarks)):
            self.prev_landmarks = current_landmarks
            return current_landmarks

        smoothed = []
        for curr, prev in zip(current_landmarks, self.prev_landmarks):
            class _S:
                pass
            lm = _S()
            lm.x = alpha * curr.x + (1.0 - alpha) * prev.x
            lm.y = alpha * curr.y + (1.0 - alpha) * prev.y
            lm.z = alpha * curr.z + (1.0 - alpha) * prev.z
            smoothed.append(lm)

        self.prev_landmarks = smoothed
        return smoothed

    # ==========================================================================
    # Stage 3 — Palm Normalisation (unchanged logic, kept as separate method)
    # ==========================================================================

    def _normalize_landmarks(self, landmarks):
        """Translate to wrist origin and scale by wrist→middle-MCP distance.

        After normalisation:
          LM0 (wrist)      = (0, 0, 0)
          LM9 (middle MCP) ≈ (0, 1, 0)  — 1 palm-size away
          Extended fingertips ≈ 2.5–3.0 palm-sizes from origin
          Curled fingertips  ≈ 0.5–1.0 palm-sizes from origin
        """
        wrist      = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size  = math.sqrt(
            (middle_mcp.x - wrist.x) ** 2 +
            (middle_mcp.y - wrist.y) ** 2 +
            (middle_mcp.z - wrist.z) ** 2
        )
        if palm_size < 1e-6:
            palm_size = 1e-6   # guard degenerate frames

        norm = []
        for lm in landmarks:
            class _N:
                pass
            n = _N()
            n.x = (lm.x - wrist.x) / palm_size
            n.y = (lm.y - wrist.y) / palm_size
            n.z = (lm.z - wrist.z) / palm_size
            norm.append(n)
        return norm

    # ==========================================================================
    # Stage 4 — Per-Finger State with Hysteresis + Pinch Distance
    # ==========================================================================

    def _compute_finger_states(self, smoothed_lm, norm_lm) -> None:
        """Update per-finger boolean states using TIP/PIP hysteresis.

        Args:
            smoothed_lm: Image-space smoothed landmarks (x,y in [0,1]).
                         Used for TIP vs PIP y-coordinate comparison.
            norm_lm:     Palm-normalised landmarks (wrist at origin).
                         Used for pinch distance (scale-invariant).

        Hysteresis rule (index, middle, ring, pinky):
            gap = PIP.y - TIP.y
            CLOSED → OPEN  if gap >  FINGER_OPEN_THRESH  (+0.015)
            OPEN   → CLOSED if gap < FINGER_CLOSE_THRESH  (-0.015)
            (In image coords y increases downward, so positive gap = tip above PIP)

        Thumb rule (lateral extension after mirror-flip):
            gap = TIP.x - IP.x
            Same hysteresis thresholds applied on x-axis.
        """
        # ── 4 fingers: image-space y comparison ────────────────────────────
        for name, tip_idx, pip_idx in self._FINGER_DEFS:
            gap = smoothed_lm[pip_idx].y - smoothed_lm[tip_idx].y
            if not self._finger_states[name]:        # currently CLOSED
                if gap > self.FINGER_OPEN_THRESH:
                    self._finger_states[name] = True   # → OPEN
            else:                                    # currently OPEN
                if gap < self.FINGER_CLOSE_THRESH:
                    self._finger_states[name] = False  # → CLOSED

        # ── Thumb: image-space x comparison ─────────────────────────────────
        thumb_gap = smoothed_lm[4].x - smoothed_lm[3].x   # TIP.x - IP.x
        if not self._finger_states["thumb"]:
            if thumb_gap > self.FINGER_OPEN_THRESH:
                self._finger_states["thumb"] = True
        else:
            if thumb_gap < self.FINGER_CLOSE_THRESH:
                self._finger_states["thumb"] = False

        # ── Pinch: thumb-tip to index-tip distance in normalised palm space ──
        t, i = norm_lm[4], norm_lm[8]
        pinch_dist = math.sqrt(
            (t.x - i.x) ** 2 + (t.y - i.y) ** 2 + (t.z - i.z) ** 2
        )
        if not self._pinch_active:
            if pinch_dist < self.PINCH_ACTIVATE:
                self._pinch_active = True
        else:
            if pinch_dist > self.PINCH_RELEASE:
                self._pinch_active = False

    # ==========================================================================
    # Stage 5 — Gesture Priority Tree
    # ==========================================================================

    def _classify_gesture(self) -> str:
        """Return one gesture candidate per frame using strict priority order.

        Priority: PINCH → V_SIGN → POINT → OPEN_PALM → FIST → UNKNOWN

        Rationale for order:
          PINCH first — distance-based, unambiguous, highest specificity.
          V_SIGN before POINT — both require index open; prevents subset match.
          POINT before OPEN_PALM — 1-finger subset of 4-finger; avoids shadowing.
          OPEN_PALM before FIST — positive case before catch-all closed case.
          FIST last named gesture — requires ALL four fingers closed, no partial.
          UNKNOWN — anything not matching a named shape.
        """
        idx = self._finger_states["index"]
        mid = self._finger_states["middle"]
        rng = self._finger_states["ring"]
        pnk = self._finger_states["pinky"]

        # 1. PINCH — thumb-index distance gate (highest priority)
        if self._pinch_active:
            return "PINCH"

        # 2. V_SIGN — index + middle open, ring + pinky closed
        if idx and mid and not rng and not pnk:
            return "V_SIGN"

        # 3. POINT — only index open
        if idx and not mid and not rng and not pnk:
            return "POINT"

        # 4. OPEN_PALM — all four fingers open
        if idx and mid and rng and pnk:
            return "OPEN_PALM"

        # 5. FIST — all four fingers closed
        if not idx and not mid and not rng and not pnk:
            return "FIST"

        # 6. UNKNOWN — no named shape matched
        return "UNKNOWN"

    # ==========================================================================
    # Stage 6 — Confirmation Window (Time-Based Debounce)
    # ==========================================================================

    def _apply_confirmation_window(self, candidate: str) -> None:
        """Promote candidate to active_gesture only after its window elapses.

        State variables:
          _confirmed_candidate  — gesture currently being timed
          _candidate_start_time — monotonic time when this candidate first seen
          _active_gesture       — last fully confirmed gesture (drives control)

        On a NEW candidate: reset timer, keep current active_gesture.
        On SAME candidate:  accumulate time; if elapsed >= window, confirm it.

        The UNKNOWN window (300 ms) is the longest deliberately — it prevents
        the active gesture from dropping to NONE during normal hand transitions
        (which typically pass through ambiguous UNKNOWN states < 200 ms).
        """
        now = time.monotonic()

        if candidate != self._confirmed_candidate:
            # New candidate — restart timer, do not change active gesture yet
            self._confirmed_candidate  = candidate
            self._candidate_start_time = now
            return

        # Same candidate — check elapsed time
        elapsed = now - (self._candidate_start_time or now)
        window  = self.CONFIRMATION_WINDOWS.get(candidate, 0.300)

        if elapsed >= window and candidate != self._active_gesture:
            # Confirmed — transition to new mode
            self._on_mode_exit(self._active_mode)
            self._active_gesture = candidate
            new_mode = self._GESTURE_MODE.get(candidate, "NONE")
            self._active_mode = new_mode
            self._on_mode_enter(new_mode)

    # ==========================================================================
    # Mode Transition Events
    # ==========================================================================

    def _on_mode_exit(self, mode: str) -> None:
        """Cleanup actions when leaving a mode.

        Always resets velocity history — prevents velocity "kick" on mode entry.
        FREEZE exit: unfreeze scene (fixes the original one-way freeze bug).
        """
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        if mode == "FREEZE":
            scene_state.frozen = False
        if mode == "POINT":            # NEW — clear nav state so next POINT entry starts fresh
            self._reset_nav_state()
        scene_state.current_gesture = "NONE"

    def _on_mode_enter(self, mode: str) -> None:
        """Initialisation actions when entering a new mode.

        FREEZE: snapshot current rotation so it can be enforced every frame.
        RESET:  apply reset values once on entry (not repeated per-frame).
        """
        if mode == "FREEZE":
            self._freeze_snapshot = scene_state.rotation_y

        elif mode == "RESET":
            scene_state.rotation_y = 0.0
            scene_state.scale      = 1.0

    # ==========================================================================
    # Stage 7 — Velocity Computation
    # ==========================================================================

    def _compute_velocity(self, smoothed_lm, dt: float) -> Tuple[float, float, float]:
        """Compute dt-normalised, EMA-smoothed, dead-zone-gated velocity.

        Center computation:
          Average of 5 palm-base landmarks [0,5,9,13,17].
          These landmarks move as a rigid unit, averaging cancels per-point noise.
          Center is EMA-smoothed with CENTER_ALPHA (0.4) for extra stability.

        Velocity:
          raw_v = (smoothed_center - prev_center) / dt
          Dividing by dt makes velocity frame-rate independent.

        Velocity EMA (alpha=0.5):
          Lower than landmark alpha because derivatives amplify noise.

        Dead zone:
          |v| < DEAD_ZONE → v = 0.0 (suppresses sub-threshold hand tremor).

        Returns:
            (vx, vy, vz) — cleaned velocities in image-space units/second.
        """
        # ── Palm center (raw) ────────────────────────────────────────────────
        cx_raw = sum(smoothed_lm[i].x for i in self._PALM_BASE) / len(self._PALM_BASE)
        cy_raw = sum(smoothed_lm[i].y for i in self._PALM_BASE) / len(self._PALM_BASE)
        cz_raw = sum(smoothed_lm[i].z for i in self._PALM_BASE) / len(self._PALM_BASE)

        if self._prev_center is None:
            self._prev_center = (cx_raw, cy_raw, cz_raw)
            return 0.0, 0.0, 0.0

        # ── Smooth the center position ───────────────────────────────────────
        cx = self.CENTER_ALPHA * cx_raw + (1.0 - self.CENTER_ALPHA) * self._prev_center[0]
        cy = self.CENTER_ALPHA * cy_raw + (1.0 - self.CENTER_ALPHA) * self._prev_center[1]
        cz = self.CENTER_ALPHA * cz_raw + (1.0 - self.CENTER_ALPHA) * self._prev_center[2]

        # ── Raw velocity (displacement / dt) ────────────────────────────────
        raw_vx = (cx - self._prev_center[0]) / dt
        raw_vy = (cy - self._prev_center[1]) / dt
        raw_vz = (cz - self._prev_center[2]) / dt

        self._prev_center = (cx, cy, cz)

        # ── Smooth velocity ──────────────────────────────────────────────────
        self._vx = self.VELOCITY_ALPHA * self._vx + (1.0 - self.VELOCITY_ALPHA) * raw_vx
        self._vy = self.VELOCITY_ALPHA * self._vy + (1.0 - self.VELOCITY_ALPHA) * raw_vy
        self._vz = self.VELOCITY_ALPHA * self._vz + (1.0 - self.VELOCITY_ALPHA) * raw_vz

        # ── Dead zone ────────────────────────────────────────────────────────
        vx = 0.0 if abs(self._vx) < self.DEAD_ZONE else self._vx
        vy = 0.0 if abs(self._vy) < self.DEAD_ZONE else self._vy
        vz = 0.0 if abs(self._vz) < self.DEAD_ZONE else self._vz

        return vx, vy, vz

    # ==========================================================================
    # Stage 8 — Modal Control Dispatch
    # ==========================================================================

    def _apply_mode_control(self, vx: float, vy: float, dt: float) -> None:
        """Apply per-frame scene mutations based on active_mode.

        Gesture = Mode (what capability is active).
        Movement = Action (magnitude and direction within that mode).

        Mode behaviours:
          ROTATE — vx (horizontal hand velocity) drives rotation_y.
                   Moving hand LEFT  → negative vx → rotation decreases.
                   Moving hand RIGHT → positive vx → rotation increases.

          ZOOM   — vy (vertical hand velocity) drives scale.
                   Moving hand UP   → negative vy (image y decreases upward)
                   → negate: scale increases (zoom in).
                   Moving hand DOWN → positive vy → scale decreases (zoom out).
                   Note: z-depth from palm-center is not reliable in Tasks API
                   VIDEO mode because z is relative to wrist (rigid sub-motion).
                   vy is used as the reliable zoom axis.

          FREEZE — enforce freeze_snapshot; prevent any rotation drift.
                   frozen = True is maintained actively (not just set once).

          RESET  — action already fired in on_mode_enter; per-frame: label only.

          POINT  — reserved; no scene mutation yet.

          NONE   — no mutation; write "NONE" gesture label.
        """
        mode = self._active_mode

        if mode == "ROTATE":
            scene_state.rotation_y   += vx * self.ROTATION_SENSITIVITY * dt
            scene_state.frozen        = False
            scene_state.current_gesture = "ROTATE"

        elif mode == "ZOOM":
            # vy negative = hand moving up = zoom in → negate vy
            scale_delta = -vy * self.ZOOM_SENSITIVITY * dt
            new_scale   = scene_state.scale + scale_delta
            scene_state.scale           = max(self.SCALE_MIN, min(self.SCALE_MAX, new_scale))
            scene_state.frozen          = False
            scene_state.current_gesture = "ZOOM"

        elif mode == "FREEZE":
            scene_state.rotation_y      = self._freeze_snapshot
            scene_state.frozen          = True
            scene_state.current_gesture = "FIST"

        elif mode == "RESET":
            scene_state.frozen          = False
            scene_state.current_gesture = "RESET"

        elif mode == "POINT":  # MODIFIED ─ nav + PSF tick each frame
            scene_state.frozen          = False
            scene_state.current_gesture = "POINT"
            if self._psf_phase == "FOCUS":
                # FOCUS mode: intercept PINCH/OPENPALM for focused object
                # (standard vx/vy already computed; apply to focused context)
                scene_state.current_gesture = "FOCUS"
                # Future: route vx/vy to focused_object transforms here
                # For now: expose focus state; renderer handles visuals
            else:
                self._step_navigate(vx, vy)
            # PSF tick runs regardless of sub-mode
            if self._prev_center is not None:
                self._tick_preview_snap(
                    self._prev_center[0], self._prev_center[1],
                    time.monotonic()
                )

        else:  # NONE
            scene_state.frozen          = False
            scene_state.current_gesture = "NONE"

    # MODIFIED ── Directional-lock continuous navigation (POINT mode) ───────────

    def _step_navigate(self, vx_raw: float, vy_raw: float) -> None:
        """Directional-lock continuous navigation (POINT mode only).

        State machine:
          IDLE       — 200 ms entry buffer; ignore all movement.
          COLLECTING — accumulate per-frame deltas over a 100 ms window.
                       Locks when one axis dominates by 1.8×.
                       After idle soft-stop, re-enters with stronger lock threshold.
          NAVIGATING — emit direction event every active frame via sliding anchor.
                       Opposite movement is damped (×0.2) not blocked.
                       Idle for 200 ms → back to COLLECTING (soft-stop / re-lock).

        Writes scene_state.nav_event every frame while hand is moving.
        """
        now = time.monotonic()

        # Stability gate — skip if palm center not yet computed
        if self._prev_center is None:
            return
        cx = self._prev_center[0]
        cy = self._prev_center[1]

        # ── IDLE: entry buffer ───────────────────────────────────────────────
        if self._nav_state == "IDLE":
            if self._nav_entry_time is None:
                self._nav_entry_time = now
            self._nav_prev_x = cx
            self._nav_prev_y = cy
            if (now - self._nav_entry_time) < self.NAV_ENTRY_BUFFER:
                return
            # Buffer elapsed → begin collecting
            self._nav_dx_sum       = 0.0
            self._nav_dy_sum       = 0.0
            self._nav_collect_start = now
            self._nav_state        = "COLLECTING"
            return

        # ── COLLECTING: accumulate directional signal ────────────────────────
        if self._nav_state == "COLLECTING":
            if self._nav_prev_x is not None:
                self._nav_dx_sum += cx - self._nav_prev_x
                self._nav_dy_sum += cy - self._nav_prev_y
            self._nav_prev_x = cx
            self._nav_prev_y = cy

            abs_dx = abs(self._nav_dx_sum)
            abs_dy = abs(self._nav_dy_sum)
            lock_thr = self.NAV_LOCK_THRESHOLD * (
                self.NAV_RELOCK_FACTOR if self._nav_is_relocking else 1.0
            )

            if abs_dx > lock_thr or abs_dy > lock_thr:
                if abs_dx > abs_dy * self.NAV_AXIS_RATIO:
                    self._nav_direction    = "RIGHT" if self._nav_dx_sum > 0 else "LEFT"
                    self._nav_anchor_x     = cx
                    self._nav_anchor_y     = cy
                    self._nav_idle_start   = None
                    self._nav_is_relocking = False
                    self._nav_state        = "NAVIGATING"
                    print(f"[NAV] LOCKED {self._nav_direction}")
                    return
                elif abs_dy > abs_dx * self.NAV_AXIS_RATIO:
                    self._nav_direction    = "DOWN" if self._nav_dy_sum > 0 else "UP"
                    self._nav_anchor_x     = cx
                    self._nav_anchor_y     = cy
                    self._nav_idle_start   = None
                    self._nav_is_relocking = False
                    self._nav_state        = "NAVIGATING"
                    print(f"[NAV] LOCKED {self._nav_direction}")
                    return
                # else: ambiguous diagonal — keep collecting

            # Reset collection window when time expires without a lock
            if (now - self._nav_collect_start) > self.NAV_COLLECT_WINDOW:
                self._nav_dx_sum       = 0.0
                self._nav_dy_sum       = 0.0
                self._nav_collect_start = now
            return

        # ── NAVIGATING: continuous direction-locked movement ─────────────────
        if self._nav_anchor_x is None:
            self._nav_anchor_x = cx
            self._nav_anchor_y = cy
            return

        # Per-frame displacement from sliding anchor
        dx = cx - self._nav_anchor_x
        dy = cy - self._nav_anchor_y

        # Apply locked direction; damp movement in opposite direction
        direction = self._nav_direction
        if direction in ("RIGHT", "LEFT"):
            on_axis  = (direction == "RIGHT" and dx >= 0) or (direction == "LEFT" and dx <= 0)
            eff      = dx if on_axis else dx * self.NAV_OPPOSITE_DAMP
            magnitude = abs(eff)
        else:  # UP / DOWN
            on_axis  = (direction == "DOWN" and dy >= 0) or (direction == "UP" and dy <= 0)
            eff      = dy if on_axis else dy * self.NAV_OPPOSITE_DAMP
            magnitude = abs(eff)

        # Slide anchor to current position (per-frame delta scheme)
        self._nav_anchor_x = cx
        self._nav_anchor_y = cy

        # Idle detection → soft stop / PREVIEW entry
        if magnitude < self.NAV_IDLE_THRESHOLD:
            if self._nav_idle_start is None:
                self._nav_idle_start = now
            elif (now - self._nav_idle_start) >= self.NAV_IDLE_TIMEOUT:
                # Soft-stop: enter PREVIEW instead of raw COLLECTING transition
                if self._psf_phase == "NAV":
                    self._enter_preview(cx, cy, now)
                else:
                    # PREVIEW/FOCUS already active — just go back to collecting
                    self._nav_is_relocking  = True
                    self._nav_dx_sum        = 0.0
                    self._nav_dy_sum        = 0.0
                    self._nav_collect_start = now
                    self._nav_prev_x        = cx
                    self._nav_prev_y        = cy
                    self._nav_state         = "COLLECTING"
            return  # no nav event while idle
        else:
            self._nav_idle_start = None

        # Accumulate pointing ray for candidate lock while navigating
        if self._latest_smoothed_lm is not None:
            rx, ry = self._get_pointing_ray(self._latest_smoothed_lm)
            self._psf_ray_buf.append((rx, ry))
            if len(self._psf_ray_buf) > self.PSF_RAY_FRAMES:
                self._psf_ray_buf.pop(0)

        # Emit continuous direction event (only in NAV phase)
        if self._psf_phase == "NAV":
            # Honour nav suppression window after FOCUS exit
            if self._psf_nav_suppress_until and now < self._psf_nav_suppress_until:
                return
            _DIR = {"RIGHT": "NEXT", "LEFT": "PREV", "UP": "UP", "DOWN": "DOWN"}
            scene_state.nav_event = _DIR[direction]

    # ── Navigation state reset (called by _on_mode_exit) ────────────────────
    def _reset_nav_state(self) -> None:
        """Clear all nav + PSF state when leaving POINT mode."""
        self._nav_state         = "IDLE"
        self._nav_entry_time    = None
        self._nav_anchor_x      = None
        self._nav_anchor_y      = None
        self._nav_collect_start = None
        self._nav_dx_sum        = 0.0
        self._nav_dy_sum        = 0.0
        self._nav_direction     = None
        self._nav_idle_start    = None
        self._nav_is_relocking  = False
        self._nav_prev_x        = None
        self._nav_prev_y        = None
        # NEW ─ reset PSF layer
        self._psf_phase              = "NAV"
        self._psf_preview_start      = None
        self._psf_ray_buf            = []
        self._psf_cancel_anchor_x    = None
        self._psf_cancel_anchor_y    = None
        self._psf_nav_suppress_until = None
        scene_state.nav_phase        = "NAV"
        scene_state.preview_candidate = ""
        scene_state.focused_object    = ""

    # NEW ── PSF helper: get normalised wrist→index-tip direction vector ──────────
    @staticmethod
    def _get_pointing_ray(lm) -> tuple:
        """Return normalised (rx, ry) direction from wrist (LM0) to index tip (LM8)."""
        wx, wy = lm[0].x, lm[0].y
        ix, iy = lm[8].x, lm[8].y
        dx, dy = ix - wx, iy - wy
        length = math.sqrt(dx * dx + dy * dy) or 1e-6
        return dx / length, dy / length

    # NEW ── PSF helper: enter PREVIEW state ──────────────────────────────────
    def _enter_preview(self, cx: float, cy: float, now: float) -> None:
        """Transition NAV → PREVIEW. Lock candidate from averaged ray buffer."""
        # Compute averaged pointing ray from last N frames
        candidate = ""
        if self._psf_ray_buf:
            avg_rx = sum(r[0] for r in self._psf_ray_buf) / len(self._psf_ray_buf)
            avg_ry = sum(r[1] for r in self._psf_ray_buf) / len(self._psf_ray_buf)
            # Expose as direction hint; concrete object selection is renderer-side
            # Encode as "RAY:rx,ry" so renderer can do spatial hit-testing
            candidate = f"RAY:{avg_rx:+.4f},{avg_ry:+.4f}"

        self._psf_phase             = "PREVIEW"
        self._psf_preview_start     = now
        self._psf_cancel_anchor_x   = cx
        self._psf_cancel_anchor_y   = cy
        # Transition nav state machine to COLLECTING (relock after preview)
        self._nav_is_relocking  = True
        self._nav_dx_sum        = 0.0
        self._nav_dy_sum        = 0.0
        self._nav_collect_start = now
        self._nav_prev_x        = cx
        self._nav_prev_y        = cy
        self._nav_state         = "COLLECTING"

        scene_state.nav_phase         = "PREVIEW"
        scene_state.preview_candidate = candidate
        print(f"[PSF] PREVIEW  candidate={candidate}")

    # NEW ── PSF per-frame tick: called every frame in POINT mode ──────────────
    def _tick_preview_snap(self, cx: float, cy: float, now: float) -> None:
        """Drive PREVIEW and FOCUS state each frame (only called in POINT mode)."""
        if self._psf_phase == "NAV":
            return  # nothing to do; nav state machine handles everything

        if self._psf_phase == "PREVIEW":
            # ─ Cancel: movement exceeds threshold ─────────────────────────
            if self._psf_cancel_anchor_x is not None:
                mv = math.sqrt(
                    (cx - self._psf_cancel_anchor_x) ** 2 +
                    (cy - self._psf_cancel_anchor_y) ** 2
                )
                cancel_thr = self.NAV_LOCK_THRESHOLD * self.PSF_CANCEL_FACTOR
                if mv > cancel_thr:
                    # Cancel preview → return to NAV
                    self._psf_phase = "NAV"
                    self._nav_is_relocking = True
                    self._nav_dx_sum = 0.0
                    self._nav_dy_sum = 0.0
                    self._nav_collect_start = now
                    self._nav_prev_x = cx
                    self._nav_prev_y = cy
                    self._nav_state  = "COLLECTING"
                    scene_state.nav_phase         = "NAV"
                    scene_state.preview_candidate = ""
                    print("[PSF] PREVIEW cancelled (movement)")
                    return

            # ─ Snap: still idle for PSF_SNAP_TIMEOUT ──────────────────────
            if (now - self._psf_preview_start) >= self.PSF_SNAP_TIMEOUT:
                focused = scene_state.preview_candidate
                self._psf_phase = "FOCUS"
                scene_state.nav_phase      = "FOCUS"
                scene_state.focused_object = focused
                print(f"[PSF] FOCUS  object={focused}")
            return

        if self._psf_phase == "FOCUS":
            # FOCUS mode: navigation disabled; gestures apply to focused_object
            # Exit FOCUS when hand leaves POINT (handled in _on_mode_exit / _exit_focus)
            return

    # NEW ── PSF helper: exit FOCUS, return to NAV with suppression window ──────
    def _exit_focus(self, now: float) -> None:
        """Transition FOCUS → NAV with a nav suppression window."""
        self._psf_phase              = "NAV"
        self._psf_preview_start      = None
        self._psf_nav_suppress_until = now + self.PSF_NAV_SUPPRESS
        scene_state.nav_phase         = "NAV"
        scene_state.preview_candidate = ""
        scene_state.focused_object    = ""
        # Reset nav so the 150ms suppression window is honoured
        self._nav_is_relocking  = True
        self._nav_dx_sum        = 0.0
        self._nav_dy_sum        = 0.0
        self._nav_collect_start = now
        self._nav_state         = "COLLECTING"
        print("[PSF] FOCUS exited → NAV")

    # ── End MODIFIED + NEW ────────────────────────────────────────────────────────

    # ==========================================================================
    # Hand-Loss Handler
    # ==========================================================================

    def _handle_hand_loss(self) -> None:
        """Manage state when no hand is detected.

        Grace period (300 ms):
          Keep active_mode; zero all velocities.
          Scene becomes static (no drift) but mode is preserved.
          Allows momentary detection dropouts (partial occlusion) to recover
          without resetting the user's control mode.

        After grace period:
          Full reset — mode, smoothing history, finger states, velocity.
          scene_state.frozen set to False (unfreeze if was frozen).
        """
        now = time.monotonic()

        # Mark hand absent immediately
        scene_state.hand_present    = False
        scene_state.current_gesture = "NONE"

        if self._hand_loss_time is None:
            self._hand_loss_time = now

        elapsed = now - self._hand_loss_time

        if elapsed < self.HAND_LOSS_GRACE:
            # ── Grace period: keep mode, zero velocities ─────────────────────
            self._vx = 0.0
            self._vy = 0.0
            self._vz = 0.0
            # Apply mode with zero movement → scene freezes at current position
            self._apply_mode_control(0.0, 0.0, self._dt)

        else:
            # ── After grace: full reset ──────────────────────────────────────
            self._on_mode_exit(self._active_mode)
            self._active_mode            = "NONE"
            self._active_gesture         = "UNKNOWN"
            self._confirmed_candidate    = "UNKNOWN"
            self._candidate_start_time   = None
            self._hand_loss_time         = None
            self.prev_landmarks          = None
            self._prev_center            = None
            self._vx = self._vy = self._vz = 0.0
            self._pinch_active = False
            for key in self._finger_states:
                self._finger_states[key] = False
            with self.thread_lock:
                self._current_pose      = None
                self.current_gesture    = "NONE"
                self.gesture_confidence = 0.0

    # ==========================================================================
    # Main Frame Processing (Tasks API path)
    # ==========================================================================

    def _process_frame_tasks_api(self, frame: np.ndarray) -> np.ndarray:
        """Run the full 9-stage pipeline on a single camera frame."""

        # ── Stage 0: real dt ────────────────────────────────────────────────
        dt = self._update_dt()
        self.frame_timestamp_ms += int(dt * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            hand_result = self.hand_landmarker.detect_for_video(
                mp_image, self.frame_timestamp_ms
            )

            if hand_result and hand_result.hand_landmarks:
                # ── Hand detected ─────────────────────────────────────────────
                self._hand_loss_time    = None        # reset grace period
                scene_state.hand_present = True

                self._draw_landmarks_on_frame(frame, hand_result)

                # Stage 1: convert
                raw_lm = self._convert_mediapipe_landmarks(hand_result)
                if not raw_lm:
                    return frame

                # Stage 2: dt-normalised EMA smoothing
                smoothed = self._smooth_landmarks(raw_lm, dt)
                self._latest_smoothed_lm = smoothed  # NEW ─ cache for PSF ray computation

                # Stage 3: palm normalisation
                norm = self._normalize_landmarks(smoothed)

                # Stage 4: finger states + pinch (hysteresis)
                self._compute_finger_states(smoothed, norm)

                # Stage 5: gesture candidate (priority tree)
                candidate = self._classify_gesture()

                # Stage 6: confirmation window → sets _active_gesture/_active_mode
                self._apply_confirmation_window(candidate)

                # Stage 7: velocity (dt-normalised, EMA, dead zone)
                vx, vy, _vz = self._compute_velocity(smoothed, dt)

                # Stage 8: modal dispatch → writes SceneState
                self._apply_mode_control(vx, vy, dt)

                # Stage 9: store pose snapshot for external callers
                with self.thread_lock:
                    self._current_pose = {
                        "active_gesture": self._active_gesture,
                        "active_mode":    self._active_mode,
                        "candidate":      candidate,
                        "vx": vx, "vy": vy,
                        "finger_states":  self._finger_states.copy(),
                        "pinch_active":   self._pinch_active,
                    }
                    self.current_gesture    = self._active_gesture
                    self.gesture_confidence = 1.0

            else:
                # ── No hand detected ─────────────────────────────────────────
                self._handle_hand_loss()

        except Exception as exc:
            print(f"Hand detection error: {exc}")

        return frame

    # ==========================================================================
    # Landmark Visualisation (unchanged)
    # ==========================================================================

    def _draw_landmarks_on_frame(self, frame: np.ndarray, hand_result) -> None:
        """Draw MediaPipe landmark dots and finger connections on the camera feed."""
        if not hand_result.hand_landmarks:
            return
        landmarks = hand_result.hand_landmarks[0]
        for lm in landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]
        for s, e in connections:
            if s < len(landmarks) and e < len(landmarks):
                sp = (int(landmarks[s].x * frame.shape[1]),
                      int(landmarks[s].y * frame.shape[0]))
                ep = (int(landmarks[e].x * frame.shape[1]),
                      int(landmarks[e].y * frame.shape[0]))
                cv2.line(frame, sp, ep, (0, 255, 0), 2)

    # ==========================================================================
    # Fallback Detection (no MediaPipe model) — unchanged
    # ==========================================================================

    def _process_frame_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Basic contour-based fallback when MediaPipe model is unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center    = (x + w // 2, y + h // 2)
                hand_size = max(w, h)
                mock_lm   = self._create_mock_landmarks(center, hand_size)
                norm      = self._normalize_landmarks(mock_lm)
                self._compute_finger_states(mock_lm, norm)
                candidate = self._classify_gesture()
                self._apply_confirmation_window(candidate)
                dt = self._dt
                vx, vy, _ = self._compute_velocity(mock_lm, dt)
                self._apply_mode_control(vx, vy, dt)
                cv2.putText(frame, "Hand Detected (Fallback Mode)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
        return frame

    def _create_mock_landmarks(self, hand_center, hand_size):
        """Generate 21 geometrically-arranged mock landmarks for fallback mode."""
        lms = []
        for i in range(21):
            angle = (i / 21) * 2 * math.pi
            x = hand_center[0] + hand_size * 0.5 * math.cos(angle)
            y = hand_center[1] + hand_size * 0.5 * math.sin(angle)
            class _M:
                def __init__(self, px, py):
                    self.x = px / FRAME_WIDTH
                    self.y = py / FRAME_HEIGHT
                    self.z = 0.0
            lms.append(_M(x, y))
        return lms

    # ==========================================================================
    # Per-Frame Dispatcher (selects Tasks API or Fallback path)
    # ==========================================================================

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.use_tasks_api and self.hand_landmarker:
            return self._process_frame_tasks_api(frame)
        return self._process_frame_fallback(frame)

    # ==========================================================================
    # Camera Loop
    # ==========================================================================

    def _camera_loop(self) -> None:
        """Background thread: capture → process → display at camera frame rate."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)          # mirror for natural interaction
            frame = self._process_frame(frame)
            cv2.imshow("Gesture Engine", frame)

            # ── Console diagnostics every 10 frames ─────────────────────────
            self.fps_counter += 1
            if self.fps_counter % 10 == 0:
                now     = time.time()
                elapsed = now - self.fps_start_time
                fps     = 10.0 / elapsed if elapsed > 0 else 0.0
                self.fps_start_time = now

                with self.thread_lock:
                    pose = self._current_pose

                if pose:
                    print(
                        f"[{self.fps_counter:>6}] FPS:{fps:5.1f} | "
                        f"mode={pose['active_mode']:<8} "
                        f"gesture={pose['active_gesture']:<10} "
                        f"candidate={pose['candidate']:<10} | "
                        f"pinch={'PINCH' if pose['pinch_active'] else '     '} | "
                        f"vx={pose['vx']:+.4f} vy={pose['vy']:+.4f}"
                    )
                else:
                    print(f"[{self.fps_counter:>6}] FPS:{fps:5.1f} | NO HAND")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ==========================================================================
    # Camera Initialisation
    # ==========================================================================

    def _initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialised: {actual_w}×{actual_h}")
            return True
        except Exception as exc:
            print(f"Camera initialisation failed: {exc}")
            return False

    # ==========================================================================
    # Public API — backward-compatible with existing callers
    # ==========================================================================

    def get_current_gesture(self) -> Dict[str, Any]:
        """Thread-safe snapshot of the currently active gesture."""
        with self.thread_lock:
            return {
                "gesture":    self.current_gesture or "NONE",
                "confidence": self.gesture_confidence,
            }

    def get_hand_pose(self) -> Optional[Dict[str, Any]]:
        """Thread-safe snapshot of the full current hand pose.

        Returns None if no hand is currently detected.
        """
        with self.thread_lock:
            return self._current_pose

    def update_state(self, state_object) -> None:
        """Push gesture state into an external state object (legacy support)."""
        with self.thread_lock:
            if hasattr(state_object, "current_gesture"):
                state_object.current_gesture = self.current_gesture
            if hasattr(state_object, "gesture_confidence"):
                state_object.gesture_confidence = self.gesture_confidence

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    def start(self) -> None:
        """Start gesture recognition in blocking mode (DAY 1 usage)."""
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

    def start_thread(self) -> None:
        """Start gesture recognition in a background daemon thread."""
        if self.thread and self.thread.is_alive():
            print("Gesture engine already running")
            return
        if not self._initialize_camera():
            return
        self.running = True
        self.thread  = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        print("Gesture Engine started in background thread")

    def stop(self) -> None:
        """Stop recognition and release all resources cleanly."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        if self.use_tasks_api and self.hand_landmarker:
            try:
                self.hand_landmarker.close()
                self.hand_landmarker = None
            except Exception as exc:
                print(f"Error closing hand landmarker: {exc}")
        cv2.destroyAllWindows()
        print("Gesture Engine stopped")

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass