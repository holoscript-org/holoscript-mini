"use client"

import { useRef, useState, useCallback, useMemo, type MutableRefObject } from "react"

// ─── Types ────────────────────────────────────────────────────────────────────

export type GestureType = "PINCH" | "OPEN_PALM" | "FIST" | "POINT" | "NONE"

export interface HandLandmark {
  x: number
  y: number
  z: number
}

export interface GestureTransformState {
  rotationY: number
  scale: number
  frozen: boolean
  gesture: GestureType
}

export interface GestureControlRefs {
  rotationYRef: MutableRefObject<number>
  scaleRef: MutableRefObject<number>
  frozenRef: MutableRefObject<boolean>
}

export function classifyGesture(landmarks: HandLandmark[]): GestureType {
  if (landmarks.length < 21) return "NONE"

  const wrist = landmarks[0]
  const thumbTip = landmarks[4]
  const indexTip = landmarks[8]
  const palmSize = Math.hypot(
    landmarks[9].x - wrist.x,
    landmarks[9].y - wrist.y
  ) || 0.001
  const pinchDistance = Math.hypot(
    thumbTip.x - indexTip.x,
    thumbTip.y - indexTip.y
  ) / palmSize

  if (pinchDistance < 0.35) return "PINCH"

  const fingers = [
    { tip: landmarks[8], pip: landmarks[6] },
    { tip: landmarks[12], pip: landmarks[10] },
    { tip: landmarks[16], pip: landmarks[14] },
    { tip: landmarks[20], pip: landmarks[18] },
  ]
  const extended = fingers.map(({ tip, pip }) => tip.y < pip.y)
  const extendedCount = extended.filter(Boolean).length

  if (extendedCount === 0) return "FIST"
  if (extendedCount >= 4) return "OPEN_PALM"
  if (extended[0] && extendedCount === 1) return "POINT"

  return "NONE"
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SCALE_MIN = 0.3
const SCALE_MAX = 4.0

/** Degrees of rotation applied per image-unit/second of horizontal velocity */
const ROTATION_SENSITIVITY = 120
/** Scale multiplier per image-unit/second of vertical velocity */
const ZOOM_SENSITIVITY = 1.5

/** EMA weight for velocity smoothing (higher = more smoothing, more lag) */
const VELOCITY_EMA = 0.5

/** Min velocity (image-units/sec) before a motion registers — kills hand tremor */
const DEAD_ZONE = 0.015

/** Debounce windows before a gesture becomes confirmed (ms) */
const CONFIRM_MS: Record<GestureType, number> = {
  PINCH:     40,
  OPEN_PALM: 50,
  FIST:      180,
  POINT:     100,
  NONE:      120,
}

/** FIST must be held this long (ms) to toggle frozen */
const FREEZE_HOLD_MS = 400

/** Landmark indices that form the stable palm base for center computation */
const PALM_BASE = [0, 5, 9, 13, 17]

/** Minimum interval between React state updates (ms) — keeps renders at ~30 fps */
const STATE_UPDATE_MS = 125

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useGestureControl() {
  // ── Transform refs — updated every camera frame, no per-frame re-render ─────
  const rotYRef   = useRef(0)
  const scaleRef  = useRef(1)
  const frozenRef = useRef(false)
  const controls = useMemo<GestureControlRefs>(
    () => ({ rotationYRef: rotYRef, scaleRef, frozenRef }),
    []
  )

  // ── Velocity computation refs ─────────────────────────────────────────────
  const prevCenterRef = useRef<{ x: number; y: number } | null>(null)
  const prevTimeRef   = useRef(0)
  const vxEmaRef      = useRef(0)
  const vyEmaRef      = useRef(0)

  // ── Gesture confirmation refs ─────────────────────────────────────────────
  const rawGestureRef       = useRef<GestureType>("NONE")
  const gestureStartRef     = useRef(0)
  const confirmedGestureRef = useRef<GestureType>("NONE")

  // ── Freeze toggle refs ────────────────────────────────────────────────────
  const fistStartRef = useRef<number | null>(null)
  const fistFiredRef = useRef(false)

  // ── React display state (throttled to STATE_UPDATE_MS) ───────────────────
  const [state, setState] = useState<GestureTransformState>({
    rotationY: 0,
    scale: 1,
    frozen: false,
    gesture: "NONE",
  })
  const lastUpdateRef = useRef(0)

  /**
   * Process one camera frame.
   * Call this on every detected frame with the raw landmarks and classified gesture.
   * Pass an empty array for landmarks when no hand is present.
   */
  const processFrame = useCallback(
    (landmarks: HandLandmark[], gesture: GestureType, nowMs: number) => {
      // ── No hand: zero velocity, preserve last transform ───────────────────
      if (landmarks.length < 21) {
        prevCenterRef.current = null
        vxEmaRef.current = 0
        vyEmaRef.current = 0
        if (rawGestureRef.current !== "NONE") {
          rawGestureRef.current = "NONE"
          gestureStartRef.current = nowMs
        }
        if (confirmedGestureRef.current !== "NONE") {
          confirmedGestureRef.current = "NONE"
          fistStartRef.current = null
          fistFiredRef.current = false
          setState({
            rotationY: rotYRef.current,
            scale: scaleRef.current,
            frozen: frozenRef.current,
            gesture: "NONE",
          })
        }
        return
      }

      // ── dt — guard against first frame and extreme values ─────────────────
      const dt = prevTimeRef.current
        ? Math.min(Math.max((nowMs - prevTimeRef.current) / 1000, 0.001), 0.1)
        : 0.033
      prevTimeRef.current = nowMs

      // ── Gesture confirmation (debounce) ───────────────────────────────────
      if (gesture !== rawGestureRef.current) {
        rawGestureRef.current = gesture
        gestureStartRef.current = nowMs
      }

      const held = nowMs - gestureStartRef.current

      if (held >= (CONFIRM_MS[gesture] ?? 200) && confirmedGestureRef.current !== gesture) {
        // On transition: reset velocity so there is no "kick" on mode entry
        vxEmaRef.current = 0
        vyEmaRef.current = 0
        prevCenterRef.current = null
        confirmedGestureRef.current = gesture
      }

      const confirmed = confirmedGestureRef.current

      // ── Palm-center velocity (EMA-smoothed, dead-zone gated) ─────────────
      const cx = PALM_BASE.reduce((s, i) => s + landmarks[i].x, 0) / PALM_BASE.length
      const cy = PALM_BASE.reduce((s, i) => s + landmarks[i].y, 0) / PALM_BASE.length

      let vx = 0
      let vy = 0

      if (prevCenterRef.current) {
        const rawVx = (cx - prevCenterRef.current.x) / dt
        const rawVy = (cy - prevCenterRef.current.y) / dt
        vxEmaRef.current = VELOCITY_EMA * vxEmaRef.current + (1 - VELOCITY_EMA) * rawVx
        vyEmaRef.current = VELOCITY_EMA * vyEmaRef.current + (1 - VELOCITY_EMA) * rawVy
        vx = Math.abs(vxEmaRef.current) < DEAD_ZONE ? 0 : vxEmaRef.current
        vy = Math.abs(vyEmaRef.current) < DEAD_ZONE ? 0 : vyEmaRef.current
      }
      prevCenterRef.current = { x: cx, y: cy }

      // ── ROTATE (PINCH + horizontal velocity) ─────────────────────────────
      if (confirmed === "PINCH" && !frozenRef.current) {
        rotYRef.current += vx * ROTATION_SENSITIVITY * dt
      }

      // ── ZOOM (OPEN_PALM + vertical velocity) ─────────────────────────────
      // image-y increases downward: negate vy so moving hand UP zooms in
      if (confirmed === "OPEN_PALM" && !frozenRef.current) {
        scaleRef.current = Math.max(
          SCALE_MIN,
          Math.min(SCALE_MAX, scaleRef.current - vy * ZOOM_SENSITIVITY * dt)
        )
      }

      // ── FREEZE toggle (FIST held for FREEZE_HOLD_MS) ─────────────────────
      if (confirmed === "FIST") {
        if (fistStartRef.current === null) {
          fistStartRef.current = nowMs
          fistFiredRef.current = false
        } else if (!fistFiredRef.current && nowMs - fistStartRef.current >= FREEZE_HOLD_MS) {
          frozenRef.current = !frozenRef.current
          fistFiredRef.current = true
        }
      } else {
        fistStartRef.current = null
        fistFiredRef.current = false
      }

      // ── Throttled React state push (~30 fps) ─────────────────────────────
      if (nowMs - lastUpdateRef.current >= STATE_UPDATE_MS) {
        lastUpdateRef.current = nowMs
        setState({
          rotationY: rotYRef.current,
          scale:     scaleRef.current,
          frozen:    frozenRef.current,
          gesture:   confirmed,
        })
      }
    },
    []
  )

  const reset = useCallback(() => {
    rotYRef.current   = 0
    scaleRef.current  = 1
    frozenRef.current = false
    vxEmaRef.current  = 0
    vyEmaRef.current  = 0
    prevCenterRef.current = null
    prevTimeRef.current = 0
    rawGestureRef.current = "NONE"
    confirmedGestureRef.current = "NONE"
    gestureStartRef.current = 0
    fistStartRef.current = null
    fistFiredRef.current = false
    setState({ rotationY: 0, scale: 1, frozen: false, gesture: "NONE" })
  }, [])

  return { state, controls, processFrame, reset }
}
