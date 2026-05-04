"use client"

import { useRef, useState, useCallback, useMemo, type MutableRefObject } from "react"

// ─── Types ────────────────────────────────────────────────────────────────────

export type GestureType = "PINCH" | "OPEN_PALM" | "FIST" | "POINT" | "V_SIGN" | "NONE"

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
  if (extended[0] && extended[1] && !extended[2] && !extended[3]) return "V_SIGN"

  return "NONE"
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SCALE_MIN = 0.3
const SCALE_MAX = 4.0

/** Gesture update target interval (ms) ~20 FPS */
const PROCESS_INTERVAL_MS = 50

/** EMA weight for transform smoothing (higher = smoother, more lag) */
const ROTATION_SMOOTH_ALPHA = 0.2
const SCALE_SMOOTH_ALPHA = 0.25

/** FIST must be held this long (ms) to toggle frozen */
const FREEZE_HOLD_MS = 400

/** Minimum interval between React state updates (ms) — keeps renders at ~30 fps */
const STATE_UPDATE_MS = 125

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useGestureControl() {
  // ── Transform refs — updated every camera frame, no per-frame re-render ─────
  const rotYRef   = useRef(0)
  const scaleRef  = useRef(1)
  const frozenRef = useRef(false)
  const targetRotYRef = useRef(0)
  const targetScaleRef = useRef(1)
  const controls = useMemo<GestureControlRefs>(
    () => ({ rotationYRef: rotYRef, scaleRef, frozenRef }),
    []
  )

  // ── Freeze toggle refs ────────────────────────────────────────────────────
  const fistStartRef = useRef<number | null>(null)
  const fistFiredRef = useRef(false)
  // ── V-sign reset refs ──────────────────────────────────────────────────────
  const vSignFiredRef = useRef(false)

  // ── React display state (throttled to STATE_UPDATE_MS) ───────────────────
  const [state, setState] = useState<GestureTransformState>({
    rotationY: 0,
    scale: 1,
    frozen: false,
    gesture: "NONE",
  })
  const lastUpdateRef = useRef(0)
  const lastProcessRef = useRef(0)

  /**
   * Process one camera frame.
   * Call this on every detected frame with the raw landmarks and classified gesture.
   * Pass an empty array for landmarks when no hand is present.
   */
  const applyWorkerUpdate = useCallback(
    (rotationDelta: number, scaleDelta: number, gesture: GestureType, nowMs: number) => {
      if (nowMs - lastProcessRef.current < PROCESS_INTERVAL_MS) return
      lastProcessRef.current = nowMs

      targetRotYRef.current += rotationDelta
      targetScaleRef.current = Math.max(
        SCALE_MIN,
        Math.min(SCALE_MAX, targetScaleRef.current + scaleDelta)
      )

      rotYRef.current = rotYRef.current + (targetRotYRef.current - rotYRef.current) * ROTATION_SMOOTH_ALPHA
      scaleRef.current = scaleRef.current + (targetScaleRef.current - scaleRef.current) * SCALE_SMOOTH_ALPHA

      // ── FREEZE toggle (FIST held for FREEZE_HOLD_MS) ─────────────────────
      if (gesture === "FIST") {
        if (fistStartRef.current === null) {
          fistStartRef.current = nowMs
          fistFiredRef.current = false
        } else if (!fistFiredRef.current && nowMs - fistStartRef.current >= FREEZE_HOLD_MS) {
          frozenRef.current = !frozenRef.current
          targetRotYRef.current = rotYRef.current
          targetScaleRef.current = scaleRef.current
          fistFiredRef.current = true
        }
      } else {
        fistStartRef.current = null
        fistFiredRef.current = false
      }

      // ── V-SIGN reset (fires once per hold) ───────────────────────────────
      if (gesture === "V_SIGN") {
        if (!vSignFiredRef.current) {
          vSignFiredRef.current = true
          rotYRef.current = 0
          scaleRef.current = 1
          frozenRef.current = false
          targetRotYRef.current = 0
          targetScaleRef.current = 1
          fistStartRef.current = null
          fistFiredRef.current = false
        }
      } else {
        vSignFiredRef.current = false
      }

      // ── Throttled React state push (~30 fps) ─────────────────────────────
      if (nowMs - lastUpdateRef.current >= STATE_UPDATE_MS) {
        lastUpdateRef.current = nowMs
        setState({
          rotationY: rotYRef.current,
          scale:     scaleRef.current,
          frozen:    frozenRef.current,
          gesture,
        })
      }
    },
    []
  )

  const reset = useCallback(() => {
    rotYRef.current   = 0
    scaleRef.current  = 1
    frozenRef.current = false
    targetRotYRef.current = 0
    targetScaleRef.current = 1
    fistStartRef.current = null
    fistFiredRef.current = false
    vSignFiredRef.current = false
    setState({ rotationY: 0, scale: 1, frozen: false, gesture: "NONE" })
  }, [])

  const setScale = useCallback((newScale: number) => {
    const clamped = Math.max(SCALE_MIN, Math.min(SCALE_MAX, newScale))
    scaleRef.current = clamped
    targetScaleRef.current = clamped
    setState((prev) => ({ ...prev, scale: clamped }))
  }, [])

  return { state, controls, applyWorkerUpdate, reset, setScale }
}
