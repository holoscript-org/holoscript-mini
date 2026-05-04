"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { HoloPanel } from "./holo-panel"
import { Video, VideoOff, Hand } from "lucide-react"
import { type GestureType, type HandLandmark } from "@/hooks/useGestureControl"

// ─── Types ────────────────────────────────────────────────────────────────────

interface DetectedHand {
  landmarks: HandLandmark[]
  handedness: "Left" | "Right"
}

// ─── MediaPipe CDN paths ──────────────────────────────────────────────────────
// WASM runtime and model are loaded at runtime from CDN — no local files needed.
const MP_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
const MP_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
const CAMERA_WIDTH = 320
const CAMERA_HEIGHT = 240
const UI_UPDATE_MS = 125
const MP_DETECT_INTERVAL_MS = 50
const POINT_HOLD_TOGGLE_MS = 700

const CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]
const TIP_INDICES = [4, 8, 12, 16, 20]

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number) => void) => number
  cancelVideoFrameCallback?: (handle: number) => void
}

interface WorkerUpdate {
  rotationDelta: number
  scaleDelta: number
  gesture: GestureType
  hands?: DetectedHand[]
  handsDetected?: number
  nowMs?: number
  type?: "update"
  indexTip?: { x: number; y: number } | null
}

// ─── Component ────────────────────────────────────────────────────────────────

/** How long after last POINT gesture PINCH is still treated as "select" not "rotate" */
const POINT_MODE_WINDOW_MS = 600

export function CameraPanel({
  onGestureUpdate,
  onPinchSelect,
  onPointMove,
  onPointHoldToggle,
  resetSeq,
  compact = false,
}: {
  onGestureUpdate?: (rotationDelta: number, scaleDelta: number, gesture: GestureType, nowMs: number) => void
  /** Called when a pinch fires shortly after a POINT gesture — provides normalised (0–1) index-tip coords */
  onPinchSelect?: (x: number, y: number) => void
  /** Called when POINT gesture provides a new index-tip position; null hides the reticle */
  onPointMove?: (point: { x: number; y: number } | null, nowMs: number) => void
  /** Called when POINT is held long enough to toggle scene-drag mode */
  onPointHoldToggle?: () => void
  /** Increments when the UI resets gesture state */
  resetSeq?: number
  compact?: boolean
}) {
  const onGestureUpdateRef = useRef(onGestureUpdate)
  const onPinchSelectRef  = useRef(onPinchSelect)
  const onPointMoveRef    = useRef(onPointMove)
  const onPointHoldToggleRef = useRef(onPointHoldToggle)
  const videoRef  = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const lastVideoTimeRef = useRef(-1)
  const lastHandsRef = useRef<DetectedHand[]>([])
  const inFlightRef = useRef(false)
  const lastSendMsRef = useRef(0)

  const animRafRef      = useRef<number | null>(null)
  const videoFrameCallbackRef = useRef<number | null>(null)
  const lastUiUpdateRef = useRef(0)
  const displayedGestureRef = useRef<GestureType>("NONE")
  const displayedHandsRef = useRef(0)
  const debugIntervalRef = useRef<number | null>(null)
  // ── Reticle / pinch-to-select ──────────────────────────────────────────────
  const indexTipRef     = useRef<{ x: number; y: number } | null>(null)
  const currentGestureRef = useRef<GestureType>("NONE")
  const lastPointMsRef  = useRef(0)
  const lastPointActiveRef = useRef(false)
  const pointHoldStartRef = useRef<number | null>(null)
  const pointHoldFiredRef = useRef(false)

  const [isStreaming,    setIsStreaming]    = useState(false)
  const [currentGesture, setCurrentGesture] = useState<GestureType>("NONE")
  const [handsDetected,  setHandsDetected]  = useState(0)
  const [mpReady,        setMpReady]        = useState(false)
  const [cameraError,    setCameraError]    = useState<string | null>(null)

  useEffect(() => { onGestureUpdateRef.current = onGestureUpdate }, [onGestureUpdate])
  useEffect(() => { onPinchSelectRef.current  = onPinchSelect  }, [onPinchSelect])
  useEffect(() => { onPointMoveRef.current    = onPointMove    }, [onPointMove])
  useEffect(() => { onPointHoldToggleRef.current = onPointHoldToggle }, [onPointHoldToggle])

  useEffect(() => {
    lastPointMsRef.current = 0
    lastPointActiveRef.current = false
    pointHoldStartRef.current = null
    pointHoldFiredRef.current = false
    currentGestureRef.current = "NONE"
    indexTipRef.current = null
    displayedGestureRef.current = "NONE"
    displayedHandsRef.current = 0
    lastUiUpdateRef.current = 0
    setCurrentGesture("NONE")
    setHandsDetected(0)
    onPointMoveRef.current?.(null, performance.now())
    workerRef.current?.postMessage({ type: "reset" })
  }, [resetSeq])

  // ── Worker setup ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isStreaming) return

    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL("../workers/gestureWorker.js", import.meta.url),
        { type: "module" }
      )

      workerRef.current.onmessage = (event) => {
        const data = event.data as WorkerUpdate | { type?: string }
        if (data.type === "ready") {
          setMpReady(true)
          return
        }
        if (data.type === "error") {
          const err = data as { message?: string }
          console.error("[GestureWorker] init error", err.message ?? "unknown")
          setMpReady(false)
          return
        }
        if (data.type === "stats") {
          const stats = data as {
            lastFrameMs?: number
            lastHandsCount?: number
            lastGesture?: GestureType
            lastError?: string
            initErrorMessage?: string
          }
          console.log("[GestureWorker] stats", stats)
          return
        }

        if ((data as WorkerUpdate).rotationDelta !== undefined) {
          const update = data as WorkerUpdate
          inFlightRef.current = false

          if (update.hands) {
            lastHandsRef.current = update.hands
          }

          // Track fingertip and gesture for reticle / pinch-to-select
          if (update.indexTip) indexTipRef.current = update.indexTip
          currentGestureRef.current = update.gesture

          const detected = update.handsDetected ?? update.hands?.length ?? 0
          const nowMs = update.nowMs ?? performance.now()

          // ── Point-mode tracking and pinch-to-select ───────────────────────
          if (update.gesture === "POINT") {
            lastPointMsRef.current = nowMs
            lastPointActiveRef.current = true
            if (pointHoldStartRef.current === null) {
              pointHoldStartRef.current = nowMs
              pointHoldFiredRef.current = false
            } else if (!pointHoldFiredRef.current && nowMs - pointHoldStartRef.current >= POINT_HOLD_TOGGLE_MS) {
              pointHoldFiredRef.current = true
              onPointHoldToggleRef.current?.()
            }
            if (indexTipRef.current) {
              onPointMoveRef.current?.(indexTipRef.current, nowMs)
            }
          } else if (update.gesture === "PINCH") {
            const inPointWindow = nowMs - lastPointMsRef.current < POINT_MODE_WINDOW_MS
            if (inPointWindow && indexTipRef.current) {
              onPinchSelectRef.current?.(indexTipRef.current.x, indexTipRef.current.y)
              lastPointMsRef.current = 0  // prevent re-fire
            }
          } else if (lastPointActiveRef.current) {
            lastPointActiveRef.current = false
            onPointMoveRef.current?.(null, nowMs)
            pointHoldStartRef.current = null
            pointHoldFiredRef.current = false
          }

          if (
            nowMs - lastUiUpdateRef.current >= UI_UPDATE_MS &&
            (displayedGestureRef.current !== update.gesture || displayedHandsRef.current !== detected)
          ) {
            lastUiUpdateRef.current = nowMs
            displayedGestureRef.current = update.gesture
            displayedHandsRef.current = detected
            setHandsDetected(detected)
            setCurrentGesture(update.gesture)
          }

          // Suppress rotation while in point-mode window so PINCH means "select" not "rotate"
          const inPointWindow = nowMs - lastPointMsRef.current < POINT_MODE_WINDOW_MS
          onGestureUpdateRef.current?.(
            inPointWindow ? 0 : update.rotationDelta,
            update.scaleDelta,
            update.gesture,
            nowMs
          )
        }
      }

      workerRef.current.onerror = () => {
        console.error("[GestureWorker] worker error")
        setMpReady(false)
      }
    }

    workerRef.current.postMessage({
      type: "init",
      wasmUrl: MP_WASM_URL,
      modelUrl: MP_MODEL_URL,
    })

    return () => {
      workerRef.current?.terminate()
      workerRef.current = null
      setMpReady(false)
      inFlightRef.current = false
    }
  }, [isStreaming])

  // ── Canvas drawing ────────────────────────────────────────────────────────────
  const drawLandmarks = useCallback(
    (ctx: CanvasRenderingContext2D, hands: DetectedHand[]) => {
      const { width, height } = ctx.canvas
      ctx.clearRect(0, 0, width, height)

      for (const hand of hands) {
        const color = hand.handedness === "Left" ? "#00ffff" : "#00ff88"
        ctx.strokeStyle = color
        ctx.lineWidth   = 2
        ctx.shadowColor = color
        ctx.shadowBlur  = 8

        for (const [s, e] of CONNECTIONS) {
          ctx.beginPath()
          ctx.moveTo(hand.landmarks[s].x * width, hand.landmarks[s].y * height)
          ctx.lineTo(hand.landmarks[e].x * width, hand.landmarks[e].y * height)
          ctx.stroke()
        }

        for (let i = 0; i < hand.landmarks.length; i++) {
          const lm = hand.landmarks[i]
          const r  = i === 0 ? 6 : 4
          ctx.beginPath()
          ctx.arc(lm.x * width, lm.y * height, r, 0, Math.PI * 2)
          ctx.fillStyle = color
          ctx.fill()
          if (TIP_INDICES.includes(i)) {
            ctx.beginPath()
            ctx.arc(lm.x * width, lm.y * height, 8, 0, Math.PI * 2)
            ctx.strokeStyle = color
            ctx.lineWidth   = 1
            ctx.stroke()
          }
        }

        const wrist = hand.landmarks[0]
        ctx.font      = "12px monospace"
        ctx.fillStyle = color
        ctx.shadowBlur = 4
        ctx.fillText(`${hand.handedness}`, wrist.x * width - 20, wrist.y * height + 30)

        // Reticle: crosshair at index fingertip when pointing
        if (currentGestureRef.current === "POINT" && hand.landmarks[8]) {
          const lm = hand.landmarks[8]
          const rx = lm.x * width
          const ry = lm.y * height
          ctx.strokeStyle = "#ffff00"
          ctx.lineWidth   = 2
          ctx.shadowColor = "#ffff00"
          ctx.shadowBlur  = 12
          ctx.beginPath(); ctx.moveTo(rx - 18, ry); ctx.lineTo(rx + 18, ry); ctx.stroke()
          ctx.beginPath(); ctx.moveTo(rx, ry - 18); ctx.lineTo(rx, ry + 18); ctx.stroke()
          ctx.beginPath(); ctx.arc(rx, ry, 9, 0, Math.PI * 2); ctx.stroke()
        }
      }

      ctx.shadowBlur = 0
    },
    []
  )

  // ── Main animation loop ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isStreaming) return

    const canvas = canvasRef.current
    const video  = videoRef.current as VideoWithFrameCallback | null
    if (!canvas || !video) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    let stopped = false

    const schedule = () => {
      if (stopped) return
      if (video.requestVideoFrameCallback) {
        videoFrameCallbackRef.current = video.requestVideoFrameCallback(processFrame)
      } else {
        animRafRef.current = requestAnimationFrame(processFrame)
      }
    }

    const processFrame = () => {
      // Sync canvas size to video
      const vw = video.videoWidth  || CAMERA_WIDTH
      const vh = video.videoHeight || CAMERA_HEIGHT
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width  = vw
        canvas.height = vh
      }

      const hands = lastHandsRef.current
      const nowMs = performance.now()

      const videoTime = video.currentTime
      const isNewFrame = videoTime !== lastVideoTimeRef.current
      const canSend =
        !!workerRef.current &&
        mpReady &&
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        !inFlightRef.current &&
        nowMs - lastSendMsRef.current >= MP_DETECT_INTERVAL_MS &&
        isNewFrame

      if (canSend) {
        inFlightRef.current = true
        lastSendMsRef.current = nowMs
        lastVideoTimeRef.current = videoTime
        createImageBitmap(video)
          .then((bitmap) => {
            workerRef.current?.postMessage({ type: "frame", bitmap, nowMs }, [bitmap])
          })
          .catch(() => {
            inFlightRef.current = false
          })
      }

      drawLandmarks(ctx, hands)
      schedule()
    }

    if (!debugIntervalRef.current) {
      debugIntervalRef.current = window.setInterval(() => {
        const nowMs = performance.now()
        const videoTime = video.currentTime
        const isNewFrame = videoTime !== lastVideoTimeRef.current
        console.log("[GestureMain] stats", {
          mpReady,
          readyState: video.readyState,
          videoWidth: video.videoWidth,
          videoHeight: video.videoHeight,
          inFlight: inFlightRef.current,
          lastSendMs: lastSendMsRef.current,
          deltaSinceSend: Math.round(nowMs - lastSendMsRef.current),
          isNewFrame,
        })
      }, 2000)
    }

    schedule()
    return () => {
      stopped = true
      if (animRafRef.current) cancelAnimationFrame(animRafRef.current)
      if (videoFrameCallbackRef.current && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(videoFrameCallbackRef.current)
      }
      if (debugIntervalRef.current) {
        clearInterval(debugIntervalRef.current)
        debugIntervalRef.current = null
      }
    }
  }, [isStreaming, drawLandmarks, mpReady])

  // ── Camera start / stop ───────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAMERA_WIDTH },
          height: { ideal: CAMERA_HEIGHT },
          frameRate: { ideal: 30, max: 30 },
          facingMode: "user",
        },
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }
    } catch (err) {
      console.warn("[CameraPanel] Camera unavailable:", err)
      setCameraError("Camera permission or device unavailable")
      return
    }
    setCameraError(null)
    setIsStreaming(true)
  }

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setIsStreaming(false)
    setHandsDetected(0)
    setCurrentGesture("NONE")
    displayedHandsRef.current = 0
    displayedGestureRef.current = "NONE"
    lastUiUpdateRef.current = 0
    setMpReady(false)
    setCameraError(null)
    lastHandsRef.current = []
    lastSendMsRef.current = 0
    lastVideoTimeRef.current = -1
    inFlightRef.current = false
    workerRef.current?.postMessage({ type: "reset" })
  }

  // ── Gesture labels ────────────────────────────────────────────────────────────
  const GESTURE_LABELS: Record<GestureType, string> = {
    PINCH:     "Pinch — Rotate",
    OPEN_PALM: "Open Palm — Zoom",
    FIST:      "Fist — Freeze (hold)",
    POINT:     "Point — Reticle",
    V_SIGN:    "V-Sign — Reset",
    NONE:      "No Gesture",
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <HoloPanel title="Gesture Control" statusIndicator className="h-full">
      <div
        className={`relative bg-background/50 rounded overflow-hidden ${
          compact ? "h-full min-h-0" : "aspect-video"
        }`}
      >
        {/* Live video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover ${
            isStreaming ? "opacity-100" : "opacity-0"
          }`}
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Landmark overlay canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Enable camera button */}
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <button
              onClick={startCamera}
              className="flex flex-col items-center gap-3 group focus:outline-none"
              aria-label="Start camera"
            >
              <div className="w-16 h-16 rounded-full border-2 border-primary/40 group-hover:border-primary/80 group-hover:bg-primary/10 flex items-center justify-center transition-all duration-200 shadow-[0_0_16px_rgba(0,255,255,0.1)] group-hover:shadow-[0_0_24px_rgba(0,255,255,0.25)]">
                <Video className="w-7 h-7 text-primary/60 group-hover:text-primary transition-colors" />
              </div>
              <span className="text-xs font-mono text-muted-foreground group-hover:text-primary transition-colors tracking-widest uppercase">
                Enable Camera
              </span>
              {cameraError && (
                <span className="max-w-[220px] text-center text-[10px] font-mono text-red-400/80">
                  {cameraError}
                </span>
              )}
            </button>
          </div>
        )}

        {/* Stop button */}
        {isStreaming && (
          <div className="absolute top-2 right-2 flex items-center gap-2">
            {mpReady && (
              <span className="text-[9px] font-mono text-green-400/80 bg-black/60 px-1.5 py-0.5 rounded border border-green-400/30">
                MediaPipe
              </span>
            )}
            {!mpReady && (
              <span className="text-[9px] font-mono text-yellow-400/60 bg-black/60 px-1.5 py-0.5 rounded border border-yellow-400/20 animate-pulse">
                Loading…
              </span>
            )}
            <button
              onClick={stopCamera}
              className="p-2 rounded border border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/25 transition-colors"
              aria-label="Stop camera"
            >
              <VideoOff className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Gesture status bar */}
        {isStreaming && (
          <div className="absolute bottom-2 left-2 right-2">
            <div className="flex items-center justify-between gap-2 px-3 py-2 rounded border border-primary/30 bg-background/90 backdrop-blur-sm">
              <div className="flex items-center gap-2">
                <Hand className="w-4 h-4 text-primary" />
                <span className="text-xs font-mono text-muted-foreground">
                  {handsDetected} hand{handsDetected !== 1 ? "s" : ""}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    currentGesture !== "NONE"
                      ? "bg-primary animate-pulse shadow-[0_0_8px_rgba(0,255,255,0.8)]"
                      : "bg-muted-foreground"
                  }`}
                />
                <span className="text-xs font-mono text-primary">
                  {GESTURE_LABELS[currentGesture]}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Scan-line animation */}
        {isStreaming && (
          <div className="absolute inset-0 pointer-events-none">
            <div
              className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent opacity-40"
              style={{ animation: "scan 2s linear infinite" }}
            />
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes scan {
          0%   { top: 0 }
          100% { top: 100% }
        }
      `}</style>
    </HoloPanel>
  )
}
