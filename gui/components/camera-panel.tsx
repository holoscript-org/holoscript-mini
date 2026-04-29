"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { HoloPanel } from "./holo-panel"
import { Video, VideoOff, Hand } from "lucide-react"
import { classifyGesture, type GestureType, type HandLandmark } from "@/hooks/useGestureControl"

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

// ─── Component ────────────────────────────────────────────────────────────────

export function CameraPanel({
  onFrameData,
  compact = false,
}: {
  /**
   * Called each processed camera frame with the primary hand's raw landmarks,
   * classified gesture, and DOMHighResTimeStamp.
   * Receives empty landmarks array when no hand is detected.
   */
  onFrameData?: (landmarks: HandLandmark[], gesture: GestureType, nowMs: number) => void
  compact?: boolean
}) {
  const videoRef  = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // MediaPipe HandLandmarker instance (null until loaded)
  const landmarkerRef   = useRef<import("@mediapipe/tasks-vision").HandLandmarker | null>(null)
  const mpLoadingRef    = useRef(false)
  const lastDetectMsRef = useRef(0)

  const animRafRef      = useRef<number | null>(null)
  const lastFrameTsRef  = useRef(0)

  const [isStreaming,    setIsStreaming]    = useState(false)
  const [currentGesture, setCurrentGesture] = useState<GestureType>("NONE")
  const [handsDetected,  setHandsDetected]  = useState(0)
  const [mpReady,        setMpReady]        = useState(false)
  const [cameraError,    setCameraError]    = useState<string | null>(null)

  // ── MediaPipe loading ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isStreaming || mpLoadingRef.current || landmarkerRef.current) return
    mpLoadingRef.current = true
    let cancelled = false

    const load = async () => {
      try {
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision")
        const vision = await FilesetResolver.forVisionTasks(MP_WASM_URL)
        const hl = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MP_MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence:  0.5,
          minTrackingConfidence:      0.5,
        })
        if (cancelled) { hl.close(); return }
        landmarkerRef.current = hl
        setMpReady(true)
        console.log("[CameraPanel] MediaPipe HandLandmarker ready")
      } catch (err) {
        if (!cancelled) {
          console.warn("[CameraPanel] MediaPipe unavailable:", err)
          setCameraError("MediaPipe failed to load")
        }
      } finally {
        mpLoadingRef.current = false
      }
    }

    load()
    return () => {
      cancelled = true
      landmarkerRef.current?.close()
      landmarkerRef.current = null
      setMpReady(false)
    }
  }, [isStreaming])

  // ── Canvas drawing ────────────────────────────────────────────────────────────
  const drawLandmarks = useCallback(
    (ctx: CanvasRenderingContext2D, hands: DetectedHand[]) => {
      const { width, height } = ctx.canvas
      ctx.clearRect(0, 0, width, height)

      const CONNECTIONS: [number, number][] = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20],
        [5,9],[9,13],[13,17],
      ]

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
          if ([4, 8, 12, 16, 20].includes(i)) {
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
      }

      ctx.shadowBlur = 0
    },
    []
  )

  // ── Main animation loop ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isStreaming) return

    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const loop = (ts: number) => {
      animRafRef.current = requestAnimationFrame(loop)

      // ~30 fps cap
      if (ts - lastFrameTsRef.current < 33) return
      lastFrameTsRef.current = ts

      // Sync canvas size to video
      const vw = video.videoWidth  || 640
      const vh = video.videoHeight || 480
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width  = vw
        canvas.height = vh
      }

      let hands: DetectedHand[] = []
      const nowMs = performance.now()

      const hl = landmarkerRef.current
      if (hl && video.readyState >= 2 && video.videoWidth > 0) {
        // Real MediaPipe detection
        if (nowMs > lastDetectMsRef.current) {
          try {
            const result = hl.detectForVideo(video, nowMs)
            lastDetectMsRef.current = nowMs + 0.001  // ensure monotonic
            if (result.landmarks?.length > 0) {
              hands = result.landmarks.map((lmList, i) => ({
                landmarks:  lmList.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
                handedness: (result.handedness[i]?.[0]?.categoryName === "Left"
                  ? "Left"
                  : "Right") as "Left" | "Right",
              }))
            }
          } catch {
            // Occasionally fails on frame boundaries — ignore
          }
        }
      }

      setHandsDetected(hands.length)
      const primaryLandmarks = hands[0]?.landmarks ?? []
      const gesture = classifyGesture(primaryLandmarks)
      setCurrentGesture(gesture)

      // Always notify parent — empty landmarks signals hand loss
      onFrameData?.(primaryLandmarks, gesture, nowMs)

      drawLandmarks(ctx, hands)
    }

    animRafRef.current = requestAnimationFrame(loop)
    return () => {
      if (animRafRef.current) cancelAnimationFrame(animRafRef.current)
    }
  }, [isStreaming, onFrameData, drawLandmarks])

  // ── Camera start / stop ───────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
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
    landmarkerRef.current?.close()
    landmarkerRef.current = null
    setIsStreaming(false)
    setHandsDetected(0)
    setCurrentGesture("NONE")
    setMpReady(false)
    setCameraError(null)
  }

  // ── Gesture labels ────────────────────────────────────────────────────────────
  const GESTURE_LABELS: Record<GestureType, string> = {
    PINCH:     "Pinch — Rotate",
    OPEN_PALM: "Open Palm — Zoom",
    FIST:      "Fist — Freeze (hold)",
    POINT:     "Point — Navigate",
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
