"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { HoloPanel } from "./holo-panel"
import { Video, VideoOff, Hand } from "lucide-react"

interface HandLandmark {
  x: number
  y: number
  z: number
}

interface DetectedHand {
  landmarks: HandLandmark[]
  handedness: "Left" | "Right"
}

type GestureType = "PINCH" | "OPEN_PALM" | "FIST" | "POINTING" | "NONE"

export function CameraPanel({
  onGestureDetected,
  compact = false,
}: {
  onGestureDetected?: (gesture: GestureType, hands: DetectedHand[]) => void
  compact?: boolean
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentGesture, setCurrentGesture] = useState<GestureType>("NONE")
  const [handsDetected, setHandsDetected] = useState(0)
  const animationRef = useRef<number | null>(null)
  const lastDrawTsRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)

  // Simulated hand landmarks for demo (in production, use MediaPipe Hands)
  const simulatedHandsRef = useRef<DetectedHand[]>([])
  const timeRef = useRef(0)

  const detectGesture = useCallback((hands: DetectedHand[]): GestureType => {
    if (hands.length === 0) return "NONE"

    const hand = hands[0]
    if (!hand.landmarks || hand.landmarks.length < 21) return "NONE"

    // Simplified gesture detection based on finger positions
    const thumb = hand.landmarks[4]
    const index = hand.landmarks[8]
    const middle = hand.landmarks[12]
    const ring = hand.landmarks[16]
    const pinky = hand.landmarks[20]
    const wrist = hand.landmarks[0]

    // Calculate distances
    const thumbIndexDist = Math.sqrt(
      Math.pow(thumb.x - index.x, 2) + Math.pow(thumb.y - index.y, 2)
    )

    const fingerSpread =
      Math.sqrt(Math.pow(index.x - pinky.x, 2) + Math.pow(index.y - pinky.y, 2))

    const fingersUp = [index, middle, ring, pinky].filter(
      (f) => f.y < wrist.y - 0.1
    ).length

    // Detect gestures
    if (thumbIndexDist < 0.05) return "PINCH"
    if (fingersUp >= 4 && fingerSpread > 0.15) return "OPEN_PALM"
    if (fingersUp <= 1 && fingerSpread < 0.1) return "FIST"
    if (fingersUp === 1 && index.y < wrist.y - 0.15) return "POINTING"

    return "NONE"
  }, [])

  const drawHandLandmarks = useCallback(
    (ctx: CanvasRenderingContext2D, hands: DetectedHand[]) => {
      const canvas = ctx.canvas

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      hands.forEach((hand, handIndex) => {
        const color = hand.handedness === "Left" ? "#00ffff" : "#00ff88"

        // Draw connections
        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
          [0, 5], [5, 6], [6, 7], [7, 8], // Index
          [0, 9], [9, 10], [10, 11], [11, 12], // Middle
          [0, 13], [13, 14], [14, 15], [15, 16], // Ring
          [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
          [5, 9], [9, 13], [13, 17], // Palm
        ]

        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.shadowColor = color
        ctx.shadowBlur = 8

        connections.forEach(([start, end]) => {
          const startLandmark = hand.landmarks[start]
          const endLandmark = hand.landmarks[end]

          ctx.beginPath()
          ctx.moveTo(startLandmark.x * canvas.width, startLandmark.y * canvas.height)
          ctx.lineTo(endLandmark.x * canvas.width, endLandmark.y * canvas.height)
          ctx.stroke()
        })

        // Draw landmarks
        hand.landmarks.forEach((landmark, i) => {
          const x = landmark.x * canvas.width
          const y = landmark.y * canvas.height
          const radius = i === 0 ? 6 : 4

          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = color
          ctx.fill()

          // Add glow effect to fingertips
          if ([4, 8, 12, 16, 20].includes(i)) {
            ctx.beginPath()
            ctx.arc(x, y, 8, 0, Math.PI * 2)
            ctx.strokeStyle = color
            ctx.lineWidth = 1
            ctx.stroke()
          }
        })

        // Draw hand label
        const wrist = hand.landmarks[0]
        ctx.font = "12px monospace"
        ctx.fillStyle = color
        ctx.shadowBlur = 4
        ctx.fillText(
          `${hand.handedness} Hand`,
          wrist.x * canvas.width - 30,
          wrist.y * canvas.height + 30
        )
      })

      ctx.shadowBlur = 0
    },
    []
  )

  const generateSimulatedHands = useCallback(() => {
    timeRef.current += 0.03
    const t = timeRef.current

    // Simulate one hand with natural movement
    const baseX = 0.5 + Math.sin(t * 0.5) * 0.15
    const baseY = 0.5 + Math.cos(t * 0.7) * 0.1

    // Generate 21 landmarks for a hand
    const landmarks: HandLandmark[] = []

    // Wrist
    landmarks.push({ x: baseX, y: baseY + 0.15, z: 0 })

    // Thumb (4 landmarks)
    for (let i = 0; i < 4; i++) {
      landmarks.push({
        x: baseX - 0.08 - i * 0.02,
        y: baseY + 0.1 - i * 0.03 + Math.sin(t * 2 + i) * 0.01,
        z: 0,
      })
    }

    // Fingers (4 landmarks each)
    const fingerOffsets = [0, 0.025, 0.05, 0.075]
    for (let f = 0; f < 4; f++) {
      const fingerX = baseX - 0.03 + f * 0.035
      for (let i = 0; i < 4; i++) {
        const curl = Math.sin(t * 1.5 + f * 0.5) * 0.02
        landmarks.push({
          x: fingerX + Math.sin(t + f) * 0.005,
          y: baseY - i * 0.035 + curl * i,
          z: 0,
        })
      }
    }

    const hands: DetectedHand[] = [
      { landmarks, handedness: "Right" },
    ]

    // Sometimes add a second hand
    if (Math.sin(t * 0.3) > 0.5) {
      const leftHandLandmarks = landmarks.map((l) => ({
        x: 1 - l.x + 0.1,
        y: l.y + Math.sin(t) * 0.02,
        z: l.z,
      }))
      hands.push({ landmarks: leftHandLandmarks, handedness: "Left" })
    }

    return hands
  }, [])

  useEffect(() => {
    if (!isStreaming) return

    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const animate = (ts: number) => {
      if (ts - lastDrawTsRef.current < 50) {
        animationRef.current = requestAnimationFrame(animate)
        return
      }
      lastDrawTsRef.current = ts

      // Update canvas size to match video
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      } else {
        canvas.width = 640
        canvas.height = 480
      }

      // Generate simulated hands (replace with MediaPipe in production)
      const hands = generateSimulatedHands()
      simulatedHandsRef.current = hands

      // Update state
      setHandsDetected(hands.length)
      const gesture = detectGesture(hands)
      setCurrentGesture(gesture)

      // Notify parent
      if (onGestureDetected) {
        onGestureDetected(gesture, hands)
      }

      // Draw hand landmarks
      drawHandLandmarks(ctx, hands)

      animationRef.current = requestAnimationFrame(animate)
    }

    animate(0)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isStreaming, detectGesture, drawHandLandmarks, generateSimulatedHands, onGestureDetected])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreaming(true)
      }
    } catch (err) {
      console.error("Failed to access camera:", err)
      // Still enable simulation mode
      setIsStreaming(true)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setHandsDetected(0)
    setCurrentGesture("NONE")
  }

  const gestureLabels: Record<GestureType, string> = {
    PINCH: "Pinch - Zoom",
    OPEN_PALM: "Open Palm - Reset",
    FIST: "Fist - Grab",
    POINTING: "Pointing - Select",
    NONE: "No Gesture",
  }

  return (
    <HoloPanel title="Gesture Control" statusIndicator className="h-full">
      <div
        className={`relative bg-background/50 rounded overflow-hidden ${
          compact ? "h-full min-h-0" : "aspect-video"
        }`}
      >
        {/* Video feed */}
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

        {/* Canvas overlay for hand tracking */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Placeholder when not streaming — full-panel click target */}
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
            </button>
          </div>
        )}

        {/* Stop button — only shown while streaming */}
        {isStreaming && (
          <div className="absolute top-2 right-2">
            <button
              onClick={stopCamera}
              className="p-2 rounded border border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/25 transition-colors"
              aria-label="Stop camera"
            >
              <VideoOff className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Gesture detection label */}
        {isStreaming && (
          <div className="absolute bottom-2 left-2 right-2">
            <div className="flex items-center justify-between gap-2 px-3 py-2 rounded border border-primary/30 bg-background/90 backdrop-blur-sm">
              <div className="flex items-center gap-2">
                <Hand className="w-4 h-4 text-primary" />
                <span className="text-xs font-mono text-muted-foreground">
                  Hands: {handsDetected}
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
                  {gestureLabels[currentGesture]}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Scanning overlay effect */}
        {isStreaming && (
          <div className="absolute inset-0 pointer-events-none">
            <div
              className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent opacity-50"
              style={{
                animation: "scan 2s linear infinite",
              }}
            />
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes scan {
          0% {
            top: 0;
          }
          100% {
            top: 100%;
          }
        }
      `}</style>
    </HoloPanel>
  )
}
