"use client"

import { useEffect, useRef, useState } from "react"
import { HoloPanel } from "./holo-panel"
import { Maximize2, ZoomIn, Plus, Minus } from "lucide-react"
import type { Frame } from "@/lib/api"

interface SimulationProps {
  zoom?: number
  rotation?: { x: number; y: number }
  onZoomChange?: (zoom: number) => void
  compact?: boolean
  /** Live cylindrical POV frame from the renderer. When provided, the real
   *  frame is drawn as a polar overlay instead of the demo animation. */
  frame?: Frame
}

export function POVSimulation({ zoom = 1, rotation, onZoomChange, compact = false, frame }: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentZoom, setCurrentZoom] = useState(zoom)
  // Keep zoom in a ref so the animation loop reads the latest value without
  // restarting when zoom changes frequently (e.g. during gesture OPEN_PALM).
  const zoomRef = useRef(zoom)
  const animationRef = useRef<number | null>(null)
  const lastDrawTsRef = useRef(0)
  const timeRef = useRef(0)
  const rotationRef = useRef({ x: 0, y: 0 })
  // Hold latest frame in a ref so the animation loop can read it without
  // re-registering the effect every poll tick.
  const frameRef = useRef<Frame>(null)

  // Update zoom when prop changes — keep both state (for JSX) and ref (for canvas loop) in sync
  useEffect(() => {
    setCurrentZoom(zoom)
    zoomRef.current = zoom
  }, [zoom])

  // Update rotation when prop changes
  useEffect(() => {
    if (rotation) {
      rotationRef.current = rotation
    }
  }, [rotation])

  // Keep live frame ref in sync without restarting the animation loop
  useEffect(() => {
    frameRef.current = frame ?? null
  }, [frame])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight
      }
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // ASCII characters for the hologram effect
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*"

    const drawPOVFrame = (liveFrame: NonNullable<Frame>) => {
      const w = canvas.width
      const h = canvas.height
      const cx = w / 2
      const cy = h / 2
      const maxRadius = Math.min(cx, cy) * 0.88 * zoomRef.current

      ctx.fillStyle = "rgb(0, 10, 20)"
      ctx.fillRect(0, 0, w, h)

      const numAngles = liveFrame.length
      const numLeds = liveFrame[0]?.length ?? 0

      for (let a = 0; a < numAngles; a++) {
        const theta = (a / numAngles) * 2 * Math.PI
        for (let led = 0; led < numLeds; led++) {
          const pixel = liveFrame[a][led]
          if (!pixel) continue
          const [r, g, b] = pixel
          if (r + g + b < 12) continue
          const radius = ((led + 1) / numLeds) * maxRadius
          const x = cx + radius * Math.cos(theta)
          const y = cy + radius * Math.sin(theta)
          const dotSize = 1.5 + (led / numLeds) * 1.5
          ctx.fillStyle = `rgb(${r},${g},${b})`
          ctx.fillRect(x - dotSize / 2, y - dotSize / 2, dotSize, dotSize)
        }
      }

      // Ring overlay
      ctx.strokeStyle = "rgba(0,255,255,0.2)"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(cx, cy, maxRadius, 0, Math.PI * 2)
      ctx.stroke()

      // Live indicator
      ctx.fillStyle = "rgba(0,255,255,0.6)"
      ctx.font = "10px monospace"
      ctx.textAlign = "left"
      ctx.fillText("● LIVE", 8, 16)
    }

    const animate = (ts: number) => {
      if (ts - lastDrawTsRef.current < 33) {
        animationRef.current = requestAnimationFrame(animate)
        return
      }
      lastDrawTsRef.current = ts
      timeRef.current += 0.02

      const liveFrame = frameRef.current
      if (liveFrame && liveFrame.length > 0) {
        drawPOVFrame(liveFrame)
        animationRef.current = requestAnimationFrame(animate)
        return
      }

      // --- Demo animation (no live signal) ---
      ctx.fillStyle = "rgba(0, 20, 30, 0.1)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw grid
      ctx.strokeStyle = "rgba(0, 255, 255, 0.05)"
      ctx.lineWidth = 1
      const gridSize = 20

      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }

      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // Draw rotating hologram sphere
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const baseRadius = Math.min(canvas.width, canvas.height) * 0.35
      const radius = baseRadius * zoomRef.current

      ctx.font = `${Math.max(8, 10 * zoomRef.current)}px monospace`

      for (let i = 0; i < 200; i++) {
        const phi = (i / 200) * Math.PI * 2 + timeRef.current
        const theta = (i * 2.39996) % (Math.PI * 2)

        const x3d = Math.sin(theta) * Math.cos(phi)
        const y3d = Math.cos(theta)
        const z3d = Math.sin(theta) * Math.sin(phi)

        const rotY = timeRef.current + rotationRef.current.y * 0.01
        const rotX = rotationRef.current.x * 0.01

        let rotatedX = x3d * Math.cos(rotY) - z3d * Math.sin(rotY)
        let rotatedZ = x3d * Math.sin(rotY) + z3d * Math.cos(rotY)

        const rotatedY = y3d * Math.cos(rotX) - rotatedZ * Math.sin(rotX)
        rotatedZ = y3d * Math.sin(rotX) + rotatedZ * Math.cos(rotX)

        const depth = (rotatedZ + 1) / 2
        const x = centerX + rotatedX * radius
        const y = centerY + rotatedY * radius

        const alpha = 0.3 + depth * 0.7
        ctx.fillStyle = `rgba(0, 255, 255, ${alpha})`
        const char = chars[Math.floor(Math.random() * chars.length)]
        ctx.fillText(char, x, y)
      }

      for (let ring = 0; ring < 3; ring++) {
        const ringRadius = radius * (0.8 + ring * 0.15)
        const ringOffset = timeRef.current * (1 + ring * 0.3)

        ctx.beginPath()
        for (let i = 0; i <= 100; i++) {
          const angle = (i / 100) * Math.PI * 2 + ringOffset
          const x = centerX + Math.cos(angle) * ringRadius
          const y = centerY + Math.sin(angle) * ringRadius * 0.3
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.strokeStyle = `rgba(0, 255, 255, ${0.3 - ring * 0.08})`
        ctx.lineWidth = 2 - ring * 0.5
        ctx.stroke()
      }

      const texts = ["SYSTEM.INIT", "HOLOGRAM.ACTIVE", "SCAN.COMPLETE", "DATA.STREAM"]
      texts.forEach((text, i) => {
        const offset = timeRef.current * 0.5 + (i * Math.PI) / 2
        const textX = 20 + Math.sin(offset) * 10
        const textY = 30 + i * 20
        ctx.fillStyle = `rgba(0, 255, 255, ${0.5 + Math.sin(offset) * 0.3})`
        ctx.font = "12px monospace"
        ctx.textAlign = "left"
        ctx.fillText(text, textX, textY)
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate(0)

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange)
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange)
    }
  }, [])

  const toggleFullscreen = () => {
    const container = containerRef.current
    if (!container) return

    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(err => {
        console.error("Error attempting to enable fullscreen:", err)
      })
    } else {
      document.exitFullscreen()
    }
  }

  return (
    <HoloPanel title="360 Projection Window" statusIndicator className="h-full">
      <div 
        ref={containerRef}
        className={`relative bg-background/50 rounded overflow-hidden ${
          compact
            ? "h-full min-h-0"
            : "aspect-square md:aspect-auto md:h-[400px] lg:h-[500px]"
        }`}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          onWheel={(e) => {
            e.preventDefault()
            const zoomChange = e.deltaY * -0.001
            const newZoom = Math.max(0.5, Math.min(3, currentZoom + zoomChange))
            zoomRef.current = newZoom
            setCurrentZoom(newZoom)
            onZoomChange?.(newZoom)
          }}
        />

        {/* Fullscreen button */}
        <button
          onClick={toggleFullscreen}
          className="absolute top-2 right-2 p-2 rounded border border-primary/30 bg-background/80 text-primary hover:bg-primary/20 transition-colors z-10 cursor-pointer"
          aria-label="Toggle fullscreen"
        >
          <Maximize2 className="w-4 h-4" />
        </button>

        {/* Zoom indicator */}
        <div className="absolute top-2 left-2 flex items-center gap-2 px-2 py-1 rounded border border-primary/30 bg-background/80">
          <button
            onClick={() => {
              const newZoom = Math.max(0.5, currentZoom - 0.1)
              zoomRef.current = newZoom
              setCurrentZoom(newZoom)
              onZoomChange?.(newZoom)
            }}
            className="text-primary hover:bg-primary/20 p-0.5 rounded transition-colors cursor-pointer z-10"
            title="Zoom Out"
          >
            <Minus className="w-3 h-3" />
          </button>
          <ZoomIn className="w-3 h-3 text-primary" />
          <span className="text-xs font-mono text-primary w-[32px] text-center">
            {(currentZoom * 100).toFixed(0)}%
          </span>
          <button
            onClick={() => {
              const newZoom = Math.min(3, currentZoom + 0.1)
              zoomRef.current = newZoom
              setCurrentZoom(newZoom)
              onZoomChange?.(newZoom)
            }}
            className="text-primary hover:bg-primary/20 p-0.5 rounded transition-colors cursor-pointer z-10"
            title="Zoom In"
          >
            <Plus className="w-3 h-3" />
          </button>
        </div>

        {/* Decorative overlays — only shown in full (non-compact) mode */}
        {!compact && (
          <>
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <div className="rounded-full border border-primary/20 transition-all duration-300"
                style={{ width: `${80 * currentZoom}%`, height: `${80 * currentZoom}%`, maxWidth: "100%", maxHeight: "100%" }} />
              <div className="absolute rounded-full border border-primary/10 transition-all duration-300"
                style={{ width: `${60 * currentZoom}%`, height: `${60 * currentZoom}%`, maxWidth: "90%", maxHeight: "90%" }} />
              <div className="absolute rounded-full border border-primary/10 transition-all duration-300"
                style={{ width: `${40 * currentZoom}%`, height: `${40 * currentZoom}%`, maxWidth: "80%", maxHeight: "80%" }} />
            </div>
            <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary/40 pointer-events-none" />
            <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary/40 pointer-events-none" />
            <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary/40 pointer-events-none" />
            <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary/40 pointer-events-none" />
          </>
        )}
      </div>
    </HoloPanel>
  )
}
