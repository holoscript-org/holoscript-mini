"use client"

import { useEffect, useRef } from "react"
import type { Frame } from "@/lib/api"

interface POVCanvasProps {
  frame: Frame
  /** Canvas diameter in pixels (square). Default 500. */
  size?: number
  className?: string
}

/**
 * Renders a cylindrical POV frame (360 × 18 × 3 uint8) as a polar plot.
 * Angle index maps to rotation angle; LED index maps to radial distance
 * (0 = center, 17 = outer edge).
 */
export default function POVCanvas({ frame, size = 500, className }: POVCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const cx = w / 2
    const cy = h / 2
    const maxRadius = Math.min(cx, cy) - 4

    ctx.clearRect(0, 0, w, h)

    // Dark background
    ctx.fillStyle = "rgb(0, 10, 20)"
    ctx.fillRect(0, 0, w, h)

    if (!frame || frame.length === 0) {
      // No-signal state — draw a dim crosshair
      ctx.strokeStyle = "rgba(0,255,255,0.15)"
      ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(cx, 4); ctx.lineTo(cx, h - 4); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(4, cy); ctx.lineTo(w - 4, cy); ctx.stroke()
      ctx.fillStyle = "rgba(0,255,255,0.3)"
      ctx.font = "12px monospace"
      ctx.textAlign = "center"
      ctx.fillText("NO SIGNAL", cx, cy)
      return
    }

    const numAngles = frame.length         // 360
    const numLeds   = frame[0]?.length ?? 0  // 18

    for (let a = 0; a < numAngles; a++) {
      const theta = (a / numAngles) * 2 * Math.PI

      for (let led = 0; led < numLeds; led++) {
        const pixel = frame[a][led]
        if (!pixel) continue

        const [r, g, b] = pixel
        // Skip near-black pixels — they're background, not signal
        if (r + g + b < 12) continue

        const radius = ((led + 1) / numLeds) * maxRadius

        const x = cx + radius * Math.cos(theta)
        const y = cy + radius * Math.sin(theta)

        // Dot size scales with LED index (outer LEDs cover more area)
        const dotSize = 1.5 + (led / numLeds) * 1.5

        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fillRect(x - dotSize / 2, y - dotSize / 2, dotSize, dotSize)
      }
    }

    // Subtle outer ring
    ctx.strokeStyle = "rgba(0,255,255,0.2)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(cx, cy, maxRadius, 0, Math.PI * 2)
    ctx.stroke()
  }, [frame])

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className={className}
      style={{ width: "100%", height: "100%" }}
    />
  )
}
