"use client"

import { useEffect, useRef, useState } from "react"
import { HoloPanel } from "./holo-panel"
import { Maximize2, ZoomIn, Plus, Minus } from "lucide-react"

interface SimulationProps {
  zoom?: number
  rotation?: { x: number; y: number }
  onZoomChange?: (zoom: number) => void
}

export function POVSimulation({ zoom = 1, rotation, onZoomChange }: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentZoom, setCurrentZoom] = useState(zoom)
  const animationRef = useRef<number | null>(null)
  const timeRef = useRef(0)
  const rotationRef = useRef({ x: 0, y: 0 })

  // Update zoom when prop changes
  useEffect(() => {
    setCurrentZoom(zoom)
  }, [zoom])

  // Update rotation when prop changes
  useEffect(() => {
    if (rotation) {
      rotationRef.current = rotation
    }
  }, [rotation])

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

    const animate = () => {
      timeRef.current += 0.02

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
      const radius = baseRadius * currentZoom

      // Draw wireframe sphere
      ctx.font = `${Math.max(8, 10 * currentZoom)}px monospace`

      for (let i = 0; i < 200; i++) {
        const phi = (i / 200) * Math.PI * 2 + timeRef.current
        const theta = (i * 2.39996) % (Math.PI * 2)

        const x3d = Math.sin(theta) * Math.cos(phi)
        const y3d = Math.cos(theta)
        const z3d = Math.sin(theta) * Math.sin(phi)

        // Apply external rotation
        const rotY = timeRef.current + rotationRef.current.y * 0.01
        const rotX = rotationRef.current.x * 0.01

        // Rotate around Y axis
        let rotatedX = x3d * Math.cos(rotY) - z3d * Math.sin(rotY)
        let rotatedZ = x3d * Math.sin(rotY) + z3d * Math.cos(rotY)

        // Rotate around X axis
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

      // Draw orbital rings
      for (let ring = 0; ring < 3; ring++) {
        const ringRadius = radius * (0.8 + ring * 0.15)
        const ringOffset = timeRef.current * (1 + ring * 0.3)

        ctx.beginPath()
        for (let i = 0; i <= 100; i++) {
          const angle = (i / 100) * Math.PI * 2 + ringOffset
          const x = centerX + Math.cos(angle) * ringRadius
          const y = centerY + Math.sin(angle) * ringRadius * 0.3

          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        }
        ctx.strokeStyle = `rgba(0, 255, 255, ${0.3 - ring * 0.08})`
        ctx.lineWidth = 2 - ring * 0.5
        ctx.stroke()
      }

      // Draw floating text
      const texts = [
        "SYSTEM.INIT",
        "HOLOGRAM.ACTIVE",
        "SCAN.COMPLETE",
        "DATA.STREAM",
      ]

      texts.forEach((text, i) => {
        const offset = timeRef.current * 0.5 + (i * Math.PI) / 2
        const textX = 20 + Math.sin(offset) * 10
        const textY = 30 + i * 20

        ctx.fillStyle = `rgba(0, 255, 255, ${0.5 + Math.sin(offset) * 0.3})`
        ctx.font = "12px monospace"
        ctx.fillText(text, textX, textY)
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [currentZoom])

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
    <HoloPanel title="360° Simulation" statusIndicator className="h-full">
      <div 
        ref={containerRef}
        className="relative aspect-square md:aspect-auto md:h-[400px] lg:h-[500px] bg-background/50 rounded overflow-hidden"
      >
        <canvas 
          ref={canvasRef} 
          className="w-full h-full"
          onWheel={(e) => {
            e.preventDefault()
            const zoomChange = e.deltaY * -0.001
            const newZoom = Math.max(0.5, Math.min(3, currentZoom + zoomChange))
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
              setCurrentZoom(newZoom)
              onZoomChange?.(newZoom)
            }}
            className="text-primary hover:bg-primary/20 p-0.5 rounded transition-colors cursor-pointer z-10"
            title="Zoom In"
          >
            <Plus className="w-3 h-3" />
          </button>
        </div>

        {/* Circular overlay */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          <div
            className="rounded-full border border-primary/20 transition-all duration-300"
            style={{
              width: `${80 * currentZoom}%`,
              height: `${80 * currentZoom}%`,
              maxWidth: "100%",
              maxHeight: "100%",
            }}
          />
          <div
            className="absolute rounded-full border border-primary/10 transition-all duration-300"
            style={{
              width: `${60 * currentZoom}%`,
              height: `${60 * currentZoom}%`,
              maxWidth: "90%",
              maxHeight: "90%",
            }}
          />
          <div
            className="absolute rounded-full border border-primary/10 transition-all duration-300"
            style={{
              width: `${40 * currentZoom}%`,
              height: `${40 * currentZoom}%`,
              maxWidth: "80%",
              maxHeight: "80%",
            }}
          />
        </div>

        {/* Corner markers */}
        <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary/40 pointer-events-none" />
        <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary/40 pointer-events-none" />
        <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary/40 pointer-events-none" />
        <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary/40 pointer-events-none" />
      </div>
    </HoloPanel>
  )
}
