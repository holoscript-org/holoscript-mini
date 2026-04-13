"use client"

import { useState, useCallback } from "react"
import { POVSimulation } from "@/components/pov-simulation"
import { CameraPanel } from "@/components/camera-panel"
import { CommandPanel } from "@/components/command-panel"

type GestureType = "PINCH" | "OPEN_PALM" | "FIST" | "POINTING" | "NONE"

interface DetectedHand {
  landmarks: { x: number; y: number; z: number }[]
  handedness: "Left" | "Right"
}

export default function HologramDashboard() {
  const [simulationZoom, setSimulationZoom] = useState(1)
  const [simulationRotation, setSimulationRotation] = useState({ x: 0, y: 0 })

  const handleGestureDetected = useCallback(
    (gesture: GestureType, hands: DetectedHand[]) => {
      switch (gesture) {
        case "PINCH":
          // Pinch to zoom
          setSimulationZoom((prev) => Math.min(prev + 0.02, 2))
          break
        case "OPEN_PALM":
          // Open palm to reset
          setSimulationZoom(1)
          setSimulationRotation({ x: 0, y: 0 })
          break
        case "FIST":
          // Fist to zoom out
          setSimulationZoom((prev) => Math.max(prev - 0.02, 0.5))
          break
        default:
          break
      }

      // Two hands for rotation
      if (hands.length === 2) {
        const hand1 = hands[0].landmarks[0]
        const hand2 = hands[1].landmarks[0]
        const deltaX = (hand2.x - hand1.x) * 100
        const deltaY = (hand2.y - hand1.y) * 100
        setSimulationRotation({ x: deltaY, y: deltaX })
      }
    },
    []
  )

  const handleCommandSend = useCallback((command: string) => {
    // Handle voice/text commands
    const lowerCommand = command.toLowerCase()

    if (lowerCommand.includes("zoom in")) {
      setSimulationZoom((prev) => Math.min(prev + 0.2, 2))
    } else if (lowerCommand.includes("zoom out")) {
      setSimulationZoom((prev) => Math.max(prev - 0.2, 0.5))
    } else if (lowerCommand.includes("reset")) {
      setSimulationZoom(1)
      setSimulationRotation({ x: 0, y: 0 })
    }
  }, [])

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_var(--glow)]" />
            <h1 className="text-xl font-mono font-bold text-primary tracking-wider">
              HoloScript
            </h1>
          </div>
          <nav className="hidden md:flex items-center gap-4 text-sm font-mono">
            <span className="text-muted-foreground hover:text-primary cursor-pointer transition-colors">
              Live Dashboard
            </span>
            <span className="flex items-center gap-1 text-primary">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Live
            </span>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_6px_rgba(0,255,0,0.5)]" />
            System Online
          </div>
          <button className="px-3 py-1.5 text-xs font-mono border border-primary/30 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
            Settings
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
        {/* Left: POV Simulation */}
        <div className="lg:row-span-2">
          <POVSimulation 
            zoom={simulationZoom} 
            rotation={simulationRotation} 
            onZoomChange={setSimulationZoom}
          />
        </div>

        {/* Right Top: Camera Panel with Gesture Control */}
        <div>
          <CameraPanel onGestureDetected={handleGestureDetected} />
        </div>

        {/* Right Bottom: Command Panel */}
        <div>
          <CommandPanel onCommandSend={handleCommandSend} />
        </div>
      </div>

      {/* Background grid overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-5 -z-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: "50px 50px",
        }}
      />

      {/* Scanline effect */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.02] -z-10"
        style={{
          backgroundImage: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 255, 0.3) 2px,
            rgba(0, 255, 255, 0.3) 4px
          )`,
        }}
      />
    </div>
  )
}
