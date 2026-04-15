"use client"

import { useState, useCallback } from "react"
import { POVSimulation } from "@/components/pov-simulation"
import { CameraPanel } from "@/components/camera-panel"
import { CommandPanel } from "@/components/command-panel"
import { PerformanceLogs } from "@/components/performance-logs"
import { SceneConfiguration } from "@/components/scene-configuration"
import { RenderWindowPanel } from "@/components/render-window-panel"
import { useSceneData } from "@/hooks/useSceneData"

type GestureType = "PINCH" | "OPEN_PALM" | "FIST" | "POINTING" | "NONE"

interface DetectedHand {
  landmarks: { x: number; y: number; z: number }[]
  handedness: "Left" | "Right"
}

export default function HologramDashboard() {
  const [simulationZoom, setSimulationZoom] = useState(1)
  const [simulationRotation, setSimulationRotation] = useState({ x: 0, y: 0 })

  // Live data from the Python renderer via FastAPI
  const { frame, scene, logs, status, connected } = useSceneData(200)

  const handleGestureDetected = useCallback(
    (gesture: GestureType, hands: DetectedHand[]) => {
      switch (gesture) {
        case "PINCH":
          setSimulationZoom((prev) => Math.min(prev + 0.02, 2))
          break
        case "OPEN_PALM":
          setSimulationZoom(1)
          setSimulationRotation({ x: 0, y: 0 })
          break
        case "FIST":
          setSimulationZoom((prev) => Math.max(prev - 0.02, 0.5))
          break
        default:
          break
      }
      if (hands.length === 2) {
        const hand1 = hands[0].landmarks[0]
        const hand2 = hands[1].landmarks[0]
        setSimulationRotation({
          x: (hand2.y - hand1.y) * 100,
          y: (hand2.x - hand1.x) * 100,
        })
      }
    },
    []
  )

  const handleCommandSend = useCallback((command: string) => {
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
    <div className="h-screen overflow-hidden bg-background p-3 md:p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-3">
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
          {/* Backend connection indicator */}
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span
              className={`w-2 h-2 rounded-full transition-colors ${
                connected
                  ? "bg-green-500 shadow-[0_0_6px_rgba(0,255,0,0.5)]"
                  : "bg-yellow-500 shadow-[0_0_6px_rgba(255,200,0,0.4)]"
              }`}
            />
            {connected ? "Renderer Online" : "Demo Mode"}
          </div>
          {/* Active gesture badge */}
          {status.gesture !== "NONE" && (
            <div className="hidden sm:flex items-center gap-1 px-2 py-1 rounded border border-primary/30 bg-primary/10 text-xs font-mono text-primary">
              {status.gesture}
            </div>
          )}
          <button className="px-3 py-1.5 text-xs font-mono border border-primary/30 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
            Settings
          </button>
        </div>
      </header>

      <div className="grid h-[calc(100vh-86px)] min-h-0 grid-cols-1 xl:grid-cols-[1fr_1.08fr] gap-3 md:gap-4">
        {/* Left: Main renderer window (largest) */}
        <div className="min-h-0">
          <RenderWindowPanel />
        </div>

        {/* Right: Figma-style stacked layout */}
        <div className="grid min-h-0 grid-rows-[minmax(0,1.28fr)_minmax(0,0.78fr)_minmax(0,0.78fr)] gap-3 md:gap-4">
          {/* Top row: 360 projection + gesture control */}
          <div className="grid min-h-0 grid-cols-[0.85fr_1.15fr] gap-3 md:gap-4">
            <div className="min-h-0">
              <POVSimulation
                zoom={simulationZoom}
                rotation={simulationRotation}
                onZoomChange={setSimulationZoom}
                frame={frame}
                compact
              />
            </div>
            <div className="min-h-0">
              <CameraPanel onGestureDetected={handleGestureDetected} compact />
            </div>
          </div>

          {/* Middle row: command + voice */}
          <div className="min-h-0">
            <CommandPanel onCommandSend={handleCommandSend} />
          </div>

          {/* Bottom row: scene + logs */}
          <div className="grid min-h-0 grid-cols-2 gap-3 md:gap-4">
            <div className="min-h-0">
              <SceneConfiguration liveScene={scene} />
            </div>
            <div className="min-h-0">
              <PerformanceLogs liveLogs={logs} />
            </div>
          </div>
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
