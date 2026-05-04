"use client"

import { useCallback, useRef, useState } from "react"
import { POVSimulation } from "@/components/pov-simulation"
import { CameraPanel } from "@/components/camera-panel"
import { CommandPanel } from "@/components/command-panel"
import { PerformanceLogs } from "@/components/performance-logs"
import { SceneConfiguration } from "@/components/scene-configuration"
import { ThreeScene } from "@/components/ThreeScene"
import { useWebGLSceneData } from "@/hooks/useWebGLSceneData"
import { useGestureControl } from "@/hooks/useGestureControl"

export default function HologramDashboard() {
  // ── Scene data (JSON, POV frame, logs) ───────────────────────────────────────
  const {
    frame,
    scene,
    logs,
    connected,
    selectedScene,
    sceneOptions,
    setSelectedScene,
  } = useWebGLSceneData()

  // ── Browser-side gesture transforms ──────────────────────────────────────────
  const { state: gestureState, controls: gestureControls, applyWorkerUpdate, reset, setScale } = useGestureControl()

  // ── Three.js FPS (passed up from ThreeScene → shown in PerformanceLogs) ──────
  const [threeFps, setThreeFps] = useState(0)

  // ── Pinch-to-select state ────────────────────────────────────────────────────
  const selectSeqRef = useRef(0)
  const [selectTrigger, setSelectTrigger] = useState<{ x: number; y: number; seq: number } | null>(null)
  const [reticlePoint, setReticlePoint] = useState<{ x: number; y: number } | null>(null)
  const lastReticleMsRef = useRef(0)
  const [sceneDragMode, setSceneDragMode] = useState(false)
  const [resetSeq, setResetSeq] = useState(0)

  const handlePinchSelect = useCallback((x: number, y: number) => {
    setSelectTrigger({ x, y, seq: ++selectSeqRef.current })
  }, [])

  const handlePointMove = useCallback((point: { x: number; y: number } | null, nowMs: number) => {
    if (nowMs - lastReticleMsRef.current < 33) return
    lastReticleMsRef.current = nowMs
    setReticlePoint(point)
  }, [])

  const handlePointHoldToggle = useCallback(() => {
    setSceneDragMode((prev) => !prev)
  }, [])

  const handleReset = useCallback(() => {
    reset()
    setSceneDragMode(false)
    setReticlePoint(null)
    setSelectTrigger(null)
    setResetSeq((prev) => prev + 1)
  }, [reset])

  const handleObjectSelect = useCallback((id: string | null) => {
    // Selection state is managed inside ThreeScene; log here for debugging
    if (id) console.log("[Selection]", id)
  }, [])

  // ── Zoom sync: POV wheel/buttons feed back into gesture scale ────────────────
  const handleZoomChange = useCallback(
    (newZoom: number) => { setScale(newZoom) },
    [setScale]
  )

  // ── Command panel integration (text commands affect transforms) ───────────────
  const handleCommandSend = useCallback(
    (command: string) => {
      const cmd = command.toLowerCase()
      if (cmd.includes("reset")) handleReset()
    },
    [handleReset]
  )

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
          {/* Scene connection indicator */}
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span
              className={`w-2 h-2 rounded-full transition-colors ${
                connected
                  ? "bg-green-500 shadow-[0_0_6px_rgba(0,255,0,0.5)]"
                  : "bg-yellow-500 shadow-[0_0_6px_rgba(255,200,0,0.4)]"
              }`}
            />
            {connected ? "WebGL Scene Loaded" : "Loading Scene"}
          </div>

          {/* Scene selector */}
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-primary/60">Scene JSON</span>
            <select
              value={selectedScene}
              onChange={(e) => setSelectedScene(e.target.value)}
              className="min-w-[220px] rounded border border-primary/30 bg-background/80 px-2 py-1 text-xs font-mono text-primary focus:outline-none focus:border-primary"
            >
              {sceneOptions.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Live gesture badge */}
          {gestureState.gesture !== "NONE" && (
            <div className="hidden sm:flex items-center gap-1 px-2 py-1 rounded border border-primary/30 bg-primary/10 text-xs font-mono text-primary">
              {gestureState.gesture}
            </div>
          )}

          {/* Frozen indicator */}
          {gestureState.frozen && (
            <div className="hidden sm:flex items-center gap-1 px-2 py-1 rounded border border-blue-500/50 bg-blue-500/10 text-xs font-mono text-blue-400">
              FROZEN
            </div>
          )}

          {sceneDragMode && (
            <div className="hidden sm:flex items-center gap-1 px-2 py-1 rounded border border-yellow-500/50 bg-yellow-500/10 text-xs font-mono text-yellow-300">
              SCENE DRAG
            </div>
          )}

          <button
            onClick={handleReset}
            className="px-3 py-1.5 text-xs font-mono border border-primary/30 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
          >
            Reset
          </button>
        </div>
      </header>

      <div className="grid h-[calc(100vh-86px)] min-h-0 grid-cols-1 xl:grid-cols-[1fr_1.08fr] gap-3 md:gap-4">
        {/* Left: WebGL 3D scene — transforms driven by browser gesture */}
        <div className="min-h-0">
          <ThreeScene
            scene={scene}
            controls={gestureControls}
            frozen={gestureState.frozen}
            selectTrigger={selectTrigger}
            reticlePoint={reticlePoint}
            sceneDragMode={sceneDragMode}
            resetSeq={resetSeq}
            onObjectSelect={handleObjectSelect}
            onFpsUpdate={setThreeFps}
          />
        </div>

        {/* Right: stacked panel layout */}
        <div className="grid min-h-0 grid-rows-[minmax(0,1.28fr)_minmax(0,0.78fr)_minmax(0,0.78fr)] gap-3 md:gap-4">
          {/* Top: POV simulation + gesture camera */}
          <div className="grid min-h-0 grid-cols-[0.85fr_1.15fr] gap-3 md:gap-4">
            <div className="min-h-0">
              {/* zoom prop synced with gesture scale; wheel/buttons feed back via setScale */}
              <POVSimulation
                zoom={gestureState.scale}
                onZoomChange={handleZoomChange}
                frame={frame}
                compact
              />
            </div>
            <div className="min-h-0">
              <CameraPanel
                onGestureUpdate={applyWorkerUpdate}
                onPinchSelect={handlePinchSelect}
                onPointMove={handlePointMove}
                onPointHoldToggle={handlePointHoldToggle}
                resetSeq={resetSeq}
                compact
              />
            </div>
          </div>

          {/* Middle: command panel */}
          <div className="min-h-0">
            <CommandPanel onCommandSend={handleCommandSend} />
          </div>

          {/* Bottom: scene config + logs */}
          <div className="grid min-h-0 grid-cols-2 gap-3 md:gap-4">
            <div className="min-h-0">
              <SceneConfiguration liveScene={scene} />
            </div>
            <div className="min-h-0">
              <PerformanceLogs liveLogs={logs} guiFps={threeFps} />
            </div>
          </div>
        </div>
      </div>

      {/* Background grid */}
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
