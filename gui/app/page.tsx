"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { POVSimulation } from "@/components/pov-simulation"
import { CameraPanel } from "@/components/camera-panel"
import { CommandPanel } from "@/components/command-panel"
import { SceneSummary } from "@/components/scene-summary"
import { SceneConfiguration } from "@/components/scene-configuration"
import { ThreeScene } from "@/components/ThreeScene"
import { PipelineOverlay } from "@/components/pipeline-overlay/PipelineOverlay"
import { useWebGLSceneData } from "@/hooks/useWebGLSceneData"
import { useGestureControl } from "@/hooks/useGestureControl"
import { usePipelineStream } from "@/hooks/usePipelineStream"

export default function HologramDashboard() {
  // ── Scene data (JSON, POV frame) ─────────────────────────────────────────────
  const {
    frame,
    scene,
    connected,
    selectedScene,
    sceneOptions,
    setSelectedScene,
    refreshFromBackend,
  } = useWebGLSceneData()

  // ── Drag-active ref — true only while PINCH is held after a selection exists ─
  const isDragActiveRef = useRef(false)
  // ── Track selected object id and scene-drag mode for use in callbacks ─────
  const selectedIdRef = useRef<string | null>(null)
  const sceneDragModeRef = useRef(false)

  // handleReset ref — allows useGestureControl to call it before it is defined
  const handleResetRef = useRef<() => void>(() => {})
  const onVSignReset = useCallback(() => handleResetRef.current(), [])

  // ── Browser-side gesture transforms ──────────────────────────────────────────
  const { state: gestureState, controls: gestureControls, applyWorkerUpdate, reset, setScale } = useGestureControl({ onVSignReset })

  // ── Three.js FPS (reported by ThreeScene; retained for the render loop) ─────
  const [, setThreeFps] = useState(0)

  // ── Pinch-to-select state ────────────────────────────────────────────────────
  const selectSeqRef = useRef(0)
  const [selectTrigger, setSelectTrigger] = useState<{ x: number; y: number; seq: number } | null>(null)
  const [reticlePoint, setReticlePoint] = useState<{ x: number; y: number } | null>(null)
  const lastReticleMsRef = useRef(0)
  const [sceneDragMode, setSceneDragMode] = useState(false)
  const [resetSeq, setResetSeq] = useState(0)
  const [deselectSeq, setDeselectSeq] = useState(0)

  // Keep refs in sync for use inside callbacks without stale closure issues
  sceneDragModeRef.current = sceneDragMode

  // Gesture flow:
  //   POINT (no selection)     → reticle only
  //   PINCH in point window    → select object under cursor          (onPinchSelect fires BEFORE onGestureUpdate)
  //   POINT (with selection)   → drag: object follows reticle
  //   PINCH while dragging     → drop + deselect                    (isDragActiveRef still true when handler fires)
  const handleGestureUpdate = useCallback(
    (rotDelta: number, scaleDelta: number, gesture: import("@/hooks/useGestureControl").GestureType, nowMs: number) => {
      applyWorkerUpdate(rotDelta, scaleDelta, gesture, nowMs)
      // POINT while something is targeted → activate drag
      if (gesture === "POINT" && (selectedIdRef.current !== null || sceneDragModeRef.current)) {
        isDragActiveRef.current = true
      } else if (gesture !== "POINT") {
        // Any non-POINT gesture stops drag (drop fires via handlePinchSelect first)
        isDragActiveRef.current = false
      }
    },
    [applyWorkerUpdate]
  )

  // Fires BEFORE handleGestureUpdate for the same frame (camera-panel ordering).
  // PINCH in point window while dragging → DROP.
  // PINCH in point window while not dragging → SELECT.
  const handlePinchSelect = useCallback((x: number, y: number) => {
    if (isDragActiveRef.current) {
      // Drop: stop drag and deselect so user can re-target next
      isDragActiveRef.current = false
      selectedIdRef.current = null
      setDeselectSeq((prev) => prev + 1)
    } else {
      // Nothing being dragged → run selection raycast
      setSelectTrigger({ x, y, seq: ++selectSeqRef.current })
    }
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
    isDragActiveRef.current = false
    selectedIdRef.current = null
    setSceneDragMode(false)
    setReticlePoint(null)
    setSelectTrigger(null)
    setResetSeq((prev) => prev + 1)
    // resetSeq already clears selectedId inside ThreeScene; no need to bump deselectSeq
  }, [reset])

  // Keep the ref pointing at the latest handleReset so V_SIGN can call it
  handleResetRef.current = handleReset

  const handleObjectSelect = useCallback((id: string | null) => {
    selectedIdRef.current = id
    // If deselected, stop any active drag
    if (!id) isDragActiveRef.current = false
  }, [])

  // ── Zoom sync: POV wheel/buttons feed back into gesture scale ────────────────
  const handleZoomChange = useCallback(
    (newZoom: number) => { setScale(newZoom) },
    [setScale]
  )

  // ── Command panel integration — sends command to Member 1 pipeline ───────────
  // Live progress streams over WebSocket (see hooks/usePipelineStream.ts and
  // the PipelineOverlay rendered below) instead of the old fetch+poll loop.
  const pipeline = usePipelineStream()

  const handleCommandSend = useCallback(
    async (command: string) => {
      const cmd = command.toLowerCase()
      if (cmd === "reset" || cmd === "reset scene") {
        handleReset()
        return
      }
      pipeline.start(command)
    },
    [handleReset, pipeline]
  )

  // On successful completion, refresh the dashboard's scene state. We still
  // call refreshFromBackend() (rather than only trusting the scene inlined in
  // the WS run_finished message) since it may sync more dashboard state than
  // just the raw scene JSON — the inlined scene is treated as a fast-path
  // preview for the overlay's own success view, not a full replacement.
  const pipelineDoneRunIdRef = useRef<string | null>(null)
  useEffect(() => {
    if (pipeline.state.status !== "done" || !pipeline.state.runId) return
    if (pipelineDoneRunIdRef.current === pipeline.state.runId) return
    pipelineDoneRunIdRef.current = pipeline.state.runId
    void refreshFromBackend()
  }, [pipeline.state.status, pipeline.state.runId, refreshFromBackend])

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
            isDragActiveRef={isDragActiveRef}
            resetSeq={resetSeq}
            deselectSeq={deselectSeq}
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
                onGestureUpdate={handleGestureUpdate}
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

          {/* Bottom: scene config + educational summary */}
          <div className="grid min-h-0 grid-cols-2 gap-3 md:gap-4">
            <div className="min-h-0">
              <SceneConfiguration liveScene={scene} />
            </div>
            <div className="min-h-0">
              <SceneSummary scene={scene} />
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

      {/* Live pipeline progress overlay — appears on command submit, shows
          every generation stage, dismisses into the hydrated scene above */}
      <PipelineOverlay state={pipeline.state} onDismiss={pipeline.dismiss} />
    </div>
  )
}
