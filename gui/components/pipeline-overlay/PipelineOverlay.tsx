"use client"

import { useEffect, useMemo, useState } from "react"
import { cn } from "@/lib/utils"
import type { PipelineRunState } from "@/hooks/usePipelineStream"
import { StageTimeline } from "./StageTimeline"
import { PromptOptimizerView } from "./stages/PromptOptimizerView"
import { IntentExtractionView } from "./stages/IntentExtractionView"
import { AssetResolutionView } from "./stages/AssetResolutionView"
import { ArchitectLayoutView } from "./stages/ArchitectLayoutView"
import { ArchitectDetailView } from "./stages/ArchitectDetailView"
import { ArchitectFinishView } from "./stages/ArchitectFinishView"
import { CriticLoopView } from "./stages/CriticLoopView"
import { ValidationRepairView } from "./stages/ValidationRepairView"

interface PipelineOverlayProps {
  state: PipelineRunState
  onDismiss: () => void
}

const RESOLUTION_STAGES = ["semantic_parse", "resolve_intent", "asset_registry", "live_search"]

// Which stage panel to show for a given "focused" stage id — resolution
// stages share one combined view since they're fast/mechanical.
function viewForStage(stageId: string): "optimizer" | "extraction" | "resolution" | "layout" | "detail" | "finish" | "critic" | "finalize" {
  if (stageId === "prompt_optimizer" || stageId === "receive_transcript") return "optimizer"
  if (stageId === "intent_extraction") return "extraction"
  if (RESOLUTION_STAGES.includes(stageId)) return "resolution"
  if (stageId === "architect_layout") return "layout"
  if (stageId === "architect_detail") return "detail"
  if (stageId === "architect_finish") return "finish"
  if (stageId.startsWith("critic_iteration_")) return "critic"
  return "finalize"
}

// Order in which the focus should auto-advance as stages complete — the most
// recently *active or completed* stage becomes the default focus so the
// panel always shows what's currently happening without the user clicking.
const AUTO_FOCUS_ORDER = [
  "prompt_optimizer",
  "intent_extraction",
  "live_search",
  "asset_registry",
  "resolve_intent",
  "semantic_parse",
  "architect_layout",
  "architect_detail",
  "architect_finish",
  "critic_iteration_3",
  "critic_iteration_2",
  "critic_iteration_1",
  "repair",
  "validate",
]

export function PipelineOverlay({ state, onDismiss }: PipelineOverlayProps) {
  const [manualFocus, setManualFocus] = useState<string | null>(null)
  const [closing, setClosing] = useState(false)

  const isOpen = state.status !== "idle"

  const autoFocus = useMemo(() => {
    for (const id of AUTO_FOCUS_ORDER) {
      if (state.stageByName[id]?.length) return id
    }
    return "prompt_optimizer"
  }, [state.stageByName])

  const focusedStage = manualFocus ?? autoFocus
  const activeStage = useMemo(() => {
    const last = state.events[state.events.length - 1]
    return last && last.status !== "completed" && last.status !== "failed" ? last.stage : null
  }, [state.events])

  // Reset manual focus pin whenever a new run starts.
  useEffect(() => {
    if (state.status === "running" && state.events.length === 0) {
      setManualFocus(null)
    }
  }, [state.status, state.events.length])

  // Auto-dismiss ~1.8s after success, with a brief "success beat" first.
  useEffect(() => {
    if (state.status !== "done") return
    const timer = setTimeout(() => {
      setClosing(true)
      setTimeout(() => {
        onDismiss()
        setClosing(false)
      }, 300)
    }, 1800)
    return () => clearTimeout(timer)
  }, [state.status, onDismiss])

  if (!isOpen) return null

  const view = viewForStage(focusedStage)
  const finalObjectCount = state.finalScene ? ((state.finalScene.objects as unknown[] | undefined)?.length ?? null) : null

  const handleRunInBackground = () => {
    setClosing(true)
    setTimeout(() => {
      onDismiss()
      setClosing(false)
    }, 300)
  }

  return (
    <div
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8",
        "bg-background/90 backdrop-blur-sm",
        closing ? "animate-out fade-out duration-300" : "animate-in fade-in duration-200"
      )}
    >
      <div
        className={cn(
          "relative flex flex-col w-full max-w-5xl h-full max-h-[720px] rounded-lg border border-primary/30 bg-card/95 overflow-hidden",
          "shadow-[0_0_40px_rgba(0,255,255,0.12)]",
          closing ? "animate-out zoom-out-95 duration-300" : "animate-in zoom-in-95 duration-300"
        )}
      >
        {/* Corner accents — matches HoloPanel's HUD-bracket language */}
        <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-primary" />
        <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-primary" />
        <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-primary" />
        <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-primary" />

        {/* Header */}
        <div className="shrink-0 flex items-center justify-between gap-3 px-4 py-3 border-b border-primary/20">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                state.status === "running" && "bg-primary animate-pulse shadow-[0_0_8px_var(--glow)]",
                state.status === "done" && "bg-primary shadow-[0_0_8px_var(--glow)]",
                state.status === "error" && "bg-destructive",
                state.status === "connecting" && "bg-muted-foreground animate-pulse"
              )}
            />
            <h2 className="text-sm font-mono text-primary tracking-wider uppercase">
              {state.status === "done"
                ? "Scene Ready"
                : state.status === "error"
                ? "Pipeline Error"
                : "Building Scene"}
            </h2>
            {state.transcript && (
              <span className="text-xs font-mono text-muted-foreground truncate max-w-md">
                "{state.transcript}"
              </span>
            )}
          </div>

          {state.status === "running" && (
            <button
              onClick={handleRunInBackground}
              className="text-[11px] font-mono text-muted-foreground hover:text-primary transition-colors underline underline-offset-2"
            >
              Run in background
            </button>
          )}
        </div>

        {state.status === "error" && (
          <div className="px-4 py-2 bg-destructive/10 border-b border-destructive/30 text-xs font-mono text-destructive">
            {state.errorMessage}
          </div>
        )}

        {/* Body: timeline rail + focused stage panel */}
        <div className="flex-1 min-h-0 grid grid-cols-[200px_1fr] gap-3 p-3">
          <div className="min-h-0 flex flex-col border-r border-primary/10 pr-2">
            <StageTimeline
              stageByName={state.stageByName}
              activeStage={activeStage}
              selectedStage={focusedStage}
              onSelectStage={setManualFocus}
            />
          </div>

          <div className="min-h-0 rounded border border-primary/15 bg-black/10 p-3">
            {view === "optimizer" && <PromptOptimizerView events={state.stageByName["prompt_optimizer"] ?? []} />}
            {view === "extraction" && <IntentExtractionView events={state.stageByName["intent_extraction"] ?? []} />}
            {view === "resolution" && <AssetResolutionView stageByName={state.stageByName} />}
            {view === "layout" && <ArchitectLayoutView events={state.stageByName["architect_layout"] ?? []} />}
            {view === "detail" && <ArchitectDetailView events={state.stageByName["architect_detail"] ?? []} />}
            {view === "finish" && <ArchitectFinishView events={state.stageByName["architect_finish"] ?? []} />}
            {view === "critic" && <CriticLoopView stageByName={state.stageByName} />}
            {view === "finalize" && (
              <ValidationRepairView stageByName={state.stageByName} finalObjectCount={finalObjectCount} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
