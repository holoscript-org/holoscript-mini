"use client"

import { cn } from "@/lib/utils"
import type { PipelineEvent } from "@/lib/pipelineTypes"
import { STAGE_LIST } from "./stageList"

interface StageTimelineProps {
  stageByName: Record<string, PipelineEvent[]>
  activeStage: string | null
  onSelectStage: (stageId: string) => void
  selectedStage: string | null
}

type StageStatus = "pending" | "active" | "done" | "failed"

function stageStatus(events: PipelineEvent[] | undefined): StageStatus {
  if (!events || events.length === 0) return "pending"
  const last = events[events.length - 1]
  if (last.status === "failed") return "failed"
  if (last.status === "completed") return "done"
  return "active"
}

function elapsedForStage(events: PipelineEvent[] | undefined): number | null {
  if (!events) return null
  const completed = [...events].reverse().find((e) => e.elapsed_ms != null)
  return completed?.elapsed_ms ?? null
}

function StatusDot({ status }: { status: StageStatus }) {
  if (status === "pending") {
    return <span className="w-2.5 h-2.5 rounded-full border border-muted-foreground/40" />
  }
  if (status === "active") {
    return (
      <span className="relative flex w-2.5 h-2.5">
        <span className="absolute inline-flex h-full w-full rounded-full bg-primary opacity-60 animate-ping" />
        <span className="relative inline-flex rounded-full w-2.5 h-2.5 bg-primary shadow-[0_0_8px_var(--glow)]" />
      </span>
    )
  }
  if (status === "failed") {
    return <span className="w-2.5 h-2.5 rounded-full bg-destructive shadow-[0_0_6px_rgba(220,38,38,0.6)]" />
  }
  return (
    <span className="w-2.5 h-2.5 rounded-full bg-primary flex items-center justify-center">
      <svg viewBox="0 0 12 12" className="w-2 h-2 text-primary-foreground" fill="none">
        <path d="M2 6l2.5 2.5L10 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  )
}

export function StageTimeline({ stageByName, activeStage, onSelectStage, selectedStage }: StageTimelineProps) {
  // Include any stage that fired but isn't in the fixed list (e.g. legacy_fallback,
  // or critic iterations beyond what's pre-listed) as trailing extra rows.
  const knownIds = new Set(STAGE_LIST.map((s) => s.id))
  const extraIds = Object.keys(stageByName).filter((id) => !knownIds.has(id))
  const rows = [
    ...STAGE_LIST,
    ...extraIds.map((id) => ({ id, label: id.replace(/_/g, " "), group: "finalize" as const })),
  ]

  return (
    <div className="flex flex-col gap-1 overflow-y-auto">
      {rows.map((stage) => {
        const events = stageByName[stage.id]
        const status = stageStatus(events)
        const elapsed = elapsedForStage(events)
        const isSelected = selectedStage === stage.id
        const isActiveNow = activeStage === stage.id

        return (
          <button
            key={stage.id}
            onClick={() => onSelectStage(stage.id)}
            disabled={status === "pending"}
            className={cn(
              "flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors",
              "disabled:cursor-default disabled:opacity-40",
              isSelected && status !== "pending" && "bg-primary/15 border border-primary/30",
              !isSelected && status !== "pending" && "hover:bg-primary/5 border border-transparent",
              isSelected ? "" : "border border-transparent"
            )}
          >
            <StatusDot status={status} />
            <span
              className={cn(
                "text-xs font-mono flex-1 truncate",
                status === "pending" && "text-muted-foreground/50",
                status === "active" && "text-primary",
                status === "done" && "text-foreground/90",
                status === "failed" && "text-destructive"
              )}
            >
              {stage.label}
              {isActiveNow && status === "active" && (
                <span className="ml-1 inline-block animate-pulse">…</span>
              )}
            </span>
            {elapsed != null && (
              <span className="text-[10px] font-mono text-muted-foreground/70 tabular-nums">
                {elapsed < 1000 ? `${elapsed}ms` : `${(elapsed / 1000).toFixed(1)}s`}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
