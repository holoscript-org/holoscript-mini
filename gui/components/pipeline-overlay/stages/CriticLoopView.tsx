"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import type { CriticIterationPayload, PipelineEvent } from "@/lib/pipelineTypes"

const CATEGORY_COLOR: Record<string, string> = {
  INTENT_MISMATCH: "text-yellow-400 border-yellow-500/40",
  SPATIAL: "text-orange-400 border-orange-500/40",
  SCALE: "text-orange-400 border-orange-500/40",
  LIGHTING: "text-blue-400 border-blue-500/40",
  ANIMATION: "text-purple-400 border-purple-500/40",
  PHYSICS: "text-purple-400 border-purple-500/40",
  CAMERA: "text-cyan-400 border-cyan-500/40",
}

interface Props {
  stageByName: Record<string, PipelineEvent[]>
}

export function CriticLoopView({ stageByName }: Props) {
  const iterationIds = Object.keys(stageByName)
    .filter((id) => id.startsWith("critic_iteration_"))
    .sort((a, b) => Number(a.split("_").pop()) - Number(b.split("_").pop()))

  const [expanded, setExpanded] = useState<number | null>(null)

  if (iterationIds.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Reviewing scene quality…
      </div>
    )
  }

  const activeExpanded = expanded ?? iterationIds.length - 1

  return (
    <div className="flex flex-col gap-2 h-full overflow-y-auto">
      {iterationIds.map((stageId, idx) => {
        const events = stageByName[stageId]
        const outputEvents = events
          .filter((e) => e.status === "output")
          .map((e) => e.payload as unknown as CriticIterationPayload | null)
          .filter((p): p is CriticIterationPayload => p != null)
        const isDone = events.some((e) => e.status === "completed")
        const isOpen = idx === activeExpanded

        const verdictPayload = outputEvents.find((p) => "verdict" in p)
        const verdict = verdictPayload?.verdict
        const issues = verdictPayload?.issues ?? []
        const fixed = outputEvents.find((p) => p.fixed !== undefined)

        if (!isOpen) {
          return (
            <button
              key={stageId}
              onClick={() => setExpanded(idx)}
              className="flex items-center justify-between px-2 py-1.5 rounded border border-primary/15 bg-black/20 text-left hover:bg-primary/5 transition-colors"
            >
              <span className="text-xs font-mono text-foreground/80">Iteration {idx + 1}</span>
              <span className="text-[10px] font-mono text-muted-foreground">
                {verdict === "OK" ? "Passed" : issues.length > 0 ? `${issues.length} issue(s) fixed` : "…"}
              </span>
            </button>
          )
        }

        return (
          <div key={stageId} className="flex flex-col gap-2 p-2 rounded border border-primary/25 bg-primary/5 animate-in fade-in">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-primary">Iteration {idx + 1}</span>
              {!isDone && <span className="text-[10px] font-mono text-muted-foreground animate-pulse">reviewing…</span>}
            </div>

            {issues.length > 0 && (
              <div className="flex flex-col gap-1.5">
                {issues.map((issue, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex flex-col gap-0.5 p-1.5 rounded border bg-black/20 animate-in fade-in slide-in-from-left-1",
                      CATEGORY_COLOR[issue.category] ?? "text-foreground/80 border-primary/20"
                    )}
                    style={{ animationDelay: `${i * 60}ms` }}
                  >
                    <span className="text-[9px] font-mono uppercase tracking-wider">{issue.category}</span>
                    <span className="text-[10px] font-mono text-foreground/70">{issue.description}</span>
                    <span className="text-[10px] font-mono text-muted-foreground italic">→ {issue.fix}</span>
                  </div>
                ))}
              </div>
            )}

            {verdict === "OK" && (
              <div className="flex items-center gap-1.5 text-xs font-mono text-primary">
                <span>✓</span> Scene passed review
              </div>
            )}

            {fixed?.fixed && (
              <div className="text-[10px] font-mono text-muted-foreground">
                {fixed.object_count_before} → {fixed.object_count_after} object(s) after fix
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
