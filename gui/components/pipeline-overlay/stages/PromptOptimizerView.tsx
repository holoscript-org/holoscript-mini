"use client"

import type { PipelineEvent } from "@/lib/pipelineTypes"
import type { PromptOptimizerPayload } from "@/lib/pipelineTypes"

export function PromptOptimizerView({ events }: { events: PipelineEvent[] }) {
  const outputEvent = [...events].reverse().find((e) => e.status === "output")
  const payload = outputEvent?.payload as PromptOptimizerPayload | undefined

  if (!payload) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Understanding your request…
      </div>
    )
  }

  const unchanged = payload.original_prompt.trim() === payload.optimized_prompt.trim()

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Original</span>
          <div className="rounded border border-primary/15 bg-black/30 p-2 text-xs font-mono text-muted-foreground leading-relaxed">
            {payload.original_prompt}
          </div>
        </div>
        <div className="flex flex-col gap-1 animate-in slide-in-from-left-2 duration-300">
          <span className="text-[10px] font-mono uppercase tracking-wider text-primary">
            {unchanged ? "Unchanged" : "Clarified"}
          </span>
          <div className="rounded border border-primary/30 bg-primary/5 p-2 text-xs font-mono text-foreground leading-relaxed">
            {payload.optimized_prompt}
          </div>
        </div>
      </div>

      {payload.clarifications_made.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Clarifications
          </span>
          <ul className="flex flex-col gap-1">
            {payload.clarifications_made.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-1.5 text-xs font-mono text-primary/80 animate-in fade-in slide-in-from-bottom-1"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <span className="text-primary/50">›</span>
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
