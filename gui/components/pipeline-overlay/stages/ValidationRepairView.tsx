"use client"

import type { PipelineEvent, RepairPayload, ValidatePayload } from "@/lib/pipelineTypes"

interface Props {
  stageByName: Record<string, PipelineEvent[]>
  finalObjectCount: number | null
}

function latestOutputPayload<T>(events: PipelineEvent[] | undefined): T | undefined {
  if (!events) return undefined
  return [...events].reverse().find((e) => e.status === "output")?.payload as T | undefined
}

export function ValidationRepairView({ stageByName, finalObjectCount }: Props) {
  const validate = latestOutputPayload<ValidatePayload>(stageByName["validate"])
  const repair = latestOutputPayload<RepairPayload>(stageByName["repair"])

  if (!validate) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Validating scene…
      </div>
    )
  }

  const checklistItems: { label: string; ok: boolean }[] = [
    { label: "All object geometries valid", ok: !validate.errors.some((e) => e.includes("geometry")) },
    { label: "Materials complete", ok: !validate.errors.some((e) => e.includes("material")) },
    { label: "Mesh paths verified", ok: !validate.errors.some((e) => e.includes("model path")) },
    { label: "No fatal errors", ok: !validate.fatal },
  ]

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      <div className="flex flex-col gap-1">
        {checklistItems.map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-2 text-xs font-mono animate-in fade-in slide-in-from-left-1"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <span className={item.ok ? "text-primary" : "text-yellow-400"}>{item.ok ? "✓" : "!"}</span>
            <span className={item.ok ? "text-foreground/80" : "text-yellow-300"}>{item.label}</span>
          </div>
        ))}
      </div>

      {repair && repair.actions.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Auto-Repaired
          </span>
          <ul className="flex flex-col gap-0.5">
            {repair.actions.slice(0, 6).map((a, i) => (
              <li key={i} className="text-[10px] font-mono text-muted-foreground/80">
                • {a}
              </li>
            ))}
          </ul>
        </div>
      )}

      {finalObjectCount != null && (
        <div className="mt-auto pt-2 border-t border-primary/15 flex items-center gap-2 text-xs font-mono">
          <span className="text-primary">✓ Scene ready</span>
          <span className="text-muted-foreground">— {finalObjectCount} object(s)</span>
        </div>
      )}
    </div>
  )
}
