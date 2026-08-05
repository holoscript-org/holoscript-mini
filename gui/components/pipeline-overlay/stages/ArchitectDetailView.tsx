"use client"

import type { ArchitectDetailPayload, PipelineEvent } from "@/lib/pipelineTypes"

export function ArchitectDetailView({ events }: { events: PipelineEvent[] }) {
  const outputEvent = [...events].reverse().find((e) => e.status === "output")
  const payload = outputEvent?.payload as ArchitectDetailPayload | undefined

  if (!payload) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Filling in materials, geometry & animation…
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-2 h-full overflow-y-auto content-start">
      {payload.objects.map((obj, i) => {
        const color = obj.material?.color ?? "#888888"
        const geomType = obj.geometry?.type ?? (obj.model ? "mesh" : "?")
        const animType = obj.animation?.type ?? "none"
        return (
          <div
            key={String(obj.id)}
            className="flex flex-col gap-1 p-2 rounded border border-primary/15 bg-black/20 animate-in fade-in slide-in-from-bottom-1"
            style={{ animationDelay: `${i * 50}ms` }}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="w-3 h-3 rounded-full border border-primary/30 shrink-0"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs font-mono text-foreground/90 truncate">{String(obj.id)}</span>
            </div>
            <div className="flex flex-wrap gap-1">
              <span className="px-1.5 py-0.5 rounded bg-muted/40 text-[9px] font-mono text-muted-foreground">
                {geomType}
              </span>
              {animType !== "none" && (
                <span className="px-1.5 py-0.5 rounded bg-primary/10 text-[9px] font-mono text-primary">
                  {animType === "physics" && obj.animation?.physics_type
                    ? `physics:${obj.animation.physics_type}`
                    : animType}
                </span>
              )}
            </div>
            {obj.description && (
              <p className="text-[10px] font-mono text-muted-foreground/80 line-clamp-2">{obj.description}</p>
            )}
          </div>
        )
      })}
    </div>
  )
}
