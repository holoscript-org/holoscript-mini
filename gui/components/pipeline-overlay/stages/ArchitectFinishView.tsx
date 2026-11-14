"use client"

import type { ArchitectFinishPayload, PipelineEvent } from "@/lib/pipelineTypes"

const LIGHT_ICON: Record<string, string> = {
  ambient: "○",
  directional: "☀",
  point: "✺",
  spot: "◎",
}

export function ArchitectFinishView({ events }: { events: PipelineEvent[] }) {
  const outputEvent = [...events].reverse().find((e) => e.status === "output")
  const payload = outputEvent?.payload as ArchitectFinishPayload | undefined

  if (!payload) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Setting lighting, camera & framing…
      </div>
    )
  }

  const maxIntensity = Math.max(1, ...payload.lights.map((l) => l.intensity))

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-300">
        <h3 className="text-sm font-mono font-bold text-primary">{payload.name}</h3>
        {payload.summary && (
          <p className="text-xs font-mono text-muted-foreground leading-relaxed mt-1">{payload.summary}</p>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Lighting Rig</span>
        {payload.lights.map((light, i) => (
          <div
            key={i}
            className="flex items-center gap-2 animate-in fade-in slide-in-from-left-1"
            style={{ animationDelay: `${i * 70}ms` }}
          >
            <span className="text-sm" style={{ color: light.color || "var(--primary)" }}>
              {LIGHT_ICON[light.type] ?? "•"}
            </span>
            <span className="text-[10px] font-mono text-muted-foreground w-16">{light.type}</span>
            <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${Math.min(100, (light.intensity / maxIntensity) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Camera</span>
        <div className="text-xs font-mono text-foreground/80">
          fov {payload.camera.fov ?? 60}° · pos [{payload.camera.position.map((n) => n.toFixed(1)).join(", ")}]
        </div>
      </div>
    </div>
  )
}
