"use client"

import { cn } from "@/lib/utils"
import type { PipelineEvent, SceneIntentIR } from "@/lib/pipelineTypes"

export function IntentExtractionView({ events }: { events: PipelineEvent[] }) {
  const outputEvent = [...events].reverse().find((e) => e.status === "output")
  const ir = outputEvent?.payload as SceneIntentIR | undefined

  if (!ir) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Extracting scene structure…
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Scene Type</span>
        <span className="px-2 py-0.5 rounded-full border border-primary/40 bg-primary/10 text-xs font-mono text-primary">
          {ir.scene_type}
        </span>
      </div>

      {ir.objects.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Objects</span>
          <div className="flex flex-wrap gap-1.5">
            {ir.objects.map((obj, i) => (
              <div
                key={i}
                className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded-full border text-xs font-mono animate-in zoom-in-95 fade-in",
                  obj.role === "primary" ? "border-primary/50 bg-primary/10 text-primary" : "border-primary/25 border-dashed text-foreground/80"
                )}
                style={{ animationDelay: `${i * 70}ms` }}
              >
                {obj.concept}
                {obj.count > 1 && (
                  <sup className="text-[9px] text-primary/70">×{obj.count}</sup>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {ir.spatial_relationships.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Spatial Relationships
          </span>
          <ul className="flex flex-col gap-0.5">
            {ir.spatial_relationships.map((r, i) => (
              <li key={i} className="text-xs font-mono text-foreground/80">
                <span className="text-primary/80">{r.subject}</span>{" "}
                <span className="text-muted-foreground">{r.relation}</span>{" "}
                <span className="text-primary/80">{r.object}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {ir.dynamics.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Dynamics</span>
          <ul className="flex flex-col gap-0.5">
            {ir.dynamics.map((d, i) => (
              <li key={i} className="text-xs font-mono text-foreground/80">
                <span className="text-primary/80">{d.target}</span>{" "}
                <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-[10px] mx-1">{d.motion}</span>
                {d.notes && <span className="text-muted-foreground">— {d.notes}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {ir.mood_style.descriptors && ir.mood_style.descriptors.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {ir.mood_style.descriptors.map((d, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded-full bg-muted/50 text-[10px] font-mono text-muted-foreground">
              {d}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
