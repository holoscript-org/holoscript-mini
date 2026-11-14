"use client"

import { useMemo } from "react"
import type { ArchitectLayoutObject, ArchitectLayoutPayload, PipelineEvent } from "@/lib/pipelineTypes"

const VIEW_SIZE = 260
const PADDING = 24

function project(objects: ArchitectLayoutObject[]) {
  const xs = objects.map((o) => o.position?.[0] ?? 0)
  const zs = objects.map((o) => o.position?.[2] ?? 0)
  const maxAbs = Math.max(1, ...xs.map(Math.abs), ...zs.map(Math.abs))
  const scale = (VIEW_SIZE / 2 - PADDING) / maxAbs
  return objects.map((o) => ({
    ...o,
    px: VIEW_SIZE / 2 + (o.position?.[0] ?? 0) * scale,
    py: VIEW_SIZE / 2 + (o.position?.[2] ?? 0) * scale,
  }))
}

export function ArchitectLayoutView({ events }: { events: PipelineEvent[] }) {
  const outputEvent = [...events].reverse().find((e) => e.status === "output")
  const payload = outputEvent?.payload as ArchitectLayoutPayload | undefined

  const projected = useMemo(() => (payload ? project(payload.objects) : []), [payload])
  const byId = useMemo(() => new Map(projected.map((o) => [o.id, o])), [projected])

  if (!payload) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Composing scene layout…
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-2 h-full">
      <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground self-start">
        Top-down placement · {payload.objects.length} object(s)
      </span>
      <svg
        viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
        className="w-full max-w-[260px] aspect-square animate-in fade-in zoom-in-95 duration-500"
      >
        <line x1={VIEW_SIZE / 2} y1={0} x2={VIEW_SIZE / 2} y2={VIEW_SIZE} stroke="var(--border)" strokeWidth={0.5} />
        <line x1={0} y1={VIEW_SIZE / 2} x2={VIEW_SIZE} y2={VIEW_SIZE / 2} stroke="var(--border)" strokeWidth={0.5} />

        {/* parent-child connector lines */}
        {projected.map((o) => {
          if (!o.parent) return null
          const parent = byId.get(o.parent)
          if (!parent) return null
          return (
            <line
              key={`edge-${o.id}`}
              x1={parent.px}
              y1={parent.py}
              x2={o.px}
              y2={o.py}
              stroke="var(--primary)"
              strokeOpacity={0.3}
              strokeWidth={1}
            />
          )
        })}

        {projected.map((o, i) => {
          const radius = o.type === "mesh" ? 5 : o.geometry?.type === "sphere" ? 4 : 3
          return (
            <g key={o.id} className="animate-in fade-in" style={{ animationDelay: `${i * 40}ms` }}>
              <circle
                cx={o.px}
                cy={o.py}
                r={radius}
                fill="var(--primary)"
                fillOpacity={o.type === "mesh" ? 0.9 : 0.6}
                stroke="var(--primary)"
                strokeWidth={0.5}
              />
              {o.label && (
                <text
                  x={o.px + radius + 3}
                  y={o.py + 3}
                  fontSize={7}
                  fontFamily="var(--font-mono)"
                  fill="var(--muted-foreground)"
                >
                  {o.label}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
