"use client"

import { HoloPanel } from "./holo-panel"

interface SceneSummaryProps {
  /** Raw scene JSON as served by the backend (same shape passed to ThreeScene). */
  scene?: Record<string, unknown> | null
}

interface DescribedObject {
  id: string
  label: string
  description: string
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined
}

/** Pull out every object carrying both a label and a description. */
function describedObjects(scene?: Record<string, unknown> | null): DescribedObject[] {
  const objects = scene?.objects
  if (!Array.isArray(objects)) return []

  const out: DescribedObject[] = []
  for (const raw of objects) {
    if (!raw || typeof raw !== "object") continue
    const o = raw as Record<string, unknown>
    const label = str(o.label)
    const description = str(o.description)
    if (label && description) {
      out.push({ id: str(o.id) ?? label, label, description })
    }
  }
  return out
}

/**
 * Educational narration panel — replaces the old log stream.
 *
 * Renders the scene-level "summary" produced by the scene architect, followed by
 * the per-object "description" of every labelled object.  Both fields are optional
 * on the schema, so each section degrades independently: a cached scene generated
 * before these fields existed still renders a sensible placeholder.
 */
export function SceneSummary({ scene }: SceneSummaryProps) {
  const name = str(scene?.name)
  const summary = str(scene?.summary)
  const described = describedObjects(scene)
  const hasContent = Boolean(summary) || described.length > 0

  return (
    <HoloPanel title="Scene Summary" statusIndicator className="h-full">
      <div className="h-full overflow-y-auto pr-1 space-y-3">
        {name && (
          <h3 className="text-xs font-mono text-primary tracking-wide uppercase">
            {name}
          </h3>
        )}

        {summary && (
          <p className="text-[12px] leading-[1.6] text-foreground/90">{summary}</p>
        )}

        {described.length > 0 && (
          <div className="space-y-2 pt-1 border-t border-primary/15">
            {described.map((obj) => (
              <div key={obj.id} className="space-y-0.5">
                <span className="text-[11px] font-mono text-cyan-300">{obj.label}</span>
                <p className="text-[11px] leading-5 text-muted-foreground">
                  {obj.description}
                </p>
              </div>
            ))}
          </div>
        )}

        {!hasContent && (
          <p className="text-[11px] font-mono text-muted-foreground leading-5">
            No explanation available for this scene yet. Generate a scene from the
            command panel to see what it demonstrates.
          </p>
        )}
      </div>
    </HoloPanel>
  )
}
