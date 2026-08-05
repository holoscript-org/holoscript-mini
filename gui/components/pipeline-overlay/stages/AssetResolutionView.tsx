"use client"

import type { PipelineEvent } from "@/lib/pipelineTypes"

interface Props {
  stageByName: Record<string, PipelineEvent[]>
}

function latestOutput(events: PipelineEvent[] | undefined) {
  if (!events) return undefined
  return [...events].reverse().find((e) => e.status === "output")?.payload as Record<string, unknown> | undefined
}

export function AssetResolutionView({ stageByName }: Props) {
  const parse = latestOutput(stageByName["semantic_parse"])
  const resolve = latestOutput(stageByName["resolve_intent"])
  const registry = latestOutput(stageByName["asset_registry"])
  const liveSearch = latestOutput(stageByName["live_search"])

  const resolvedConcepts = (resolve?.resolved as Record<string, string[]> | undefined) ?? {}
  const resolvedCount = Object.values(resolvedConcepts).reduce((sum, arr) => sum + (arr?.length ?? 0), 0)
  const unresolved = (resolve?.unresolved as string[] | undefined) ?? []
  const verifiedAssets = (registry?.verified_assets as Array<{ concept: string; label: string }> | undefined) ?? []
  const downloaded = (liveSearch?.downloaded as Array<{ concept: string; label: string }> | undefined) ?? []
  const candidates = (liveSearch?.candidates as Array<{ concept: string }> | undefined) ?? []

  if (!parse && !resolve) {
    return (
      <div className="flex items-center justify-center h-full text-xs font-mono text-muted-foreground animate-pulse">
        Resolving concepts against the asset library…
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      <div className="flex items-center gap-4">
        <div className="flex flex-col items-center">
          <span className="text-2xl font-mono font-bold text-primary">{resolvedCount}</span>
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Resolved</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-2xl font-mono font-bold text-foreground/70">{verifiedAssets.length}</span>
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Verified Meshes
          </span>
        </div>
        {unresolved.length > 0 && (
          <div className="flex flex-col items-center">
            <span className="text-2xl font-mono font-bold text-yellow-400">{unresolved.length}</span>
            <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Unresolved</span>
          </div>
        )}
      </div>

      {verifiedAssets.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Verified on Disk
          </span>
          <ul className="flex flex-col gap-0.5 max-h-24 overflow-y-auto">
            {verifiedAssets.map((a, i) => (
              <li key={i} className="flex items-center gap-1.5 text-xs font-mono text-foreground/80">
                <span className="text-primary">✓</span>
                {a.label || a.concept}
              </li>
            ))}
          </ul>
        </div>
      )}

      {candidates.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            {downloaded.length > 0 ? `Downloaded ${downloaded.length} new asset(s)` : "Searching Poly Pizza…"}
          </span>
          <ul className="flex flex-col gap-0.5 max-h-20 overflow-y-auto">
            {candidates.map((c, i) => {
              const wasDownloaded = downloaded.some((d) => d.concept === c.concept)
              return (
                <li key={i} className="flex items-center gap-1.5 text-xs font-mono text-foreground/80">
                  <span className={wasDownloaded ? "text-primary" : "text-muted-foreground animate-pulse"}>
                    {wasDownloaded ? "↓" : "…"}
                  </span>
                  {c.concept}
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
