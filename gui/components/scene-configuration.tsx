"use client"

import { HoloPanel } from "./holo-panel"

const initialConfig = {
  version: "1.0",
  title: "Demo Hologram",
  objects: [
    {
      id: "sphere_01",
      type: "sphere",
      radius: 1.0,
      position: { x: 0, y: 0, z: 0 },
      material: {
        color: "#00ffff",
        emissive: true,
        opacity: 0.8,
      },
    },
    {
      id: "ring_01",
      type: "torus",
      radius: 1.5,
      tube: 0.05,
      rotation: { x: 90, y: 0, z: 0 },
    },
  ],
  effects: {
    glow: true,
    scanlines: true,
    pulse: {
      enabled: true,
      speed: 1.0,
    },
  },
  lighting: {
    ambient: 0.3,
    point: {
      intensity: 1.0,
      color: "#00ffff",
    },
  },
}

interface SyntaxHighlightedProps {
  json: object
  depth?: number
}

function SyntaxHighlighted({ json, depth = 0 }: SyntaxHighlightedProps) {
  const indent = "  ".repeat(depth)
  const entries = Object.entries(json)

  return (
    <>
      <span className="text-muted-foreground">{"{"}</span>
      {"\n"}
      {entries.map(([key, value], index) => (
        <span key={key}>
          {indent}  <span className="text-cyan-400">&quot;{key}&quot;</span>
          <span className="text-muted-foreground">: </span>
          {typeof value === "object" && value !== null ? (
            Array.isArray(value) ? (
              <>
                <span className="text-muted-foreground">[</span>
                {"\n"}
                {value.map((item, i) => (
                  <span key={i}>
                    {indent}    {typeof item === "object" ? (
                      <SyntaxHighlighted json={item} depth={depth + 2} />
                    ) : (
                      <span className="text-emerald-400">{JSON.stringify(item)}</span>
                    )}
                    {i < value.length - 1 && <span className="text-muted-foreground">,</span>}
                    {"\n"}
                  </span>
                ))}
                {indent}  <span className="text-muted-foreground">]</span>
              </>
            ) : (
              <SyntaxHighlighted json={value} depth={depth + 1} />
            )
          ) : typeof value === "string" ? (
            <span className="text-amber-400">&quot;{value}&quot;</span>
          ) : typeof value === "number" ? (
            <span className="text-purple-400">{value}</span>
          ) : typeof value === "boolean" ? (
            <span className="text-rose-400">{String(value)}</span>
          ) : (
            <span className="text-muted-foreground">null</span>
          )}
          {index < entries.length - 1 && <span className="text-muted-foreground">,</span>}
          {"\n"}
        </span>
      ))}
      {indent}<span className="text-muted-foreground">{"}"}</span>
    </>
  )
}

interface SceneConfigurationProps {
  /** Live scene JSON from the backend. When provided, shown instead of demo data. */
  liveScene?: Record<string, unknown> | null
}

export function SceneConfiguration({ liveScene }: SceneConfigurationProps) {
  const config = (liveScene && Object.keys(liveScene).length > 0)
    ? liveScene
    : initialConfig

  return (
    <HoloPanel title="Scene Configuration" statusIndicator className="h-full">
      <div className="h-full min-h-0 overflow-auto rounded bg-background/50 p-3 font-mono text-xs leading-relaxed">
        <pre className="whitespace-pre">
          <SyntaxHighlighted json={config} />
        </pre>
      </div>
    </HoloPanel>
  )
}
