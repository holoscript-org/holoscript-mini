"use client"

import { useEffect, useState, useRef } from "react"
import { HoloPanel } from "./holo-panel"

interface LogEntry {
  timestamp: string
  message: string
}

const initialLogs: LogEntry[] = [
  { timestamp: "10:59:27.057", message: "System boot → GUI demo mode" },
  { timestamp: "10:59:27.058", message: "SceneState initialised" },
  { timestamp: "10:59:27.060", message: "Voice → 'Show me a spinning sphere'" },
  { timestamp: "10:59:31.066", message: "Voice → 'Add a glowing ring around it'" },
  { timestamp: "10:59:35.071", message: "Voice → 'Make the ring pulse with blue light'" },
  { timestamp: "10:59:39.075", message: "Voice → 'Explode the scene gently'" },
  { timestamp: "10:59:43.085", message: "Voice → 'Reset and show a galaxy cluster'" },
  { timestamp: "10:59:47.090", message: "Voice → 'Show me a spinning sphere'" },
  { timestamp: "10:59:51.095", message: "Voice → 'Add a glowing ring around it'" },
  { timestamp: "10:59:55.100", message: "Voice → 'Make the ring pulse with blue light'" },
  { timestamp: "10:59:59.102", message: "Voice → 'Explode the scene gently'" },
  { timestamp: "11:00:03.106", message: "Voice → 'Reset and show a galaxy cluster'" },
]

const BACKEND_STREAM = "http://localhost:8000/stream"

interface PerformanceLogsProps {
  /** Live log strings from the backend. When provided, shown instead of demo data. */
  liveLogs?: string[]
  /** Three.js render FPS passed in from the parent (via ThreeScene onFpsUpdate). */
  guiFps?: number
}

export function PerformanceLogs({ liveLogs, guiFps: guiFpsProp }: PerformanceLogsProps) {
  const [guiFps, setGuiFps]   = useState(guiFpsProp ?? 27.0)
  const [simFps, setSimFps]   = useState<number | null>(null)
  const [fpsHistory, setFpsHistory] = useState<number[]>(Array(60).fill(27))
  const [simHistory, setSimHistory] = useState<number[]>(Array(60).fill(0))
  const logsRef = useRef<HTMLDivElement>(null)

  // Sync guiFps prop into state (throttled by React batching)
  useEffect(() => {
    if (guiFpsProp !== undefined) {
      setGuiFps(guiFpsProp)
      setFpsHistory((prev) => [...prev.slice(1), guiFpsProp])
    }
  }, [guiFpsProp])

  // Derive Sim FPS from /stream MJPEG boundary timing
  useEffect(() => {
    let active = true
    let controller: AbortController | null = null

    const measure = async () => {
      try {
        controller = new AbortController()
        const res = await fetch(BACKEND_STREAM, {
          signal: controller.signal,
        })
        if (!res.body) return

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let frameCount = 0
        let windowStartMs = performance.now()
        let remainder = ""

        while (active) {
          const { done, value } = await reader.read()
          if (done) break
          const text = decoder.decode(value, { stream: true })
          const combined = remainder + text
          const parts = combined.split("--frame")
          remainder = parts.pop() ?? ""
          frameCount += parts.length

          const nowMs = performance.now()
          if (nowMs - windowStartMs >= 1000) {
            const fps = Math.round((frameCount * 1000) / (nowMs - windowStartMs))
            if (active) {
              setSimFps(fps)
              setSimHistory((prev) => [...prev.slice(1), fps])
            }
            frameCount = 0
            windowStartMs = nowMs
          }
        }
      } catch {
        if (active) setSimFps(null)
      }
    }

    measure()
    return () => {
      active = false
      controller?.abort()
    }
  }, [])

  // Fallback simulated GUI FPS when no prop is provided
  useEffect(() => {
    if (guiFpsProp !== undefined) return
    const interval = setInterval(() => {
      const fps = 25 + Math.random() * 10
      setGuiFps(fps)
      setFpsHistory((prev) => [...prev.slice(1), fps])
    }, 450)
    return () => clearInterval(interval)
  }, [guiFpsProp])

  // Convert live log strings (e.g. "[10:00:01.123] msg") into LogEntry shape,
  // falling back to static demo logs when backend is offline.
  const logs: LogEntry[] = liveLogs && liveLogs.length > 0
    ? liveLogs.map((raw) => {
        const match = raw.match(/^\[(\d{2}:\d{2}:\d{2}\.\d+)\]\s(.*)$/)
        return match
          ? { timestamp: match[1], message: match[2] }
          : { timestamp: "", message: raw }
      })
    : initialLogs

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const renderGraph = (data: number[], color: string, maxVal: number) => {
    const width = 400
    const height = 30
    const points = data
      .map((val, i) => {
        const x = (i / (data.length - 1)) * width
        const y = height - (val / maxVal) * height
        return `${x},${y}`
      })
      .join(" ")

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-8">
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          points={points}
          className="drop-shadow-[0_0_3px_rgba(0,255,255,0.5)]"
        />
      </svg>
    )
  }

  return (
    <HoloPanel title="Performance & Logs" statusIndicator className="h-full">
      <div className="space-y-4">
        {/* FPS Metrics */}
        <div className="grid grid-cols-[100px_1fr] gap-4 items-center">
          <div>
            <div className="text-xs text-muted-foreground font-mono">GUI FPS</div>
            <div className="text-lg font-mono text-primary">{guiFps.toFixed(1)} fps</div>
          </div>
          <div className="bg-background/50 rounded p-2 border border-primary/10">
            {renderGraph(fpsHistory, "#00ffff", 40)}
          </div>
        </div>

        <div className="grid grid-cols-[100px_1fr] gap-4 items-center">
          <div>
            <div className="text-xs text-muted-foreground font-mono">Sim FPS</div>
            <div className="text-lg font-mono text-primary">
              {simFps !== null ? `${simFps} fps` : "— fps"}
            </div>
          </div>
          <div className="bg-background/50 rounded p-2 border border-primary/10">
            {renderGraph(simHistory, "#00ffff", 35)}
          </div>
        </div>

        {/* Logs Console */}
        <div
          ref={logsRef}
          className="h-[120px] overflow-auto rounded bg-background/50 p-2 font-mono text-xs space-y-0.5 border border-primary/10"
        >
          {logs.map((log, i) => (
            <div key={i} className="text-muted-foreground">
              <span className="text-primary/60">[{log.timestamp}]</span> {log.message}
            </div>
          ))}
        </div>
      </div>
    </HoloPanel>
  )
}
