"use client"

import { useEffect, useRef, useState } from "react"
import { HoloPanel } from "./holo-panel"

interface LogEntry {
  timestamp: string
  level: "info" | "warn" | "error" | "system"
  message: string
}

function parseLevel(msg: string): LogEntry["level"] {
  const m = msg.toLowerCase()
  if (m.includes("error") || m.includes("failed") || m.includes("fatal")) return "error"
  if (m.includes("warn") || m.includes("warning")) return "warn"
  if (m.includes("[pipeline]") || m.includes("[voice]") || m.includes("[scene]")) return "system"
  return "info"
}

function parseLogs(raw: string[]): LogEntry[] {
  return raw.map((line) => {
    const match = line.match(/^\[(\d{2}:\d{2}:\d{2}[.\d]*)\]\s(.*)$/)
    if (match) {
      return { timestamp: match[1], level: parseLevel(match[2]), message: match[2] }
    }
    return { timestamp: "", level: parseLevel(line), message: line }
  })
}

const DEMO_LOGS: LogEntry[] = [
  { timestamp: "00:00:00.000", level: "system", message: "[Pipeline] Waiting for voice input…" },
  { timestamp: "00:00:00.001", level: "info",   message: "Semantic parser ready (215 concepts)" },
  { timestamp: "00:00:00.002", level: "info",   message: "Asset registry ready" },
]

const LEVEL_COLOR: Record<LogEntry["level"], string> = {
  error:  "text-red-400",
  warn:   "text-yellow-400",
  system: "text-cyan-300",
  info:   "text-muted-foreground",
}

interface PerformanceLogsProps {
  liveLogs?: string[]
  guiFps?: number   // kept for API compat, unused
}

export function PerformanceLogs({ liveLogs }: PerformanceLogsProps) {
  const logsRef = useRef<HTMLDivElement>(null)
  const [userScrolled, setUserScrolled] = useState(false)

  const entries: LogEntry[] =
    liveLogs && liveLogs.length > 0 ? parseLogs(liveLogs) : DEMO_LOGS

  // Auto-scroll to bottom unless user scrolled up manually
  useEffect(() => {
    const el = logsRef.current
    if (!el || userScrolled) return
    el.scrollTop = el.scrollHeight
  }, [entries, userScrolled])

  const handleScroll = () => {
    const el = logsRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    setUserScrolled(!atBottom)
  }

  return (
    <HoloPanel title="Logs" statusIndicator className="h-full">
      <div
        ref={logsRef}
        onScroll={handleScroll}
        className="h-full overflow-y-auto font-mono text-[11px] leading-5 space-y-px pr-1"
        style={{ maxHeight: "calc(100% - 0px)" }}
      >
        {entries.map((log, i) => (
          <div key={i} className="flex gap-2 group">
            {log.timestamp && (
              <span className="shrink-0 text-primary/40 select-none tabular-nums">
                [{log.timestamp}]
              </span>
            )}
            <span className={LEVEL_COLOR[log.level]}>{log.message}</span>
          </div>
        ))}
      </div>
    </HoloPanel>
  )
}
