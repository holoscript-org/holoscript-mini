"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { HoloPanel } from "./holo-panel"
import { Monitor } from "lucide-react"

const STREAM_URL  = "http://localhost:8000/stream"
const CONTROL_URL = "http://localhost:8000/control"

// Keys that map to renderer actions (mirrors safe transform subset).
const KEY_ACTION_MAP: Record<string, string> = {
  " ":           "space",
  ArrowLeft:     "left",
  ArrowRight:    "right",
  ArrowUp:       "up",
  ArrowDown:     "down",
  e:             "e",
  E:             "e",
  r:             "r",
  R:             "r",
}

// Keys whose browser default (scroll, etc.) we want to suppress
const PREVENT_DEFAULT_KEYS = new Set([" ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"])

export function RenderWindowPanel() {
  const imgRef    = useRef<HTMLImageElement>(null)
  const panelRef  = useRef<HTMLDivElement>(null)
  const retryRef  = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [state, setState] = useState<"connecting" | "live" | "offline">("connecting")
  const [isFocused, setIsFocused] = useState(false)
  // Last key action shown briefly in the HUD
  const [lastAction, setLastAction] = useState<string | null>(null)
  const actionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── MJPEG stream ─────────────────────────────────────────────────────────
  const attachStream = () => {
    const img = imgRef.current
    if (!img) return
    img.src = `${STREAM_URL}?t=${Date.now()}`
  }

  useEffect(() => {
    attachStream()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleLoad = () => {
    setState("live")
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null }
  }

  const handleError = () => {
    setState("offline")
    retryRef.current = setTimeout(() => { setState("connecting"); attachStream() }, 3000)
  }

  // ── Keyboard control ──────────────────────────────────────────────────────
  const sendControl = useCallback((action: string) => {
    fetch(CONTROL_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ action }),
    }).catch(() => {/* renderer offline — silently ignore */})

    // Show action label in HUD for 800 ms
    setLastAction(action.toUpperCase())
    if (actionTimerRef.current) clearTimeout(actionTimerRef.current)
    actionTimerRef.current = setTimeout(() => setLastAction(null), 800)
  }, [])

  useEffect(() => {
    // Start in a known visible state.
    sendControl("k")
    sendControl("r")
  }, [sendControl])

  const handlePanelKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const action = KEY_ACTION_MAP[e.key]
    if (!action) return
    if (PREVENT_DEFAULT_KEYS.has(e.key)) e.preventDefault()
    sendControl(action)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <HoloPanel title="Renderer Window" statusIndicator className="h-full">
      {/* tabIndex makes the div focusable so keydown fires on click */}
      <div
        ref={panelRef}
        tabIndex={0}
        onClick={() => panelRef.current?.focus()}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        onKeyDown={handlePanelKeyDown}
        className="relative h-full min-h-0 w-full rounded overflow-hidden border border-primary/20 bg-black/80"
      >
        {/* MJPEG stream */}
        <img
          ref={imgRef}
          alt="Live renderer stream"
          className="h-full w-full object-contain"
          onLoad={handleLoad}
          onError={handleError}
        />

        {/* Overlay while offline / connecting */}
        {state !== "live" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground bg-black/60">
            <Monitor className="h-10 w-10 text-primary/40" />
            <p className="text-sm font-mono">
              {state === "offline" ? "Renderer offline — retrying…" : "Connecting…"}
            </p>
          </div>
        )}

        {/* Status badge */}
        <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 rounded border border-primary/30 bg-background/75 px-2 py-1 text-xs font-mono text-primary">
          {state === "live" && (
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 shadow-[0_0_6px_rgba(0,255,100,0.6)]" />
          )}
          {state === "live" ? "LIVE" : state === "offline" ? "OFFLINE" : "CONNECTING"}
        </div>

        <div className="absolute right-3 top-3 flex items-center gap-1">
          <button
            onClick={() => sendControl("k")}
            className="rounded border border-primary/30 bg-background/70 px-2 py-1 text-[10px] font-mono text-primary hover:bg-primary/20"
            title="Load Solar"
          >
            Solar
          </button>
          <button
            onClick={() => sendControl("h")}
            className="rounded border border-primary/30 bg-background/70 px-2 py-1 text-[10px] font-mono text-primary hover:bg-primary/20"
            title="Load Heart"
          >
            Heart
          </button>
          <button
            onClick={() => sendControl("j")}
            className="rounded border border-primary/30 bg-background/70 px-2 py-1 text-[10px] font-mono text-primary hover:bg-primary/20"
            title="Load Test Scene"
          >
            Test
          </button>
          <button
            onClick={() => sendControl("r")}
            className="rounded border border-primary/30 bg-background/70 px-2 py-1 text-[10px] font-mono text-primary hover:bg-primary/20"
            title="Reset View"
          >
            Reset
          </button>
        </div>

        {/* Key action HUD — flashes briefly when a key is pressed */}
        {lastAction && (
          <div className="pointer-events-none absolute right-3 top-3 rounded border border-primary/40 bg-background/80 px-2 py-1 text-xs font-mono text-primary animate-pulse">
            {lastAction}
          </div>
        )}

        {/* Key-hint strip at bottom — visible only while focused */}
        {state === "live" && (
          <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded border border-primary/20 bg-background/70 px-3 py-1 text-[10px] font-mono text-primary/50 whitespace-nowrap">
            <span>{isFocused ? "SPACE freeze" : "Click panel to enable keys"}</span>
            <span className="text-primary/25">·</span>
            <span>← → rotate</span>
            <span className="text-primary/25">·</span>
            <span>↑ ↓ scale</span>
            <span className="text-primary/25">·</span>
            <span>E explode</span>
            <span className="text-primary/25">·</span>
            <span>R reset</span>
          </div>
        )}
      </div>
    </HoloPanel>
  )
}
