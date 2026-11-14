"use client"

import { useCallback, useReducer, useRef } from "react"
import { connectPipelineWebSocket } from "@/lib/api"
import type { PipelineEvent, WireMessage } from "@/lib/pipelineTypes"

export type PipelineRunStatus = "idle" | "connecting" | "running" | "done" | "error"

export interface PipelineRunState {
  status: PipelineRunStatus
  runId: string | null
  transcript: string | null
  events: PipelineEvent[]
  stageByName: Record<string, PipelineEvent[]>
  finalScene: Record<string, unknown> | null
  errorMessage: string | null
}

const INITIAL_STATE: PipelineRunState = {
  status: "idle",
  runId: null,
  transcript: null,
  events: [],
  stageByName: {},
  finalScene: null,
  errorMessage: null,
}

type Action =
  | { type: "CONNECTING" }
  | { type: "RUN_STARTED"; runId: string; transcript: string }
  | { type: "EVENT"; event: PipelineEvent }
  | { type: "RUN_FINISHED"; status: "done" | "error"; scene?: Record<string, unknown>; message?: string }
  | { type: "ERROR"; message: string }
  | { type: "RESET" }

function reducer(state: PipelineRunState, action: Action): PipelineRunState {
  switch (action.type) {
    case "CONNECTING":
      return { ...INITIAL_STATE, status: "connecting" }
    case "RUN_STARTED":
      return {
        ...INITIAL_STATE,
        status: "running",
        runId: action.runId,
        transcript: action.transcript,
      }
    case "EVENT": {
      const stage = action.event.stage
      const existing = state.stageByName[stage] ?? []
      return {
        ...state,
        events: [...state.events, action.event],
        stageByName: { ...state.stageByName, [stage]: [...existing, action.event] },
      }
    }
    case "RUN_FINISHED":
      return {
        ...state,
        status: action.status,
        finalScene: action.scene ?? state.finalScene,
        errorMessage: action.status === "error" ? action.message ?? "Pipeline failed" : null,
      }
    case "ERROR":
      return { ...state, status: "error", errorMessage: action.message }
    case "RESET":
      return INITIAL_STATE
    default:
      return state
  }
}

const SAFETY_TIMEOUT_MS = 5 * 60 * 1000 // 5 min — generous given the multi-pass pipeline's latency budget

export function usePipelineStream() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE)
  const wsRef = useRef<WebSocket | null>(null)
  const safetyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearSafetyTimer = useCallback(() => {
    if (safetyTimerRef.current) {
      clearTimeout(safetyTimerRef.current)
      safetyTimerRef.current = null
    }
  }, [])

  const sendCommand = useCallback((ws: WebSocket, command: string) => {
    dispatch({ type: "CONNECTING" })
    ws.send(JSON.stringify({ command }))
    clearSafetyTimer()
    safetyTimerRef.current = setTimeout(() => {
      dispatch({ type: "ERROR", message: "Pipeline timed out — no response after 5 minutes" })
    }, SAFETY_TIMEOUT_MS)
  }, [clearSafetyTimer])

  const start = useCallback(
    (command: string) => {
      const existing = wsRef.current
      if (existing && existing.readyState === WebSocket.OPEN) {
        sendCommand(existing, command)
        return
      }

      dispatch({ type: "CONNECTING" })
      const ws = connectPipelineWebSocket()
      wsRef.current = ws

      ws.onopen = () => {
        sendCommand(ws, command)
      }

      ws.onmessage = (evt: MessageEvent<string>) => {
        let msg: WireMessage
        try {
          msg = JSON.parse(evt.data) as WireMessage
        } catch {
          return
        }
        if (msg.type === "run_started") {
          dispatch({ type: "RUN_STARTED", runId: msg.run_id, transcript: msg.transcript })
        } else if (msg.type === "pipeline_event") {
          const { type: _type, ...event } = msg
          void _type
          dispatch({ type: "EVENT", event: event as PipelineEvent })
        } else if (msg.type === "run_finished") {
          clearSafetyTimer()
          dispatch({ type: "RUN_FINISHED", status: msg.status, scene: msg.scene, message: msg.message })
        }
      }

      ws.onerror = () => {
        dispatch({ type: "ERROR", message: "WebSocket connection error" })
      }

      ws.onclose = () => {
        wsRef.current = null
      }
    },
    [sendCommand, clearSafetyTimer]
  )

  const dismiss = useCallback(() => {
    clearSafetyTimer()
    dispatch({ type: "RESET" })
  }, [clearSafetyTimer])

  return { state, start, dismiss }
}
