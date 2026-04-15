"use client"

import { useEffect, useRef, useState } from "react"
import {
  fetchFrame,
  fetchScene,
  fetchLogs,
  fetchStatus,
  Frame,
  SceneStatus,
} from "@/lib/api"

interface SceneData {
  frame: Frame
  scene: Record<string, unknown>
  logs: string[]
  status: SceneStatus
  connected: boolean
}

const DEFAULT_STATUS: SceneStatus = {
  rotation_y: 0,
  scale: 1,
  explode: 0,
  frozen: false,
  gesture: "NONE",
  transcript: "",
}

export function useSceneData(pollMs = 100): SceneData {
  const [frame, setFrame]     = useState<Frame>(null)
  const [scene, setScene]     = useState<Record<string, unknown>>({})
  const [logs, setLogs]       = useState<string[]>([])
  const [status, setStatus]   = useState<SceneStatus>(DEFAULT_STATUS)
  const [connected, setConnected] = useState(false)

  // Track last log count to avoid redundant state updates
  const lastLogCount = useRef(0)
  const inFlight = useRef(false)

  useEffect(() => {
    let active = true

    const poll = async () => {
      if (!active) return
      if (inFlight.current) return
      inFlight.current = true

      // Fire all four requests in parallel
      const [frameRes, sceneRes, logsRes, statusRes] = await Promise.all([
        fetchFrame(),
        fetchScene(),
        fetchLogs(),
        fetchStatus(),
      ])

      inFlight.current = false
      if (!active) return

      const gotData =
        frameRes.frame !== null ||
        Object.keys(sceneRes).length > 0 ||
        logsRes.logs.length > 0

      setConnected(gotData)
      setFrame(frameRes.frame)
      setScene(sceneRes)
      setStatus(statusRes)

      // Only update logs when the list changes length (avoids churn)
      if (logsRes.logs.length !== lastLogCount.current) {
        lastLogCount.current = logsRes.logs.length
        setLogs(logsRes.logs)
      }
    }

    poll()
    const id = setInterval(poll, pollMs)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [pollMs])

  return { frame, scene, logs, status, connected }
}
