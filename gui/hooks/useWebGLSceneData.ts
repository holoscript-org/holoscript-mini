"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { fetchStatus, Frame, SceneStatus } from "@/lib/api"
import { buildWebglPovFrame } from "@/hooks/webglPov"
import { validateScene } from "@/lib/sceneFactory"

export interface SceneOption {
  id: string
  label: string
}

interface WebGLSceneResponse {
  scenes: SceneOption[]
  selected?: string
  scene?: Record<string, unknown>
  error?: string
}

interface WebGLSceneData {
  frame: Frame
  scene: Record<string, unknown>
  logs: string[]
  status: SceneStatus
  connected: boolean
  selectedScene: string
  sceneOptions: SceneOption[]
  setSelectedScene: (sceneId: string) => void
}

const DEFAULT_STATUS: SceneStatus = {
  rotation_y: 0,
  scale: 1,
  explode: 0,
  frozen: false,
  gesture: "NONE",
  transcript: "",
}

const PREFERRED_DEFAULT_SCENES = [
  "examples/solar_system_v2.json",
  "solar_system.json",
  "examples/mechanical_orrery.json",
]

function timestamp(): string {
  const now = new Date()
  const hh = now.getHours().toString().padStart(2, "0")
  const mm = now.getMinutes().toString().padStart(2, "0")
  const ss = now.getSeconds().toString().padStart(2, "0")
  const ms = now.getMilliseconds().toString().padStart(3, "0")
  return `${hh}:${mm}:${ss}.${ms}`
}

export function useWebGLSceneData(): WebGLSceneData {
  const [frame, setFrame] = useState<Frame>(null)
  const [scene, setScene] = useState<Record<string, unknown>>({})
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<SceneStatus>(DEFAULT_STATUS)
  const prevStatusRef = useRef<SceneStatus>(DEFAULT_STATUS)
  const [connected, setConnected] = useState(false)
  const [selectedScene, setSelectedScene] = useState("")
  const [sceneOptions, setSceneOptions] = useState<SceneOption[]>([])

  const appendLog = useCallback((message: string) => {
    setLogs((prev) => {
      const next = [...prev, `[${timestamp()}] ${message}`]
      return next.slice(-100)
    })
  }, [])

  // Poll /status at 50 ms — only update state when values change meaningfully.
  useEffect(() => {
    const poll = async () => {
      const next = await fetchStatus()
      // Merge so transcript always has a defined value even if backend omits it.
      const merged: SceneStatus = { transcript: "", ...next }
      const prev = prevStatusRef.current

      const changed =
        Math.abs(merged.rotation_y - prev.rotation_y) > 0.5 ||
        Math.abs(merged.scale - prev.scale) > 0.005 ||
        merged.frozen !== prev.frozen ||
        merged.gesture !== prev.gesture

      if (changed) {
        prevStatusRef.current = merged
        setStatus(merged)
        console.log("gesture", merged)
      }
    }

    const id = setInterval(poll, 50)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    let active = true

    const loadCatalog = async () => {
      try {
        const res = await fetch("/api/scenes", { cache: "no-store" })
        if (!res.ok) throw new Error(`Failed to list scenes (${res.status})`)
        const data = (await res.json()) as WebGLSceneResponse
        if (!active) return

        const options = data.scenes ?? []
        setSceneOptions(options)

        if (!options.length) {
          appendLog("[WebGL] No scene JSON files available")
          return
        }

        const preferred = PREFERRED_DEFAULT_SCENES.find((id) =>
          options.some((opt) => opt.id === id)
        )
        const defaultId = preferred ?? options[0].id
        setSelectedScene(defaultId)
      } catch (error) {
        const msg = error instanceof Error ? error.message : "Unknown catalog error"
        appendLog(`[WebGL] Scene catalog error: ${msg}`)
      }
    }

    loadCatalog()
    return () => {
      active = false
    }
  }, [appendLog])

  useEffect(() => {
    if (!selectedScene) return
    let active = true

    const loadScene = async () => {
      try {
        const res = await fetch(`/api/scenes?scene=${encodeURIComponent(selectedScene)}`, {
          cache: "no-store",
        })
        if (!res.ok) throw new Error(`Failed to load scene '${selectedScene}' (${res.status})`)

        const data = (await res.json()) as WebGLSceneResponse
        if (!active) return

        if (!data.scene || typeof data.scene !== "object") {
          throw new Error(`Scene payload for '${selectedScene}' is invalid`)
        }

        setScene(data.scene)
        setConnected(true)
        appendLog(`[WebGL] Loaded scene: ${selectedScene}`)
      } catch (error) {
        const msg = error instanceof Error ? error.message : "Unknown load error"
        setConnected(false)
        appendLog(`[WebGL] Scene load error: ${msg}`)
      }
    }

    loadScene()
    return () => {
      active = false
    }
  }, [selectedScene, appendLog])

  const validated = useMemo(() => validateScene(scene), [scene])
  const errorsFingerprint = useMemo(() => validated.errors.join("\n"), [validated.errors])

  useEffect(() => {
    if (validated.fatal) {
      appendLog(`[WebGL] Scene rejected: ${validated.fatal}`)
      setFrame(null)
      return
    }

    if (validated.errors.length) {
      appendLog(`[WebGL] Validation warnings: ${validated.errors.length}`)
    }
  }, [validated.fatal, errorsFingerprint, appendLog, validated.errors.length])

  const sceneRef = useRef(validated.scene)
  useEffect(() => {
    sceneRef.current = validated.scene
  }, [validated.scene])

  useEffect(() => {
    if (validated.fatal) return

    let raf = 0
    let lastTickMs = 0
    const startMs = performance.now()

    const loop = (nowMs: number) => {
      if (!lastTickMs || nowMs - lastTickMs >= 66) {
        const t = (nowMs - startMs) / 1000
        setFrame(buildWebglPovFrame(sceneRef.current, t))
        lastTickMs = nowMs
      }
      raf = requestAnimationFrame(loop)
    }

    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
    }
  }, [validated.fatal, selectedScene])

  return {
    frame,
    scene,
    logs,
    status,
    connected,
    selectedScene,
    sceneOptions,
    setSelectedScene,
  }
}
