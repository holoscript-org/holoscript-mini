const BASE_URL = "http://localhost:8000"

export type Frame = number[][][] | null   // [360][18][3] uint8, or null when no renderer

export interface SceneStatus {
  rotation_y: number
  scale: number
  explode: number
  frozen: boolean
  gesture: string
  transcript: string
}

async function safeFetch<T>(url: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(url, { cache: "no-store" })
    if (!res.ok) return fallback
    return (await res.json()) as T
  } catch {
    return fallback
  }
}

export async function fetchFrame(): Promise<{ frame: Frame }> {
  return safeFetch(`${BASE_URL}/frame`, { frame: null })
}

export async function fetchScene(): Promise<Record<string, unknown>> {
  return safeFetch(`${BASE_URL}/scene`, {})
}

export async function fetchLogs(): Promise<{ logs: string[] }> {
  return safeFetch(`${BASE_URL}/logs`, { logs: [] })
}

export async function fetchStatus(): Promise<SceneStatus> {
  return safeFetch(`${BASE_URL}/status`, {
    rotation_y: 0,
    scale: 1,
    explode: 0,
    frozen: false,
    gesture: "NONE",
    transcript: "",
  })
}

export async function pushScene(scene: Record<string, unknown>): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/scene`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene }),
    })
    return res.ok
  } catch {
    return false
  }
}
