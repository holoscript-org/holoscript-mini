import { Frame } from "@/lib/api"
import { SceneDef, SceneObject } from "@/lib/sceneFactory"

type Vec3 = [number, number, number]

interface ResolvedObject {
  id: string
  position: Vec3
  size: number
  color: [number, number, number]
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.startsWith("#") ? hex.slice(1) : hex
  if (h.length !== 6) return [0, 255, 255]
  const r = Number.parseInt(h.slice(0, 2), 16)
  const g = Number.parseInt(h.slice(2, 4), 16)
  const b = Number.parseInt(h.slice(4, 6), 16)
  return [r, g, b]
}

function estimateObjectSize(obj: SceneObject): number {
  if (obj.type !== "primitive" || !obj.geometry) {
    const sy = obj.scale?.[1] ?? 1
    return Math.max(0.5, sy)
  }

  const g = obj.geometry
  switch (g.type) {
    case "sphere":
      return Math.max(0.2, g.radius ?? 1)
    case "ring":
      return Math.max(0.2, g.outerRadius ?? 1.2)
    case "box":
      return Math.max(0.2, (g.height ?? 1) * 0.5)
    case "cylinder":
      return Math.max(0.2, g.radius ?? (g.height ?? 1) * 0.25)
    case "plane":
      return Math.max(0.2, Math.max(g.width ?? 1, g.depth ?? 1) * 0.25)
    default:
      return 0.5
  }
}

function resolveObjects(scene: SceneDef, timeSec: number): ResolvedObject[] {
  const resolvedMap = new Map<string, Vec3>()
  const result: ResolvedObject[] = []

  for (const obj of scene.objects) {
    const animation = obj.animation ?? { type: "none" as const }
    let position: Vec3 = [...obj.position]

    if (animation.type === "orbit") {
      const staticCenter: Vec3 = animation.center ?? [0, 0, 0]
      const currentCenter = animation.center_ref
        ? (resolvedMap.get(animation.center_ref) ?? staticCenter)
        : staticCenter

      const dx0 = obj.position[0] - staticCenter[0]
      const dz0 = obj.position[2] - staticCenter[2]
      const radius = Math.max(0.0001, Math.hypot(dx0, dz0))
      const baseAngle = Math.atan2(dz0, dx0)
      const angle = baseAngle + (animation.phase ?? 0) + (animation.speed ?? 1.0) * timeSec

      position = [
        currentCenter[0] + radius * Math.cos(angle),
        obj.position[1],
        currentCenter[2] + radius * Math.sin(angle),
      ]
    }

    resolvedMap.set(obj.id, position)

    const baseColor = hexToRgb(obj.material.color)
    const emissiveBoost = obj.material.emissive ? Math.max(0, obj.material.emissiveIntensity ?? 0.4) : 0
    const brightness = 1 + emissiveBoost * 0.8

    result.push({
      id: obj.id,
      position,
      size: estimateObjectSize(obj),
      color: [
        clamp(Math.round(baseColor[0] * brightness), 0, 255),
        clamp(Math.round(baseColor[1] * brightness), 0, 255),
        clamp(Math.round(baseColor[2] * brightness), 0, 255),
      ],
    })
  }

  return result
}

function luminance(c: [number, number, number]): number {
  return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

function writeBrightest(frame: number[][][], a: number, l: number, rgb: [number, number, number]) {
  const current = frame[a][l] as [number, number, number]
  if (luminance(rgb) > luminance(current)) {
    frame[a][l] = rgb
  }
}

export function buildWebglPovFrame(scene: SceneDef, timeSec: number): Frame {
  const frame: number[][][] = Array.from({ length: 360 }, () =>
    Array.from({ length: 18 }, () => [0, 0, 0])
  )

  if (!scene.objects.length) return frame

  const resolved = resolveObjects(scene, timeSec)

  let yMin = Number.POSITIVE_INFINITY
  let yMax = Number.NEGATIVE_INFINITY
  for (const obj of resolved) {
    yMin = Math.min(yMin, obj.position[1] - obj.size)
    yMax = Math.max(yMax, obj.position[1] + obj.size)
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) {
    yMin = -1
    yMax = 1
  }

  for (const obj of resolved) {
    const [x, y, z] = obj.position
    const theta = Math.atan2(z, x)
    const angleFloat = ((theta + 2 * Math.PI) % (2 * Math.PI)) * (360 / (2 * Math.PI))
    const baseAngle = Math.floor(angleFloat)

    const ledFloat = ((y - yMin) / (yMax - yMin)) * 17
    const baseLed = Math.floor(clamp(ledFloat, 0, 17))

    const angleSpread = clamp(Math.round(obj.size * 1.8), 1, 8)
    const ledSpread = clamp(Math.round(obj.size * 1.2), 0, 4)

    for (let da = -angleSpread; da <= angleSpread; da++) {
      const a = (baseAngle + da + 360) % 360
      const af = 1 - Math.abs(da) / (angleSpread + 1)

      for (let dl = -ledSpread; dl <= ledSpread; dl++) {
        const l = baseLed + dl
        if (l < 0 || l > 17) continue

        const lf = ledSpread === 0 ? 1 : 1 - Math.abs(dl) / (ledSpread + 1)
        const falloff = af * lf
        const rgb: [number, number, number] = [
          clamp(Math.round(obj.color[0] * falloff), 0, 255),
          clamp(Math.round(obj.color[1] * falloff), 0, 255),
          clamp(Math.round(obj.color[2] * falloff), 0, 255),
        ]

        writeBrightest(frame, a, l, rgb)
      }
    }
  }

  return frame
}
