/**
 * sceneFactory.ts — Strict schema definition + validation.
 *
 * This is the SINGLE SOURCE OF TRUTH for the HoloScript scene format.
 * There is NO normalization, NO fallback guessing, NO legacy compatibility.
 *
 * If a scene JSON does not conform to this schema it will be rejected.
 * Invalid individual objects are skipped with logged errors; valid ones still render.
 */

// ─── Geometry ─────────────────────────────────────────────────────────────────

export type PrimitiveGeomType =
  | "sphere"
  | "box"
  | "cylinder"
  | "plane"
  | "ring"
  | "capsule"
  | "torus"

export interface GeometryDef {
  type: PrimitiveGeomType
  /** sphere / capsule */
  radius?: number
  /** capsule: length of the cylindrical section */
  length?: number
  /** torus: tube (pipe) radius */
  tube?: number
  /** ring */
  innerRadius?: number
  outerRadius?: number
  thetaSegments?: number
  /** box / plane */
  width?: number
  height?: number
  depth?: number
  /** cylinder: if present, overrides position/rotation and auto-orients the mesh */
  from?: [number, number, number]
  to?: [number, number, number]
}

// ─── Material ─────────────────────────────────────────────────────────────────

export interface MaterialDef {
  type: "standard"
  color: string             // "#rrggbb" — required
  roughness: number         // 0.0 – 1.0
  metalness: number         // 0.0 – 1.0
  opacity?: number          // 0.0 – 1.0, default 1.0
  transparent?: boolean     // explicit transparency flag
  /** Texture paths — loaded at runtime */
  map?: string
  normalMap?: string
  roughnessMap?: string
  metalnessMap?: string
  emissive?: string         // "#rrggbb"
  emissiveMap?: string
  emissiveIntensity?: number
}

// ─── Animation ────────────────────────────────────────────────────────────────

export type AnimationType = "none" | "orbit" | "spin" | "physics"

export type PhysicsType = "gravity" | "shm" | "pendulum" | "projectile"

/** Physics animation — generalized simulation driven by the LLM per scene context */
export interface PhysicsAnimationDef {
  type: "physics"
  physics_type: PhysicsType
  /** gravity / projectile: gravitational acceleration (9.8=earth, 1.6=moon, 3.7=mars, 24.8=jupiter, 0.6=asteroid) */
  g?: number
  /** gravity / projectile: y-level of the floor; auto-repaired if >= object start y */
  floor_y?: number
  /** gravity / projectile: energy retained per bounce, 0–1 */
  restitution?: number
  /** shm: oscillation axis, defaults to "y" */
  axis?: "x" | "y" | "z"
  /** shm / pendulum: max displacement (shm) or max angle in radians (pendulum) */
  amplitude?: number
  /** shm / pendulum: cycles per second */
  frequency?: number
  /** all types: decay coefficient — 0=perpetual, 0.05=slow decay, 0.5=fast decay */
  damping?: number
  /** pendulum: world-space pivot point; MUST be above the bob's starting y position */
  pivot?: [number, number, number]
  /** pendulum: arm length hint; overridden at render time by actual pivot–bob distance */
  arm_length?: number
  /** projectile: initial velocity vector [vx, vy, vz] */
  initial_velocity?: [number, number, number]
}

/**
 * Animation is a strict discriminated union — only fields valid for the
 * active `type` are present.  Renderers narrow by `type` before accessing
 * type-specific fields.
 */
export type AnimationDef =
  | { type: "none" }
  | {
      type: "orbit"
      /** world-space center (ignored when center_ref is set) */
      center?: [number, number, number]
      /** follow another object's live world position; only for non-parented objects */
      center_ref?: string
      /** rotation axis, default [0,1,0] */
      axis?: [number, number, number]
      /** radians/sec */
      speed?: number
      /** starting angle offset in radians */
      phase?: number
    }
  | {
      type: "spin"
      /** rotation axis, default [0,1,0] */
      axis?: [number, number, number]
      /** radians/sec */
      speed?: number
    }
  | PhysicsAnimationDef

// ─── Object ───────────────────────────────────────────────────────────────────

export type ObjectType = "primitive" | "mesh"

export interface SceneObject {
  id: string
  type: ObjectType
  /**
   * Parent object id.  When set, `position` / `rotation` / `scale` are
   * relative to the parent's local coordinate space.  Orbit animations
   * therefore orbit the parent's origin by default.
   */
  parent?: string
  /** Required when type === "primitive" */
  geometry?: GeometryDef
  /** Required when type === "mesh" — path to .glb/.gltf/.obj */
  model?: string
  position: [number, number, number]
  rotation?: [number, number, number]   // Euler degrees [x, y, z]
  scale?: [number, number, number]
  material: MaterialDef
  label?: string
  animation?: AnimationDef
}

// ─── Lights ───────────────────────────────────────────────────────────────────

export type LightType = "ambient" | "directional" | "point" | "spot"

export interface LightDef {
  type: LightType
  color?: string      // "#rrggbb", default "#ffffff"
  intensity: number
  position?: [number, number, number]  // not used for ambient
  castShadow?: boolean
}

// ─── Camera ───────────────────────────────────────────────────────────────────

export interface CameraDef {
  position: [number, number, number]
  target: [number, number, number]
  fov?: number
}

// ─── Scene ────────────────────────────────────────────────────────────────────

export interface SceneDef {
  name?: string
  objects: SceneObject[]
  lights: LightDef[]
  camera: CameraDef
}

// ─── Validation result ────────────────────────────────────────────────────────

export interface ValidationResult {
  scene: SceneDef
  errors: string[]    // non-fatal: individual objects that failed
  fatal?: string      // fatal: entire scene rejected
}

// ─── Constants ────────────────────────────────────────────────────────────────

export const VALID_OBJECT_TYPES:  readonly ObjectType[]        = ["primitive", "mesh"]
export const VALID_GEOM_TYPES:    readonly PrimitiveGeomType[] = ["sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"]
export const VALID_ANIM_TYPES:    readonly AnimationType[]     = ["none", "orbit", "spin", "physics"]
export const VALID_PHYSICS_TYPES: readonly PhysicsType[]      = ["gravity", "shm", "pendulum", "projectile"]
export const VALID_LIGHT_TYPES:   readonly LightType[]         = ["ambient", "directional", "point", "spot"]

export const DEFAULT_CAMERA: CameraDef = {
  position: [0, 5, 20],
  target:   [0, 0, 0],
  fov:      65,
}

export const DEFAULT_LIGHTS: LightDef[] = [
  { type: "ambient",     intensity: 0.4 },
  { type: "directional", intensity: 1.2, position: [10, 10, 10], castShadow: true },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function isHex(v: unknown): v is string {
  return typeof v === "string" && /^#[0-9a-fA-F]{6}$/.test(v)
}

function isVec3(v: unknown): v is [number, number, number] {
  return Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && isFinite(n))
}

function isNum(v: unknown, min = -Infinity, max = Infinity): v is number {
  return typeof v === "number" && isFinite(v) && v >= min && v <= max
}

// ─── Material validation ──────────────────────────────────────────────────────

function validateMaterial(raw: unknown, prefix: string): { mat: MaterialDef | null; errors: string[] } {
  const errors: string[] = []
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { mat: null, errors: [`${prefix}: material is required`] }
  }
  const m = raw as Record<string, unknown>

  if (m.type !== "standard")          errors.push(`${prefix}.material.type must be "standard", got "${m.type}"`)
  if (!isHex(m.color))                errors.push(`${prefix}.material.color must be "#rrggbb" hex, got "${m.color}"`)
  if (!isNum(m.roughness, 0, 1))      errors.push(`${prefix}.material.roughness must be 0–1, got ${m.roughness}`)
  if (!isNum(m.metalness, 0, 1))      errors.push(`${prefix}.material.metalness must be 0–1, got ${m.metalness}`)
  if (m.opacity     !== undefined && !isNum(m.opacity, 0, 1))     errors.push(`${prefix}.material.opacity must be 0–1`)
  if (m.transparent !== undefined && typeof m.transparent !== "boolean") errors.push(`${prefix}.material.transparent must be a boolean`)
  if (m.emissive    !== undefined && !isHex(m.emissive))          errors.push(`${prefix}.material.emissive must be hex`)
  if (m.emissiveIntensity !== undefined && !isNum(m.emissiveIntensity, 0)) errors.push(`${prefix}.material.emissiveIntensity must be >= 0`)

  // Texture path fields — must be strings if provided
  for (const field of ["map", "normalMap", "roughnessMap", "metalnessMap", "emissiveMap"] as const) {
    if (m[field] !== undefined && typeof m[field] !== "string") {
      errors.push(`${prefix}.material.${field} must be a string path`)
    }
  }

  if (errors.length) return { mat: null, errors }
  return {
    mat: {
      type:              "standard",
      color:             m.color as string,
      roughness:         m.roughness as number,
      metalness:         m.metalness as number,
      opacity:           m.opacity           as number  | undefined,
      transparent:       m.transparent       as boolean | undefined,
      map:               m.map               as string  | undefined,
      normalMap:         m.normalMap         as string  | undefined,
      roughnessMap:      m.roughnessMap      as string  | undefined,
      metalnessMap:      m.metalnessMap      as string  | undefined,
      emissive:          m.emissive          as string  | undefined,
      emissiveMap:       m.emissiveMap       as string  | undefined,
      emissiveIntensity: m.emissiveIntensity as number  | undefined,
    },
    errors: [],
  }
}

// ─── Geometry validation ──────────────────────────────────────────────────────

function validateGeometry(raw: unknown, prefix: string): { geom: GeometryDef | null; errors: string[] } {
  const errors: string[] = []
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { geom: null, errors: [`${prefix}: geometry is required for type "primitive"`] }
  }
  const g = raw as Record<string, unknown>

  if (!VALID_GEOM_TYPES.includes(g.type as PrimitiveGeomType)) {
    return { geom: null, errors: [`${prefix}.geometry.type must be one of ${VALID_GEOM_TYPES.join("|")}, got "${g.type}"`] }
  }

  const t = g.type as PrimitiveGeomType

  if ((t === "sphere" || t === "capsule") && g.radius !== undefined && !isNum(g.radius, 0)) {
    errors.push(`${prefix}.geometry.radius must be > 0`)
  }
  if (t === "capsule" && g.length !== undefined && !isNum(g.length, 0)) {
    errors.push(`${prefix}.geometry.length must be > 0`)
  }
  if (t === "torus") {
    if (g.radius !== undefined && !isNum(g.radius, 0)) errors.push(`${prefix}.geometry.radius must be > 0`)
    if (g.tube   !== undefined && !isNum(g.tube, 0))   errors.push(`${prefix}.geometry.tube must be > 0`)
  }
  if (t === "ring") {
    if (g.innerRadius !== undefined && !isNum(g.innerRadius, 0)) errors.push(`${prefix}.geometry.innerRadius must be > 0`)
    if (g.outerRadius !== undefined && !isNum(g.outerRadius, 0)) errors.push(`${prefix}.geometry.outerRadius must be > 0`)
    if (isNum(g.innerRadius, 0) && isNum(g.outerRadius, 0) && (g.outerRadius as number) <= (g.innerRadius as number)) {
      errors.push(`${prefix}.geometry.outerRadius must be greater than innerRadius`)
    }
    if (g.thetaSegments !== undefined && !isNum(g.thetaSegments, 3)) errors.push(`${prefix}.geometry.thetaSegments must be >= 3`)
  }
  if ((t === "box" || t === "plane") && g.width !== undefined && !isNum(g.width, 0)) {
    errors.push(`${prefix}.geometry.width must be > 0`)
  }
  if (g.from !== undefined && !isVec3(g.from)) errors.push(`${prefix}.geometry.from must be [x,y,z]`)
  if (g.to   !== undefined && !isVec3(g.to))   errors.push(`${prefix}.geometry.to must be [x,y,z]`)

  if (errors.length) return { geom: null, errors }

  return {
    geom: {
      type:          t,
      radius:        g.radius        as number | undefined,
      length:        g.length        as number | undefined,
      tube:          g.tube          as number | undefined,
      innerRadius:   g.innerRadius   as number | undefined,
      outerRadius:   g.outerRadius   as number | undefined,
      thetaSegments: g.thetaSegments as number | undefined,
      width:         g.width         as number | undefined,
      height:        g.height        as number | undefined,
      depth:         g.depth         as number | undefined,
      from:          g.from          as [number,number,number] | undefined,
      to:            g.to            as [number,number,number] | undefined,
    },
    errors: [],
  }
}

// ─── Animation validation ─────────────────────────────────────────────────────

function validateAnimation(raw: unknown, prefix: string): { anim: AnimationDef | null; errors: string[] } {
  if (raw === undefined || raw === null) return { anim: { type: "none" }, errors: [] }
  if (typeof raw !== "object" || Array.isArray(raw)) {
    return { anim: null, errors: [`${prefix}.animation must be an object`] }
  }
  const a = raw as Record<string, unknown>

  if (!VALID_ANIM_TYPES.includes(a.type as AnimationType)) {
    return { anim: null, errors: [`${prefix}.animation.type must be one of ${VALID_ANIM_TYPES.join("|")}, got "${a.type}"`] }
  }

  // ── none ──────────────────────────────────────────────────────────────────
  if (a.type === "none") return { anim: { type: "none" }, errors: [] }

  // ── orbit ─────────────────────────────────────────────────────────────────
  if (a.type === "orbit") {
    const errors: string[] = []
    if (a.center     !== undefined && !isVec3(a.center))             errors.push(`${prefix}.animation.center must be [x,y,z]`)
    if (a.axis       !== undefined && !isVec3(a.axis))               errors.push(`${prefix}.animation.axis must be [x,y,z]`)
    if (a.center_ref !== undefined && typeof a.center_ref !== "string") errors.push(`${prefix}.animation.center_ref must be a string`)
    if (a.speed      !== undefined && !isNum(a.speed))               errors.push(`${prefix}.animation.speed must be a finite number`)
    if (a.phase      !== undefined && !isNum(a.phase))               errors.push(`${prefix}.animation.phase must be a finite number`)
    if (errors.length) return { anim: null, errors }
    return {
      anim: {
        type:       "orbit",
        center:     isVec3(a.center)                   ? a.center as [number,number,number] : undefined,
        center_ref: typeof a.center_ref === "string"   ? a.center_ref                      : undefined,
        axis:       isVec3(a.axis)                     ? a.axis   as [number,number,number] : undefined,
        speed:      isNum(a.speed)                     ? a.speed  as number                : undefined,
        phase:      isNum(a.phase)                     ? a.phase  as number                : undefined,
      },
      errors: [],
    }
  }

  // ── spin ──────────────────────────────────────────────────────────────────
  if (a.type === "spin") {
    const errors: string[] = []
    if (a.axis  !== undefined && !isVec3(a.axis)) errors.push(`${prefix}.animation.axis must be [x,y,z]`)
    if (a.speed !== undefined && !isNum(a.speed)) errors.push(`${prefix}.animation.speed must be a finite number`)
    if (errors.length) return { anim: null, errors }
    return {
      anim: {
        type:  "spin",
        axis:  isVec3(a.axis) ? a.axis as [number,number,number] : undefined,
        speed: isNum(a.speed) ? a.speed as number                : undefined,
      },
      errors: [],
    }
  }

  // ── physics ───────────────────────────────────────────────────────────────
  if (a.type === "physics") {
    if (!VALID_PHYSICS_TYPES.includes(a.physics_type as PhysicsType)) {
      return {
        anim: null,
        errors: [`${prefix}.animation.physics_type must be one of ${VALID_PHYSICS_TYPES.join("|")}, got "${a.physics_type}"`],
      }
    }

    // Clamp: silently bring values into valid range rather than rejecting
    const clamp = (v: unknown, lo: number, hi: number): number | undefined => {
      if (typeof v !== "number" || !isFinite(v)) return undefined
      return Math.max(lo, Math.min(hi, v))
    }

    const phys: PhysicsAnimationDef = {
      type:         "physics",
      physics_type: a.physics_type as PhysicsType,
    }

    const g           = clamp(a.g,           0.1, 30.0);  if (g           !== undefined) phys.g           = g
    const restitution = clamp(a.restitution,  0,   1.0);  if (restitution !== undefined) phys.restitution = restitution
    const amplitude   = clamp(a.amplitude,    0,  20.0);  if (amplitude   !== undefined) phys.amplitude   = amplitude
    const frequency   = clamp(a.frequency,  0.01, 10.0);  if (frequency   !== undefined) phys.frequency   = frequency
    const damping     = clamp(a.damping,      0,   2.0);  if (damping     !== undefined) phys.damping     = damping
    const arm_length  = clamp(a.arm_length,  0.1, 20.0);  if (arm_length  !== undefined) phys.arm_length  = arm_length

    // floor_y: pass through as-is (floor_y auto-repair is done at Python validation time)
    if (isNum(a.floor_y)) phys.floor_y = a.floor_y as number

    if (isVec3(a.pivot))            phys.pivot            = a.pivot            as [number,number,number]
    if (isVec3(a.initial_velocity)) phys.initial_velocity = a.initial_velocity as [number,number,number]

    // SHM oscillation axis — string "x"|"y"|"z", distinct from orbit/spin axis (vec3)
    if (typeof a.axis === "string" && ["x","y","z"].includes(a.axis)) {
      phys.axis = a.axis as "x" | "y" | "z"
    }

    return { anim: phys, errors: [] }
  }

  return { anim: { type: "none" }, errors: [] }
}

// ─── Object validation ────────────────────────────────────────────────────────

function validateObject(raw: unknown, index: number): { obj: SceneObject | null; errors: string[] } {
  const errors: string[] = []

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { obj: null, errors: [`objects[${index}]: must be an object`] }
  }
  const r = raw as Record<string, unknown>
  const prefix = `objects[${index}](id="${r.id}")`

  // id
  if (typeof r.id !== "string" || r.id.trim() === "") {
    errors.push(`${prefix}: id must be a non-empty string`)
  }

  // type
  if (!VALID_OBJECT_TYPES.includes(r.type as ObjectType)) {
    return { obj: null, errors: [`${prefix}: type must be "primitive" or "mesh", got "${r.type}"`] }
  }
  const type = r.type as ObjectType

  // parent (optional — cross-reference check done at scene level)
  if (r.parent !== undefined && (typeof r.parent !== "string" || r.parent.trim() === "")) {
    errors.push(`${prefix}: parent must be a non-empty string id`)
  }

  // position (required)
  if (!isVec3(r.position)) {
    errors.push(`${prefix}: position must be [x,y,z] of finite numbers`)
  }

  // type-specific required fields
  let geom: GeometryDef | null = null
  if (type === "primitive") {
    const { geom: g, errors: ge } = validateGeometry(r.geometry, prefix)
    geom = g
    errors.push(...ge)
  } else {
    if (typeof r.model !== "string" || r.model.trim() === "") {
      errors.push(`${prefix}: model (path to .glb/.gltf) is required for type "mesh"`)
    }
  }

  // material (required)
  const { mat, errors: me } = validateMaterial(r.material, prefix)
  errors.push(...me)

  // optional transform fields
  if (r.rotation !== undefined && !isVec3(r.rotation)) errors.push(`${prefix}: rotation must be [rx,ry,rz]`)
  if (r.scale    !== undefined && !isVec3(r.scale))    errors.push(`${prefix}: scale must be [sx,sy,sz]`)
  if (r.label    !== undefined && typeof r.label !== "string") errors.push(`${prefix}: label must be a string`)

  // animation
  const { anim, errors: ae } = validateAnimation(r.animation, prefix)
  errors.push(...ae)

  // fail object if any required field is invalid
  if (!isVec3(r.position) || !mat || (type === "primitive" && !geom) || (type === "mesh" && !r.model)) {
    return { obj: null, errors }
  }

  return {
    obj: {
      id:        r.id as string,
      type,
      parent:    (typeof r.parent === "string" && r.parent.trim()) ? r.parent : undefined,
      geometry:  geom ?? undefined,
      model:     type === "mesh" ? (r.model as string) : undefined,
      position:  r.position as [number,number,number],
      rotation:  isVec3(r.rotation) ? r.rotation : undefined,
      scale:     isVec3(r.scale)    ? r.scale    : undefined,
      material:  mat,
      label:     typeof r.label === "string" ? r.label : undefined,
      animation: anim ?? { type: "none" },
    },
    errors,
  }
}

// ─── Parent / hierarchy validation ───────────────────────────────────────────

function validateParentRefs(objects: SceneObject[]): string[] {
  const errors: string[] = []
  const idSet = new Set(objects.map((o) => o.id))

  for (const obj of objects) {
    if (obj.parent === undefined) continue

    if (!idSet.has(obj.parent)) {
      errors.push(`objects(id="${obj.id}"): parent "${obj.parent}" references an unknown id`)
      continue
    }
    if (obj.parent === obj.id) {
      errors.push(`objects(id="${obj.id}"): parent cannot reference self`)
      continue
    }

    // Cycle detection: walk the parent chain
    const visited = new Set<string>()
    let cur: string | undefined = obj.id
    while (cur !== undefined) {
      if (visited.has(cur)) {
        errors.push(`objects(id="${obj.id}"): circular parent dependency detected`)
        break
      }
      visited.add(cur)
      cur = objects.find((o) => o.id === cur)?.parent
    }
  }

  return errors
}

// ─── Scene validation ─────────────────────────────────────────────────────────

export function validateScene(raw: unknown): ValidationResult {
  const allErrors: string[] = []

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {
      scene:  { objects: [], lights: DEFAULT_LIGHTS, camera: DEFAULT_CAMERA },
      errors: [],
      fatal:  "Scene root must be a JSON object",
    }
  }
  const r = raw as Record<string, unknown>

  // objects
  const validObjects: SceneObject[] = []
  if (!Array.isArray(r.objects) || r.objects.length === 0) {
    allErrors.push("scene.objects must be a non-empty array")
  } else {
    for (let i = 0; i < r.objects.length; i++) {
      const { obj, errors } = validateObject(r.objects[i], i)
      allErrors.push(...errors)
      if (obj) validObjects.push(obj)
    }
  }

  // parent reference + cycle check (requires full object list)
  allErrors.push(...validateParentRefs(validObjects))

  // lights — optional, fall back to defaults
  let lights: LightDef[] = DEFAULT_LIGHTS
  if (r.lights !== undefined) {
    if (!Array.isArray(r.lights)) {
      allErrors.push("scene.lights must be an array")
    } else {
      const validLights: LightDef[] = []
      for (const l of r.lights as Record<string, unknown>[]) {
        if (!VALID_LIGHT_TYPES.includes(l?.type as LightType)) {
          allErrors.push(`light: type must be one of ${VALID_LIGHT_TYPES.join("|")}, got "${l?.type}"`)
          continue
        }
        if (!isNum(l.intensity, 0)) {
          allErrors.push(`light(${l.type}): intensity must be >= 0`)
          continue
        }
        if (l.color !== undefined && !isHex(l.color)) {
          allErrors.push(`light(${l.type}): color must be "#rrggbb" hex`)
          continue
        }
        if (l.position !== undefined && !isVec3(l.position)) {
          allErrors.push(`light(${l.type}): position must be [x,y,z]`)
          continue
        }
        validLights.push({
          type:       l.type as LightType,
          intensity:  l.intensity as number,
          color:      (l.color as string) || "#ffffff",
          position:   l.position as [number,number,number] | undefined,
          castShadow: (l.castShadow as boolean) ?? false,
        })
      }
      if (validLights.length > 0) lights = validLights
    }
  }

  // camera — optional, fall back to defaults
  let camera: CameraDef = DEFAULT_CAMERA
  if (r.camera !== undefined) {
    const c = r.camera as Record<string, unknown>
    if (!isVec3(c.position)) {
      allErrors.push("scene.camera.position must be [x,y,z]")
    } else if (!isVec3(c.target)) {
      allErrors.push("scene.camera.target must be [x,y,z]")
    } else {
      camera = {
        position: c.position,
        target:   c.target,
        fov:      isNum(c.fov, 1, 179) ? (c.fov as number) : DEFAULT_CAMERA.fov,
      }
    }
  }

  return {
    scene: {
      name:    typeof r.name === "string" ? r.name : undefined,
      objects: validObjects,
      lights,
      camera,
    },
    errors: allErrors,
  }
}

// ─── Demo scene (backend offline) ────────────────────────────────────────────

export const DEMO_SCENE: Record<string, unknown> = {
  name: "Demo",
  objects: [
    {
      id: "star", type: "primitive",
      geometry: { type: "sphere", radius: 2 },
      position: [0, 0, 0], scale: [1, 1, 1],
      material: { type: "standard", color: "#ffcc22", roughness: 0.3, metalness: 0.0, emissive: "#ff8800", emissiveIntensity: 0.8 },
      label: "Star",
      animation: { type: "none" },
    },
    {
      id: "planet_a", type: "primitive",
      geometry: { type: "sphere", radius: 1 },
      position: [7, 0, 0], scale: [1, 1, 1],
      material: { type: "standard", color: "#3b82f6", roughness: 0.7, metalness: 0.1 },
      label: "Planet A",
      animation: { type: "orbit", center: [0, 0, 0], speed: 0.5 },
    },
    {
      id: "moon_a", type: "primitive",
      geometry: { type: "sphere", radius: 1 },
      position: [1.5, 0, 0], scale: [0.2, 0.2, 0.2],
      parent: "planet_a",
      material: { type: "standard", color: "#cccccc", roughness: 0.95, metalness: 0.0 },
      animation: { type: "orbit", center: [0, 0, 0], speed: 4.0 },
    },
    {
      id: "planet_b", type: "primitive",
      geometry: { type: "sphere", radius: 1 },
      position: [11, 0, 0], scale: [0.55, 0.55, 0.55],
      material: { type: "standard", color: "#ef4444", roughness: 0.9, metalness: 0.0 },
      label: "Planet B",
      animation: { type: "orbit", center: [0, 0, 0], speed: 0.3 },
    },
    {
      id: "cube_a", type: "primitive",
      geometry: { type: "box", width: 1, height: 1, depth: 1 },
      position: [-7, 0, 2], scale: [1, 1, 1],
      material: { type: "standard", color: "#22c55e", roughness: 0.4, metalness: 0.6 },
      label: "Cube",
      animation: { type: "orbit", center: [0, 0, 0], speed: 0.4 },
    },
    {
      id: "spinner", type: "primitive",
      geometry: { type: "box", width: 1, height: 0.2, depth: 3 },
      position: [0, 4, 0], scale: [1, 1, 1],
      material: { type: "standard", color: "#8b5cf6", roughness: 0.2, metalness: 0.8 },
      animation: { type: "spin", axis: [0, 1, 0], speed: 1.5 },
    },
  ],
  lights: [
    { type: "ambient",     intensity: 0.3, color: "#ffffff" },
    { type: "directional", intensity: 1.5, color: "#ffffff", position: [15, 15, 10], castShadow: true },
    { type: "point",       intensity: 3.0, color: "#ffaa33", position: [0, 0, 0] },
  ],
  camera: { position: [0, 8, 25], target: [0, 0, 0], fov: 60 },
}
