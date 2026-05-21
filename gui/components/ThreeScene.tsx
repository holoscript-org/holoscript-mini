"use client"

/**
 * ThreeScene.tsx — Strict, generic Three.js renderer.
 *
 * Renders any scene that passes validateScene(). No normalization,
 * no special-casing by object ID or name.
 *
 * Architecture:
 *  - Scene graph: parent/child hierarchy via nested THREE.Groups
 *  - PBR materials: optional texture maps via useTexture (Suspense-backed)
 *  - Animations: orbit (local or world) + spin, computed each frame
 *  - ObjectRegistry: world-position lookup for center_ref orbit (non-parented)
 */

import { createContext, Suspense, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, useFrame, useThree, useLoader } from "@react-three/fiber"
import { Environment, OrbitControls, PerspectiveCamera, Text, useGLTF, useTexture } from "@react-three/drei"
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js"
import * as THREE from "three"
import type { GestureControlRefs } from "@/hooks/useGestureControl"
import {
  DEMO_SCENE,
  GeometryDef,
  LightDef,
  MaterialDef,
  SceneDef,
  SceneObject,
  validateScene,
} from "@/lib/sceneFactory"

// ─── Object registry (for center_ref world-space orbit) ──────────────────────

type RegistryRef = React.MutableRefObject<Map<string, THREE.Group>>
const ObjectRegistry = createContext<RegistryRef>({ current: new Map() })

// ─── Frozen context (gesture freeze stops all animation) ──────────────────────

const FrozenContext = createContext<boolean>(false)

// ─── Selected object context ─────────────────────────────────────────────────

const SelectedContext = createContext<string | null>(null)

// ─── Drag offsets context ───────────────────────────────────────────────────

type DragOffsetRef = React.MutableRefObject<Map<string, THREE.Vector3>>
const DragOffsetContext = createContext<DragOffsetRef>({ current: new Map() })

// ─── Material helpers ─────────────────────────────────────────────────────────

/** Returns THREE material parameters from a MaterialDef (no texture loading). */
function buildMaterialParams(m: MaterialDef): THREE.MeshStandardMaterialParameters {
  const params: THREE.MeshStandardMaterialParameters = {
    color:     new THREE.Color(m.color),
    roughness: m.roughness,
    metalness: m.metalness,
  }
  // Transparency: explicit flag or implied by opacity < 1
  if (m.transparent === true || (m.opacity !== undefined && m.opacity < 1)) {
    params.transparent = true
    params.opacity = m.opacity ?? 1.0
  }
  if (m.emissive) {
    params.emissive = new THREE.Color(m.emissive)
    params.emissiveIntensity = m.emissiveIntensity ?? 1.0
  }
  return params
}

/**
 * Renders a <meshStandardMaterial> with optional texture maps.
 * Must only be mounted when at least one texture URL is defined.
 * Suspends while textures are loading.
 */
function TexturedMat({ mat }: { mat: MaterialDef }) {
  const urls: Record<string, string> = {}
  if (mat.map)          urls.map          = mat.map
  if (mat.normalMap)    urls.normalMap    = mat.normalMap
  if (mat.roughnessMap) urls.roughnessMap = mat.roughnessMap
  if (mat.metalnessMap) urls.metalnessMap = mat.metalnessMap
  if (mat.emissiveMap)  urls.emissiveMap  = mat.emissiveMap
  const textures = useTexture(urls)
  return <meshStandardMaterial {...buildMaterialParams(mat)} {...(textures as object)} />
}

/** Picks plain or textured material automatically. */
function MaterialSlot({ mat }: { mat: MaterialDef }) {
  const hasTextures = !!(mat.map || mat.normalMap || mat.roughnessMap || mat.metalnessMap || mat.emissiveMap)
  if (!hasTextures) return <meshStandardMaterial {...buildMaterialParams(mat)} />
  return (
    <Suspense fallback={<meshStandardMaterial {...buildMaterialParams(mat)} />}>
      <TexturedMat mat={mat} />
    </Suspense>
  )
}

// ─── Geometry components ──────────────────────────────────────────────────────

interface GeomProps { geom: GeometryDef; mat: MaterialDef }

function CylinderGeom({ geom, mat }: GeomProps) {
  const r = geom.radius ?? 0.5

  if (geom.from && geom.to) {
    const fromV = new THREE.Vector3(...geom.from)
    const toV   = new THREE.Vector3(...geom.to)
    const dir   = toV.clone().sub(fromV)
    const len   = dir.length()
    const mid   = fromV.clone().add(toV).multiplyScalar(0.5)
    const quat  = new THREE.Quaternion()
    if (len > 1e-6) quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize())
    return (
      <mesh position={mid} quaternion={quat} castShadow receiveShadow>
        <cylinderGeometry args={[r, r, len, 32]} />
        <MaterialSlot mat={mat} />
      </mesh>
    )
  }

  return (
    <mesh castShadow receiveShadow>
      <cylinderGeometry args={[r, r, geom.height ?? 2, 32]} />
      <MaterialSlot mat={mat} />
    </mesh>
  )
}

function PrimitiveGeom({ geom, mat }: GeomProps) {
  switch (geom.type) {
    case "sphere":
      return (
        <mesh castShadow receiveShadow>
          <sphereGeometry args={[geom.radius ?? 1, 64, 64]} />
          <MaterialSlot mat={mat} />
        </mesh>
      )
    case "box":
      return (
        <mesh castShadow receiveShadow>
          <boxGeometry args={[geom.width ?? 1, geom.height ?? 1, geom.depth ?? 1]} />
          <MaterialSlot mat={mat} />
        </mesh>
      )
    case "cylinder":
      return <CylinderGeom geom={geom} mat={mat} />
    case "plane":
      return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[geom.width ?? 1, geom.depth ?? 1]} />
          <meshStandardMaterial {...buildMaterialParams(mat)} side={THREE.DoubleSide} />
        </mesh>
      )
    case "ring":
      return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} castShadow receiveShadow>
          <ringGeometry
            args={[
              geom.innerRadius ?? 0.8,
              geom.outerRadius ?? 1.3,
              Math.max(8, Math.floor(geom.thetaSegments ?? 96)),
            ]}
          />
          <meshStandardMaterial {...buildMaterialParams(mat)} side={THREE.DoubleSide} />
        </mesh>
      )
    case "capsule":
      return (
        <mesh castShadow receiveShadow>
          <capsuleGeometry args={[geom.radius ?? 0.5, geom.length ?? 1, 16, 32]} />
          <MaterialSlot mat={mat} />
        </mesh>
      )
    case "torus":
      return (
        <mesh castShadow receiveShadow>
          <torusGeometry args={[geom.radius ?? 1, geom.tube ?? 0.2, 16, 100]} />
          <MaterialSlot mat={mat} />
        </mesh>
      )
  }
}

// ─── GLTF mesh loader ─────────────────────────────────────────────────────────

function GltfObject({ obj }: { obj: SceneObject }) {
  const { scene } = useGLTF(obj.model!)
  const clone = useMemo(() => scene.clone(true), [scene])
  const params = useMemo(() => buildMaterialParams(obj.material), [obj.material])

  // #ffffff with no extras = "preserve the GLB's original vertex colors / textures"
  const preserveOriginal = useMemo(() => {
    const m = obj.material
    return (
      m.color === "#ffffff" &&
      !m.emissive && !m.map && !m.normalMap &&
      !m.roughnessMap && !m.metalnessMap
    )
  }, [obj.material])

  useEffect(() => {
    clone.traverse((child) => {
      const mesh = child as THREE.Mesh
      if (!mesh.isMesh) return
      if (!preserveOriginal) {
        mesh.material = new THREE.MeshStandardMaterial(params)
      }
      mesh.castShadow = true
      mesh.receiveShadow = true
    })
  }, [clone, params, preserveOriginal])

  return <primitive object={clone} />
}

// ─── OBJ mesh loader ──────────────────────────────────────────────────────────

function ObjObject({ obj }: { obj: SceneObject }) {
  const group = useLoader(OBJLoader, obj.model!)
  const clone = useMemo(() => group.clone(true), [group])
  const params = useMemo(() => buildMaterialParams(obj.material), [obj.material])

  useEffect(() => {
    clone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        ;(child as THREE.Mesh).material = new THREE.MeshStandardMaterial(params)
        ;(child as THREE.Mesh).castShadow = true
        ;(child as THREE.Mesh).receiveShadow = true
      }
    })
  }, [clone, params])

  return <primitive object={clone} />
}

// ─── Unified mesh loader ──────────────────────────────────────────────────────

function MeshObject({ obj }: { obj: SceneObject }) {
  const isObj = obj.model!.toLowerCase().endsWith(".obj")
  return isObj ? <ObjObject obj={obj} /> : <GltfObject obj={obj} />
}

// ─── Single scene object node ─────────────────────────────────────────────────

function SceneObjectNode({ obj, children }: { obj: SceneObject; children?: React.ReactNode }) {
  const registry = useContext(ObjectRegistry)
  const dragOffsets = useContext(DragOffsetContext)
  const groupRef  = useRef<THREE.Group>(null!)

  // Register this group for center_ref lookup and raycast identification
  useEffect(() => {
    registry.current.set(obj.id, groupRef.current)
    groupRef.current.userData.sceneObjectId = obj.id
    return () => { registry.current.delete(obj.id) }
  }, [obj.id, registry])

  const frozen     = useContext(FrozenContext)
  const selectedId = useContext(SelectedContext)
  const isSelected = selectedId === obj.id
  const anim      = obj.animation ?? { type: "none" }
  const isOrbit   = anim.type === "orbit"
  const hasParent = !!obj.parent

  // Orbit bookkeeping (local-space for parented, world-space otherwise)
  const staticCenter = anim.center ?? [0, 0, 0]
  const dx0 = obj.position[0] - staticCenter[0]
  const dz0 = obj.position[2] - staticCenter[2]
  const orbitRadius = useRef(Math.sqrt(dx0 * dx0 + dz0 * dz0))
  const orbitAngle  = useRef((anim.phase ?? 0) + Math.atan2(dz0, dx0))
  const orbitY      = useRef(obj.position[1])

  useFrame((_, delta) => {
    if (!groupRef.current || frozen) return

    const offset = dragOffsets.current.get(obj.id)

    if (isOrbit) {
      let cx = staticCenter[0], cy = orbitY.current, cz = staticCenter[2]
      // center_ref: world-space lookup, only meaningful for non-parented objects
      if (!hasParent && anim.center_ref) {
        const refGroup = registry.current.get(anim.center_ref)
        if (refGroup) { cx = refGroup.position.x; cy = refGroup.position.y; cz = refGroup.position.z }
      }
      orbitAngle.current += (anim.speed ?? 1.0) * delta
      groupRef.current.position.x = cx + orbitRadius.current * Math.cos(orbitAngle.current)
      groupRef.current.position.y = cy
      groupRef.current.position.z = cz + orbitRadius.current * Math.sin(orbitAngle.current)
      if (offset) {
        groupRef.current.position.x += offset.x
        groupRef.current.position.y += offset.y
        groupRef.current.position.z += offset.z
      }
    } else if (offset) {
      groupRef.current.position.set(
        obj.position[0] + offset.x,
        obj.position[1] + offset.y,
        obj.position[2] + offset.z
      )
    }

    if (anim.type === "spin") {
      const axis  = anim.axis ?? [0, 1, 0]
      const speed = (anim.speed ?? 1.0) * delta
      groupRef.current.rotation.x += axis[0] * speed
      groupRef.current.rotation.y += axis[1] * speed
      groupRef.current.rotation.z += axis[2] * speed
    }
  })

  // Convert Euler rotation from degrees → radians
  const rotRad = obj.rotation
    ? ([
        THREE.MathUtils.degToRad(obj.rotation[0]),
        THREE.MathUtils.degToRad(obj.rotation[1]),
        THREE.MathUtils.degToRad(obj.rotation[2]),
      ] as [number, number, number])
    : undefined

  // Label: stay above geometry
  const geomSize = obj.type === "primitive"
    ? Math.max(
        (obj.geometry?.radius ?? 1) * (obj.scale?.[1] ?? 1),
        ((obj.geometry?.height ?? 1) / 2) * (obj.scale?.[1] ?? 1)
      )
    : (obj.scale?.[1] ?? 1)
  const labelY = geomSize + 0.4

  return (
    <group
      ref={groupRef}
      position={obj.position}
      rotation={rotRad}
      scale={obj.scale ?? [1, 1, 1]}
    >
      {obj.type === "primitive" && obj.geometry && (
        <PrimitiveGeom geom={obj.geometry} mat={obj.material} />
      )}
      {obj.type === "mesh" && obj.model && (
        <Suspense fallback={null}>
          <MeshObject obj={obj} />
        </Suspense>
      )}
      {obj.label && (
        <Text
          position={[0, labelY / (obj.scale?.[1] ?? 1), 0]}
          fontSize={0.35}
          color="#00ffff"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.025}
          outlineColor="#000000"
          renderOrder={1}
        >
          {obj.label}
        </Text>
      )}
      {/* Selection glow — cyan point light at object centre */}
      {isSelected && (
        <pointLight color="#00ffff" intensity={6} distance={3} decay={2} />
      )}
      {/* Child objects in parent-local space */}
      {children}
    </group>
  )
}

// ─── Recursive object tree renderer ──────────────────────────────────────────

interface ObjectTreeProps {
  obj: SceneObject
  childMap: Map<string, SceneObject[]>
}

function ObjectTree({ obj, childMap }: ObjectTreeProps) {
  const children = childMap.get(obj.id) ?? []
  return (
    <SceneObjectNode obj={obj}>
      {children.map((child) => (
        <ObjectTree key={child.id} obj={child} childMap={childMap} />
      ))}
    </SceneObjectNode>
  )
}

// ─── Lights ───────────────────────────────────────────────────────────────────

function SceneLights({ lights }: { lights: LightDef[] }) {
  return (
    <>
      {lights.map((l, i) => {
        switch (l.type) {
          case "ambient":
            return <ambientLight key={i} intensity={l.intensity} color={l.color ?? "#ffffff"} />
          case "directional":
            return (
              <directionalLight key={i}
                position={l.position ?? [10, 10, 10]}
                intensity={l.intensity}
                color={l.color ?? "#ffffff"}
                castShadow={l.castShadow ?? false}
                shadow-mapSize={[2048, 2048]}
                shadow-camera-near={0.5}
                shadow-camera-far={500}
                shadow-camera-left={-60}
                shadow-camera-right={60}
                shadow-camera-top={60}
                shadow-camera-bottom={-60}
              />
            )
          case "point":
            return (
              <pointLight key={i}
                position={l.position ?? [0, 0, 0]}
                intensity={l.intensity}
                color={l.color ?? "#ffffff"}
              />
            )
          case "spot":
            return (
              <spotLight key={i}
                position={l.position ?? [0, 10, 0]}
                intensity={l.intensity}
                color={l.color ?? "#ffffff"}
                castShadow={l.castShadow ?? false}
              />
            )
          default:
            return null
        }
      })}
    </>
  )
}

// ─── FPS tracker (inside Canvas) ─────────────────────────────────────────────

function FPSMeter({
  fpsRef,
  onFpsUpdate,
}: {
  fpsRef: React.MutableRefObject<number>
  onFpsUpdate?: (fps: number) => void
}) {
  const frames      = useRef(0)
  const lastTime    = useRef(performance.now())
  const onUpdateRef = useRef(onFpsUpdate)
  useEffect(() => { onUpdateRef.current = onFpsUpdate }, [onFpsUpdate])

  useFrame(() => {
    frames.current++
    const now = performance.now()
    if (now - lastTime.current >= 500) {
      const fps = Math.round((frames.current * 1000) / (now - lastTime.current))
      fpsRef.current   = fps
      frames.current   = 0
      lastTime.current = now
      onUpdateRef.current?.(fps)
    }
  })
  return null
}

// ─── Raycast selector (inside Canvas) ────────────────────────────────────────

interface SelectTrigger {
  x: number   // normalised 0–1 (camera space, pre-mirror)
  y: number   // normalised 0–1
  seq: number // increments on each new trigger so useFrame can diff
}

function RaycastSelector({
  triggerRef,
  onSelect,
}: {
  triggerRef: React.MutableRefObject<SelectTrigger | null>
  onSelect: (id: string | null) => void
}) {
  const { camera, scene } = useThree()
  const raycaster  = useRef(new THREE.Raycaster())
  const lastSeqRef = useRef(-1)

  useFrame(() => {
    const t = triggerRef.current
    if (!t || t.seq === lastSeqRef.current) return
    lastSeqRef.current = t.seq

    // Flip x for mirror, invert y for NDC
    const ndcX = 1 - 2 * t.x
    const ndcY = 1 - 2 * t.y
    raycaster.current.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera)
    const hits = raycaster.current.intersectObjects(scene.children, true)

    for (const hit of hits) {
      let obj: THREE.Object3D | null = hit.object
      while (obj) {
        if (obj.userData.sceneObjectId) {
          onSelect(obj.userData.sceneObjectId as string)
          return
        }
        obj = obj.parent
      }
    }
    onSelect(null)
  })
  return null
}

function GestureSceneRoot({
  controls,
  rotationY,
  scale,
  frozen,
  sceneOffsetRef,
  sceneRootRef,
  children,
}: {
  controls?: GestureControlRefs
  rotationY: number
  scale: number
  frozen: boolean
  sceneOffsetRef?: React.MutableRefObject<THREE.Vector3>
  sceneRootRef?: React.MutableRefObject<THREE.Group | null>
  children: React.ReactNode
}) {
  const groupRef = useRef<THREE.Group>(null)

  useEffect(() => {
    if (sceneRootRef) sceneRootRef.current = groupRef.current
  }, [sceneRootRef])

  useFrame(() => {
    if (!groupRef.current) return

    const nextRotationY = controls?.rotationYRef.current ?? rotationY
    const nextScale = controls?.scaleRef.current ?? scale
    groupRef.current.rotation.y = THREE.MathUtils.degToRad(nextRotationY)
    groupRef.current.scale.setScalar(nextScale)
    if (sceneOffsetRef) {
      groupRef.current.position.copy(sceneOffsetRef.current)
    }
  })

  return (
    <FrozenContext.Provider value={frozen}>
      <group ref={groupRef} rotation={[0, THREE.MathUtils.degToRad(rotationY), 0]} scale={[scale, scale, scale]}>
        {children}
      </group>
    </FrozenContext.Provider>
  )
}

// ─── Debug panel (outside Canvas) ────────────────────────────────────────────

function DebugPanel({
  scene, isDemo, fps, errors, textureMaps, meshCount,
}: {
  scene: SceneDef
  isDemo: boolean
  fps: number
  errors: string[]
  textureMaps: number
  meshCount: number
}) {
  return (
    <div className="absolute top-3 left-3 flex flex-col gap-1 pointer-events-none select-none">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono text-primary/70 bg-black/60 px-1.5 py-0.5 rounded">
          {scene.name ?? "Scene"}
        </span>
        {isDemo && (
          <span className="text-[10px] font-mono text-yellow-400/80 border border-yellow-400/30 bg-black/60 px-1.5 py-0.5 rounded">
            DEMO
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] font-mono text-primary/50 bg-black/40 px-1.5 py-0.5 rounded">
          {scene.objects.length} obj
        </span>
        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 ${
          fps >= 50 ? "text-green-400/70" : fps >= 30 ? "text-yellow-400/70" : "text-red-400/70"
        }`}>
          {fps} fps
        </span>
        {meshCount > 0 && (
          <span className="text-[10px] font-mono text-blue-400/60 bg-black/40 px-1.5 py-0.5 rounded">
            {meshCount} mesh
          </span>
        )}
        {textureMaps > 0 && (
          <span className="text-[10px] font-mono text-purple-400/60 bg-black/40 px-1.5 py-0.5 rounded">
            {textureMaps} tex
          </span>
        )}
        <span className="text-[10px] font-mono text-primary/30 bg-black/40 px-1.5 py-0.5 rounded">
          WebGL
        </span>
      </div>
      {errors.length > 0 && (
        <div className="mt-0.5 max-w-xs">
          {errors.slice(0, 3).map((e, i) => (
            <div key={i} className="text-[9px] font-mono text-red-400/70 bg-black/60 px-1.5 py-0.5 rounded truncate">
              {e}
            </div>
          ))}
          {errors.length > 3 && (
            <div className="text-[9px] font-mono text-red-400/50 bg-black/40 px-1.5 py-0.5 rounded">
              +{errors.length - 3} more errors
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

interface ThreeSceneProps {
  scene: Record<string, unknown>
  controls?: GestureControlRefs
  rotationY?: number
  scale?: number
  frozen?: boolean
  selectTrigger?: SelectTrigger | null
  reticlePoint?: { x: number; y: number } | null
  sceneDragMode?: boolean
  isDragActiveRef?: React.MutableRefObject<boolean>
  resetSeq?: number
  deselectSeq?: number
  onObjectSelect?: (id: string | null) => void
  onFpsUpdate?: (fps: number) => void
}

export function ThreeScene({
  scene,
  controls,
  rotationY = 0,
  scale = 1,
  frozen = false,
  selectTrigger,
  reticlePoint,
  sceneDragMode = false,
  isDragActiveRef,
  resetSeq = 0,
  deselectSeq = 0,
  onObjectSelect,
  onFpsUpdate,
}: ThreeSceneProps) {
  const [mounted, setMounted]     = useState(false)
  const [fps, setFps]             = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const fpsRef                    = useRef(0)
  const registryRef               = useRef<Map<string, THREE.Group>>(new Map())
  const selectTriggerRef          = useRef<SelectTrigger | null>(null)
  const onObjectSelectRef         = useRef(onObjectSelect)
  const dragOffsetsRef            = useRef<Map<string, THREE.Vector3>>(new Map())
  const sceneOffsetRef            = useRef(new THREE.Vector3(0, 0, 0))
  const sceneRootRef              = useRef<THREE.Group | null>(null)
  const orbitRef                  = useRef<unknown>(null)
  const initialCamRef             = useRef<{ position: [number, number, number]; target: [number, number, number] } | null>(null)
  const lastSceneSigRef           = useRef<string>("")

  useEffect(() => { onObjectSelectRef.current = onObjectSelect }, [onObjectSelect])
  useEffect(() => { selectTriggerRef.current = selectTrigger ?? null }, [selectTrigger])

  const handleSelect = useCallback((id: string | null) => {
    setSelectedId(id)
    onObjectSelectRef.current?.(id)
  }, [])

  useEffect(() => {
    setMounted(true)
    const id = setInterval(() => setFps(fpsRef.current), 500)
    return () => clearInterval(id)
  }, [])

  const isEmpty  = !scene || Object.keys(scene).length === 0
  const rawInput = isEmpty ? DEMO_SCENE : scene

  const { scene: validScene, errors, fatal } = useMemo(
    () => validateScene(rawInput),
    [rawInput]
  )

  // Build parent → children map for scene-graph hierarchy
  const childMap = useMemo(() => {
    const map = new Map<string, SceneObject[]>()
    for (const obj of validScene.objects) {
      if (obj.parent) {
        const list = map.get(obj.parent) ?? []
        list.push(obj)
        map.set(obj.parent, list)
      }
    }
    return map
  }, [validScene.objects])

  // Root objects = objects with no parent
  const rootObjects = useMemo(
    () => validScene.objects.filter((o) => !o.parent),
    [validScene.objects]
  )

  // Stats for debug panel
  const textureMaps = useMemo(
    () => validScene.objects.filter((o) =>
      o.material.map || o.material.normalMap || o.material.roughnessMap ||
      o.material.metalnessMap || o.material.emissiveMap
    ).length,
    [validScene.objects]
  )
  const meshCount = useMemo(
    () => validScene.objects.filter((o) => o.type === "mesh").length,
    [validScene.objects]
  )

  const cam = validScene.camera

  useEffect(() => {
    const sig = JSON.stringify({ pos: cam.position, tgt: cam.target })
    if (lastSceneSigRef.current !== sig) {
      lastSceneSigRef.current = sig
      initialCamRef.current = {
        position: [cam.position[0], cam.position[1], cam.position[2]],
        target: [cam.target[0], cam.target[1], cam.target[2]],
      }
    }
  }, [cam.position, cam.target])

  useEffect(() => {
    dragOffsetsRef.current.clear()
    sceneOffsetRef.current.set(0, 0, 0)
    setSelectedId(null)
  }, [resetSeq])

  // Drop gesture: deselect without touching camera or drag offsets
  useEffect(() => {
    if (deselectSeq === 0) return
    setSelectedId(null)
  }, [deselectSeq])

  if (!mounted) {
    return (
      <div className="w-full h-full flex items-center justify-center rounded-lg border border-primary/20 bg-black/30">
        <span className="text-xs font-mono text-primary/50 animate-pulse">Initializing 3D Engine…</span>
      </div>
    )
  }

  if (fatal) {
    return (
      <div className="w-full h-full flex items-center justify-center rounded-lg border border-red-500/30 bg-black/30">
        <span className="text-xs font-mono text-red-400/70">Scene error: {fatal}</span>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full rounded-lg overflow-hidden border border-primary/20 bg-black">
      <SelectedContext.Provider value={selectedId}>
        <ObjectRegistry.Provider value={registryRef}>
          <DragOffsetContext.Provider value={dragOffsetsRef}>
          <Canvas
            shadows
            dpr={[1, 2]}
            frameloop="always"
            gl={{ toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.0, antialias: true }}
            onCreated={({ gl }) => {
              gl.shadowMap.enabled = true
              gl.shadowMap.type    = THREE.PCFSoftShadowMap
            }}
          >
            <PerspectiveCamera makeDefault position={cam.position} fov={cam.fov ?? 65} near={0.05} far={5000} />
            <OrbitControls
              ref={orbitRef}
              target={cam.target}
              enablePan
              enableZoom
              enableRotate
              minDistance={0.5}
              maxDistance={2000}
            />

            <SceneLights lights={validScene.lights} />

            <Suspense fallback={null}>
              <Environment preset="city" />
            </Suspense>

            <GestureSceneRoot
              controls={controls}
              rotationY={rotationY}
              scale={scale}
              frozen={frozen}
              sceneOffsetRef={sceneOffsetRef}
              sceneRootRef={sceneRootRef}
            >
              <Suspense fallback={null}>
                {rootObjects.map((obj) => (
                  <ObjectTree key={obj.id} obj={obj} childMap={childMap} />
                ))}
              </Suspense>
            </GestureSceneRoot>

            <FPSMeter fpsRef={fpsRef} onFpsUpdate={onFpsUpdate} />
            <RaycastSelector triggerRef={selectTriggerRef} onSelect={handleSelect} />
            <CameraResetController resetSeq={resetSeq} cam={initialCamRef.current ?? cam} orbitRef={orbitRef} />
            <DragController
              reticlePoint={reticlePoint}
              selectedId={selectedId}
              registryRef={registryRef}
              dragOffsetsRef={dragOffsetsRef}
              sceneOffsetRef={sceneOffsetRef}
              sceneRootRef={sceneRootRef}
              sceneDragMode={sceneDragMode}
              isDragActiveRef={isDragActiveRef}
            />
          </Canvas>
          </DragOffsetContext.Provider>
        </ObjectRegistry.Provider>
      </SelectedContext.Provider>

      {reticlePoint && (
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute h-6 w-6 rounded-full border-2 border-yellow-400 shadow-[0_0_10px_rgba(250,204,21,0.9)]"
            style={{
              left: `${(1 - reticlePoint.x) * 100}%`,
              top: `${reticlePoint.y * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
          <div
            className="absolute h-3 w-3 rounded-full border border-yellow-300/80"
            style={{
              left: `${(1 - reticlePoint.x) * 100}%`,
              top: `${reticlePoint.y * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
          <div
            className="absolute h-px w-7 bg-yellow-300/90"
            style={{
              left: `${(1 - reticlePoint.x) * 100}%`,
              top: `${reticlePoint.y * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
          <div
            className="absolute h-7 w-px bg-yellow-300/90"
            style={{
              left: `${(1 - reticlePoint.x) * 100}%`,
              top: `${reticlePoint.y * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
        </div>
      )}

      <DebugPanel
        scene={validScene}
        isDemo={isEmpty}
        fps={fps}
        errors={errors}
        textureMaps={textureMaps}
        meshCount={meshCount}
      />
    </div>
  )
}

// ─── Camera reset controller (inside Canvas) ───────────────────────────────

function CameraResetController({
  resetSeq,
  cam,
  orbitRef,
}: {
  resetSeq: number
  cam: { position: [number, number, number]; target: [number, number, number] }
  orbitRef: React.MutableRefObject<unknown>
}) {
  const { camera } = useThree()

  useEffect(() => {
    camera.position.set(cam.position[0], cam.position[1], cam.position[2])
    camera.lookAt(cam.target[0], cam.target[1], cam.target[2])
    camera.updateProjectionMatrix()

    const controls = orbitRef.current as {
      target?: THREE.Vector3
      update?: () => void
      reset?: () => void
      saveState?: () => void
    } | null
    if (controls?.target) {
      controls.target.set(cam.target[0], cam.target[1], cam.target[2])
      controls.update?.()
      controls.saveState?.()
    }
    controls?.reset?.()
  }, [resetSeq, cam, camera, orbitRef])

  return null
}

// ─── Drag controller (inside Canvas) ─────────────────────────────────────────

function DragController({
  reticlePoint,
  selectedId,
  registryRef,
  dragOffsetsRef,
  sceneOffsetRef,
  sceneRootRef,
  sceneDragMode,
  isDragActiveRef,
}: {
  reticlePoint?: { x: number; y: number } | null
  selectedId: string | null
  registryRef: React.MutableRefObject<Map<string, THREE.Group>>
  dragOffsetsRef: React.MutableRefObject<Map<string, THREE.Vector3>>
  sceneOffsetRef: React.MutableRefObject<THREE.Vector3>
  sceneRootRef: React.MutableRefObject<THREE.Group | null>
  sceneDragMode: boolean
  isDragActiveRef?: React.MutableRefObject<boolean>
}) {
  const { camera } = useThree()
  const raycaster = useRef(new THREE.Raycaster())
  const plane = useRef(new THREE.Plane())
  const dragModeRef = useRef<"scene" | "object" | null>(null)
  const dragTargetRef = useRef<string | null>(null)
  const grabOffsetRef = useRef(new THREE.Vector3())
  const baseLocalRef = useRef(new THREE.Vector3())

  useFrame(() => {
    if (!reticlePoint) {
      dragModeRef.current = null
      dragTargetRef.current = null
      return
    }

    // Drag only fires when explicitly activated by a PINCH-while-pointing gesture
    if (isDragActiveRef && !isDragActiveRef.current) {
      dragModeRef.current = null
      dragTargetRef.current = null
      return
    }

    const targetMode: "scene" | "object" | null = sceneDragMode
      ? "scene"
      : selectedId
        ? "object"
        : null

    if (!targetMode) return

    const ndcX = 1 - 2 * reticlePoint.x
    const ndcY = 1 - 2 * reticlePoint.y
    raycaster.current.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera)

    if (dragModeRef.current !== targetMode || dragTargetRef.current !== selectedId) {
      dragModeRef.current = targetMode
      dragTargetRef.current = selectedId

      const camDir = new THREE.Vector3()
      camera.getWorldDirection(camDir)

      if (targetMode === "scene") {
        const root = sceneRootRef.current
        if (!root) return
        const worldPos = root.getWorldPosition(new THREE.Vector3())
        plane.current.setFromNormalAndCoplanarPoint(camDir, worldPos)
        const hit = raycaster.current.ray.intersectPlane(plane.current, new THREE.Vector3())
        if (hit) grabOffsetRef.current.copy(worldPos).sub(hit)
        return
      }

      if (!selectedId) return
      const group = registryRef.current.get(selectedId)
      if (!group) return
      const worldPos = group.getWorldPosition(new THREE.Vector3())
      plane.current.setFromNormalAndCoplanarPoint(camDir, worldPos)
      const hit = raycaster.current.ray.intersectPlane(plane.current, new THREE.Vector3())
      if (hit) grabOffsetRef.current.copy(worldPos).sub(hit)
      const base = group.parent
        ? group.parent.worldToLocal(worldPos.clone())
        : worldPos.clone()
      baseLocalRef.current.copy(base)
    }

    const hit = raycaster.current.ray.intersectPlane(plane.current, new THREE.Vector3())
    if (!hit) return
    const desiredWorld = hit.add(grabOffsetRef.current)

    if (targetMode === "scene") {
      sceneOffsetRef.current.copy(desiredWorld)
      return
    }

    if (!selectedId) return
    const group = registryRef.current.get(selectedId)
    if (!group) return

    const localPos = group.parent
      ? group.parent.worldToLocal(desiredWorld.clone())
      : desiredWorld.clone()
    const offset = localPos.sub(baseLocalRef.current)
    const stored = dragOffsetsRef.current.get(selectedId)
    if (stored) {
      stored.copy(offset)
    } else {
      dragOffsetsRef.current.set(selectedId, offset.clone())
    }
  })

  return null
}
