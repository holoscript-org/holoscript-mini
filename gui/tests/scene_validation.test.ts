/**
 * scene_validation.test.ts
 *
 * Validates every JSON file in core/assets/examples/ against the strict
 * scene schema defined in sceneFactory.ts.
 *
 * Run:  npm test
 */

import { readFileSync, readdirSync } from "fs"
import { join } from "path"
import { describe, it, expect } from "vitest"
import {
  validateScene,
  VALID_OBJECT_TYPES,
  VALID_GEOM_TYPES,
  VALID_ANIM_TYPES,
  VALID_LIGHT_TYPES,
  DEMO_SCENE,
  type SceneDef,
  type SceneObject,
} from "@/lib/sceneFactory"

// ─── Helpers ──────────────────────────────────────────────────────────────────

const EXAMPLES_DIR = join(__dirname, "../../core/assets/examples")

function loadJson(filename: string): Record<string, unknown> {
  const raw = readFileSync(join(EXAMPLES_DIR, filename), "utf-8")
  return JSON.parse(raw) as Record<string, unknown>
}

function exampleFiles(): string[] {
  return readdirSync(EXAMPLES_DIR).filter((f) => f.endsWith(".json"))
}

function minimalPrimitive(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "obj1", type: "primitive",
    geometry: { type: "sphere", radius: 1 },
    position: [0, 0, 0],
    material: { type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0 },
    animation: { type: "none" },
    ...overrides,
  }
}

function minimalScene(objects: unknown[]): Record<string, unknown> {
  return {
    objects,
    lights: [{ type: "ambient", intensity: 0.5 }],
    camera: { position: [0, 5, 10], target: [0, 0, 0] },
  }
}

// ─── Schema unit tests ────────────────────────────────────────────────────────

describe("validateScene — unit tests", () => {
  it("rejects null input with fatal error", () => {
    const result = validateScene(null)
    expect(result.fatal).toBeTruthy()
    expect(result.scene.objects).toHaveLength(0)
  })

  it("rejects a string input with fatal error", () => {
    const result = validateScene("bad input")
    expect(result.fatal).toBeTruthy()
  })

  it("rejects empty objects array", () => {
    const result = validateScene({ objects: [], lights: [], camera: { position: [0,0,10], target: [0,0,0] } })
    expect(result.errors.some((e) => e.includes("objects"))).toBe(true)
  })

  it("accepts a minimal valid scene", () => {
    const result = validateScene(minimalScene([minimalPrimitive()]))
    expect(result.fatal).toBeUndefined()
    expect(result.scene.objects).toHaveLength(1)
    expect(result.errors).toHaveLength(0)
  })

  it("rejects object with missing id", () => {
    const obj = minimalPrimitive()
    delete obj.id
    const result = validateScene(minimalScene([obj]))
    expect(result.errors.some((e) => e.includes("id"))).toBe(true)
  })

  it("rejects object with invalid type", () => {
    const result = validateScene(minimalScene([{ ...minimalPrimitive(), type: "sphere" }]))
    expect(result.errors.some((e) => e.includes("primitive") || e.includes("type"))).toBe(true)
    expect(result.scene.objects).toHaveLength(0)
  })

  it("rejects primitive without geometry", () => {
    const obj = minimalPrimitive()
    delete obj.geometry
    const result = validateScene(minimalScene([obj]))
    expect(result.errors.some((e) => e.includes("geometry"))).toBe(true)
  })

  it("rejects mesh without model path", () => {
    const result = validateScene(minimalScene([{
      id: "mesh1", type: "mesh",
      position: [0, 0, 0],
      material: { type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0 },
    }]))
    expect(result.errors.some((e) => e.includes("model"))).toBe(true)
  })

  it("rejects invalid hex color", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ material: { type: "standard", color: "red", roughness: 0.5, metalness: 0.1 } }),
    ]))
    expect(result.errors.some((e) => e.includes("color"))).toBe(true)
  })

  it("rejects out-of-range roughness", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ material: { type: "standard", color: "#ffffff", roughness: 1.5, metalness: 0.0 } }),
    ]))
    expect(result.errors.some((e) => e.includes("roughness"))).toBe(true)
  })

  it("rejects invalid position vector", () => {
    const result = validateScene(minimalScene([minimalPrimitive({ position: [0, 0] })]))
    expect(result.errors.some((e) => e.includes("position"))).toBe(true)
  })

  it("rejects unknown animation type", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ animation: { type: "float" } }),
    ]))
    expect(result.errors.some((e) => e.includes("animation"))).toBe(true)
  })

  it("accepts orbit animation with center_ref", () => {
    const scene = minimalScene([
      minimalPrimitive({ id: "planet", position: [5, 0, 0], animation: { type: "orbit", center: [0,0,0], speed: 1.0 } }),
      minimalPrimitive({ id: "moon",   position: [6.5, 0, 0], animation: { type: "orbit", center_ref: "planet", speed: 4.0 } }),
    ])
    const result = validateScene(scene)
    expect(result.fatal).toBeUndefined()
    expect(result.scene.objects).toHaveLength(2)
    expect(result.errors).toHaveLength(0)
    const moon = result.scene.objects.find((o) => o.id === "moon")!
    expect(moon.animation?.center_ref).toBe("planet")
  })

  it("accepts cylinder with from/to geometry", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({
        id: "bond",
        geometry: { type: "cylinder", radius: 0.1, from: [0, 0, 0], to: [1, 1, 0] },
      }),
    ]))
    expect(result.fatal).toBeUndefined()
    expect(result.scene.objects[0].geometry?.from).toEqual([0, 0, 0])
    expect(result.scene.objects[0].geometry?.to).toEqual([1, 1, 0])
  })

  it("skips invalid objects but keeps valid ones", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "good" }),
      { id: "bad", type: "sphere" },
      minimalPrimitive({ id: "good2", geometry: { type: "box", width: 1, height: 1, depth: 1 }, position: [3,0,0] }),
    ]))
    expect(result.scene.objects).toHaveLength(2)
    expect(result.errors.length).toBeGreaterThan(0)
    expect(result.scene.objects.map((o) => o.id)).toContain("good")
    expect(result.scene.objects.map((o) => o.id)).toContain("good2")
    expect(result.scene.objects.map((o) => o.id)).not.toContain("bad")
  })

  it("falls back to default camera when scene.camera is missing", () => {
    const result = validateScene({ objects: [minimalPrimitive()] })
    expect(result.scene.camera).toBeDefined()
    expect(result.scene.camera.position).toHaveLength(3)
  })

  it("falls back to default lights when scene.lights is missing", () => {
    const result = validateScene({ objects: [minimalPrimitive()] })
    expect(result.scene.lights.length).toBeGreaterThan(0)
  })
})

// ─── New geometry types ───────────────────────────────────────────────────────

describe("validateScene — new geometry types", () => {
  it("accepts capsule geometry", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "cap", geometry: { type: "capsule", radius: 0.5, length: 2.0 } }),
    ]))
    expect(result.fatal).toBeUndefined()
    expect(result.scene.objects).toHaveLength(1)
    expect(result.errors).toHaveLength(0)
    expect(result.scene.objects[0].geometry?.type).toBe("capsule")
    expect(result.scene.objects[0].geometry?.radius).toBe(0.5)
    expect(result.scene.objects[0].geometry?.length).toBe(2.0)
  })

  it("accepts torus geometry", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "torus1", geometry: { type: "torus", radius: 2.0, tube: 0.3 } }),
    ]))
    expect(result.fatal).toBeUndefined()
    expect(result.scene.objects).toHaveLength(1)
    expect(result.errors).toHaveLength(0)
    expect(result.scene.objects[0].geometry?.type).toBe("torus")
    expect(result.scene.objects[0].geometry?.tube).toBe(0.3)
  })

  it("rejects capsule with invalid length", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ geometry: { type: "capsule", radius: 0.5, length: -1 } }),
    ]))
    expect(result.errors.some((e) => e.includes("length"))).toBe(true)
  })

  it("rejects torus with invalid tube radius", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ geometry: { type: "torus", radius: 1.0, tube: -0.1 } }),
    ]))
    expect(result.errors.some((e) => e.includes("tube"))).toBe(true)
  })
})

// ─── Material: textures + transparency ───────────────────────────────────────

describe("validateScene — material texture maps and transparency", () => {
  it("accepts material with texture map paths", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({
        material: {
          type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0,
          map:          "/textures/diffuse.jpg",
          normalMap:    "/textures/normal.jpg",
          roughnessMap: "/textures/roughness.jpg",
          metalnessMap: "/textures/metalness.jpg",
          emissiveMap:  "/textures/emissive.jpg",
        },
      }),
    ]))
    expect(result.fatal).toBeUndefined()
    expect(result.errors).toHaveLength(0)
    const mat = result.scene.objects[0].material
    expect(mat.map).toBe("/textures/diffuse.jpg")
    expect(mat.normalMap).toBe("/textures/normal.jpg")
    expect(mat.roughnessMap).toBe("/textures/roughness.jpg")
    expect(mat.metalnessMap).toBe("/textures/metalness.jpg")
    expect(mat.emissiveMap).toBe("/textures/emissive.jpg")
  })

  it("accepts transparent material with opacity", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({
        material: { type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0, opacity: 0.4, transparent: true },
      }),
    ]))
    expect(result.errors).toHaveLength(0)
    const mat = result.scene.objects[0].material
    expect(mat.opacity).toBe(0.4)
    expect(mat.transparent).toBe(true)
  })

  it("rejects non-string texture path", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({
        material: { type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0, map: 123 },
      }),
    ]))
    expect(result.errors.some((e) => e.includes("map"))).toBe(true)
  })

  it("rejects non-boolean transparent field", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({
        material: { type: "standard", color: "#ffffff", roughness: 0.5, metalness: 0.0, transparent: "yes" },
      }),
    ]))
    expect(result.errors.some((e) => e.includes("transparent"))).toBe(true)
  })
})

// ─── Parent / scene graph ─────────────────────────────────────────────────────

describe("validateScene — parent/child hierarchy", () => {
  it("accepts valid parent reference", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "parent_obj", position: [5, 0, 0] }),
      minimalPrimitive({ id: "child_obj",  position: [1, 0, 0], parent: "parent_obj" }),
    ]))
    expect(result.fatal).toBeUndefined()
    expect(result.errors).toHaveLength(0)
    expect(result.scene.objects).toHaveLength(2)
    const child = result.scene.objects.find((o) => o.id === "child_obj")!
    expect(child.parent).toBe("parent_obj")
  })

  it("rejects unknown parent id", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "child_obj", position: [1, 0, 0], parent: "does_not_exist" }),
    ]))
    expect(result.errors.some((e) => e.includes("does_not_exist"))).toBe(true)
  })

  it("rejects self-referencing parent", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "self_ref", parent: "self_ref" }),
    ]))
    expect(result.errors.some((e) => e.includes("self"))).toBe(true)
  })

  it("rejects circular parent dependency", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "a", position: [0, 0, 0], parent: "b" }),
      minimalPrimitive({ id: "b", position: [1, 0, 0], parent: "a" }),
    ]))
    expect(result.errors.some((e) => e.includes("circular"))).toBe(true)
  })

  it("accepts deep valid parent chain (3 levels)", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "root",  position: [0, 0, 0] }),
      minimalPrimitive({ id: "child", position: [1, 0, 0], parent: "root" }),
      minimalPrimitive({ id: "grand", position: [1, 0, 0], parent: "child" }),
    ]))
    expect(result.errors).toHaveLength(0)
    expect(result.scene.objects).toHaveLength(3)
  })

  it("orbit animation with parent uses local center", () => {
    const result = validateScene(minimalScene([
      minimalPrimitive({ id: "planet", position: [5, 0, 0], animation: { type: "orbit", center: [0,0,0], speed: 1 } }),
      minimalPrimitive({ id: "moon",   position: [1, 0, 0], parent: "planet",
                         animation: { type: "orbit", center: [0,0,0], speed: 4 } }),
    ]))
    expect(result.errors).toHaveLength(0)
    const moon = result.scene.objects.find((o) => o.id === "moon")!
    expect(moon.parent).toBe("planet")
    expect(moon.animation?.type).toBe("orbit")
  })
})

// ─── DEMO_SCENE validates cleanly ────────────────────────────────────────────

describe("DEMO_SCENE", () => {
  it("passes validation with zero errors", () => {
    const result = validateScene(DEMO_SCENE)
    expect(result.fatal).toBeUndefined()
    expect(result.errors).toHaveLength(0)
    expect(result.scene.objects.length).toBeGreaterThan(0)
  })

  it("moon_a is parented to planet_a", () => {
    const result = validateScene(DEMO_SCENE)
    const moon = result.scene.objects.find((o) => o.id === "moon_a")
    expect(moon).toBeDefined()
    expect(moon!.parent).toBe("planet_a")
  })
})

// ─── Constants exhaustiveness ─────────────────────────────────────────────────

describe("Schema constants", () => {
  it("VALID_OBJECT_TYPES covers primitive and mesh", () => {
    expect(VALID_OBJECT_TYPES).toContain("primitive")
    expect(VALID_OBJECT_TYPES).toContain("mesh")
  })
  it("VALID_GEOM_TYPES covers all seven geometry types", () => {
    for (const t of ["sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"] as const) {
      expect(VALID_GEOM_TYPES).toContain(t)
    }
  })
  it("VALID_ANIM_TYPES covers none, orbit, spin", () => {
    expect(VALID_ANIM_TYPES).toContain("none")
    expect(VALID_ANIM_TYPES).toContain("orbit")
    expect(VALID_ANIM_TYPES).toContain("spin")
  })
  it("VALID_LIGHT_TYPES covers all four types", () => {
    for (const t of ["ambient", "directional", "point", "spot"] as const) {
      expect(VALID_LIGHT_TYPES).toContain(t)
    }
  })
})

// ─── Example JSON files ───────────────────────────────────────────────────────

describe("Example JSON files", () => {
  const files = exampleFiles()

  it("finds at least 10 example JSON files", () => {
    expect(files.length).toBeGreaterThanOrEqual(10)
  })

  for (const file of files) {
    describe(file, () => {
      let raw: Record<string, unknown>

      it("is valid JSON", () => {
        expect(() => { raw = loadJson(file) }).not.toThrow()
      })

      it("passes validateScene with no fatal error", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        expect(result.fatal).toBeUndefined()
      })

      it("has at least one valid object", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        expect(result.scene.objects.length).toBeGreaterThan(0)
      })

      it("has zero schema errors", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        if (result.errors.length > 0) {
          console.warn(`[${file}] errors:`, result.errors)
        }
        expect(result.errors).toHaveLength(0)
      })

      it("all objects have valid type (primitive|mesh)", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        for (const obj of result.scene.objects) {
          expect(VALID_OBJECT_TYPES).toContain(obj.type)
        }
      })

      it("all primitives have valid geometry type", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        for (const obj of result.scene.objects) {
          if (obj.type === "primitive") {
            expect(obj.geometry).toBeDefined()
            expect(VALID_GEOM_TYPES).toContain(obj.geometry!.type)
          }
        }
      })

      it("all objects have valid #rrggbb material color", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        for (const obj of result.scene.objects) {
          expect(obj.material.color).toMatch(/^#[0-9a-fA-F]{6}$/)
        }
      })

      it("all animations have valid type", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        for (const obj of result.scene.objects) {
          if (obj.animation) {
            expect(VALID_ANIM_TYPES).toContain(obj.animation.type)
          }
        }
      })

      it("has a valid camera", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        expect(result.scene.camera.position).toHaveLength(3)
        expect(result.scene.camera.target).toHaveLength(3)
      })

      it("has at least one light", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        expect(result.scene.lights.length).toBeGreaterThan(0)
        for (const l of result.scene.lights) {
          expect(VALID_LIGHT_TYPES).toContain(l.type)
        }
      })

      it("all object IDs are unique", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        const ids = result.scene.objects.map((o) => o.id)
        const unique = new Set(ids)
        expect(unique.size).toBe(ids.length)
      })

      it("all parent references point to existing object ids", () => {
        raw = loadJson(file)
        const result = validateScene(raw)
        const ids = new Set(result.scene.objects.map((o) => o.id))
        for (const obj of result.scene.objects) {
          if (obj.parent !== undefined) {
            expect(ids.has(obj.parent)).toBe(true)
          }
        }
      })
    })
  }
})
