import { NextRequest, NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"

interface SceneCatalogItem {
  id: string
  label: string
  filePath: string
}

const PREFERRED_SCENE_ORDER = [
  "examples/solar_system_v2.json",
  "solar_system.json",
  "examples/mechanical_orrery.json",
  "examples/binary_stars.json",
]

function formatSceneLabel(id: string): string {
  const base = id.replace(/^examples\//, "").replace(/\.json$/i, "")
  return base
    .split(/[_\-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

async function listJsonFiles(dirPath: string): Promise<string[]> {
  try {
    const names = await fs.readdir(dirPath)
    return names.filter((name) => name.toLowerCase().endsWith(".json"))
  } catch {
    return []
  }
}

async function buildSceneCatalog(): Promise<SceneCatalogItem[]> {
  const repoRoot = path.resolve(process.cwd(), "..")
  const assetsDir = path.join(repoRoot, "renderer", "assets")
  const examplesDir = path.join(repoRoot, "core", "assets", "examples")

  const [assetJson, exampleJson] = await Promise.all([
    listJsonFiles(assetsDir),
    listJsonFiles(examplesDir),
  ])

  const entries: SceneCatalogItem[] = []

  for (const filename of assetJson) {
    const id = filename
    entries.push({
      id,
      label: formatSceneLabel(id),
      filePath: path.join(assetsDir, filename),
    })
  }

  for (const filename of exampleJson) {
    const id = `examples/${filename}`
    entries.push({
      id,
      label: formatSceneLabel(id),
      filePath: path.join(examplesDir, filename),
    })
  }

  entries.sort((a, b) => {
    const ai = PREFERRED_SCENE_ORDER.indexOf(a.id)
    const bi = PREFERRED_SCENE_ORDER.indexOf(b.id)
    const ap = ai === -1 ? Number.MAX_SAFE_INTEGER : ai
    const bp = bi === -1 ? Number.MAX_SAFE_INTEGER : bi
    if (ap !== bp) return ap - bp
    return a.label.localeCompare(b.label)
  })

  return entries
}

export async function GET(request: NextRequest) {
  try {
    const scenes = await buildSceneCatalog()
    const publicScenes = scenes.map(({ id, label }) => ({ id, label }))

    const selected = request.nextUrl.searchParams.get("scene")
    if (!selected) {
      return NextResponse.json({ scenes: publicScenes })
    }

    const match = scenes.find((scene) => scene.id === selected)
    if (!match) {
      return NextResponse.json(
        { error: `Unknown scene '${selected}'.`, scenes: publicScenes },
        { status: 404 }
      )
    }

    const raw = await fs.readFile(match.filePath, "utf-8")
    const parsed = JSON.parse(raw)
    return NextResponse.json({
      scenes: publicScenes,
      selected: match.id,
      scene: parsed,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load scenes"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
