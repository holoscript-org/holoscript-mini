import { NextRequest, NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"

const MIME: Record<string, string> = {
  glb:  "model/gltf-binary",
  gltf: "model/gltf+json",
  obj:  "model/obj",
  mtl:  "model/mtl",
  png:  "image/png",
  jpg:  "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  hdr:  "image/vnd.radiance",
  exr:  "image/x-exr",
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await params

  // Resolve strictly within core/assets — no path traversal
  const repoRoot  = path.resolve(process.cwd(), "..")
  const assetsDir = path.join(repoRoot, "core", "assets")
  const filePath  = path.resolve(assetsDir, ...segments)

  if (!filePath.startsWith(assetsDir)) {
    return new NextResponse("Forbidden", { status: 403 })
  }

  try {
    const data = await fs.readFile(filePath)
    const ext  = path.extname(filePath).slice(1).toLowerCase()
    const mime = MIME[ext] ?? "application/octet-stream"
    return new NextResponse(data, {
      headers: {
        "Content-Type":  mime,
        "Cache-Control": "public, max-age=86400, immutable",
      },
    })
  } catch {
    return new NextResponse("Not Found", { status: 404 })
  }
}
