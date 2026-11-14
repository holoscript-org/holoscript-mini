import { dirname } from "path"
import { fileURLToPath } from "url"

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Pin the workspace root to gui/. Without this, Next infers it from whichever
  // lockfile it finds walking up the tree — a stray package-lock.json at the repo
  // root made it treat the whole project as the workspace, so Turbopack watched
  // core/assets (158 MB of GLBs) and .venv on every change.
  turbopack: {
    root: dirname(fileURLToPath(import.meta.url)),
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
