# Member 1 — Voice → JSON → Scene Pipeline
## Complete Implementation Specification

> **Date:** 2026-05-07  
> **Schema authority:** `gui/lib/sceneFactory.ts` → `validateScene()`  
> **Renderer:** Three.js (browser, `gui/components/ThreeScene.tsx`)  
> **Your job:** Everything from voice transcript to valid scene JSON on disk.

---

## HOW TO READ THIS DOCUMENT

This spec is divided into three layers:

1. **WHY** — explains the design decision before any code
2. **WHAT** — the exact contract / interface
3. **HOW** — the full implementation

Read the WHY sections even if you plan to skim the rest.
Every important architectural decision is explained there.

---

## 0. SYSTEM OVERVIEW

### 0.1 Two distinct phases

**Phase A — One-time setup** (run before any voice commands)

```
asset_ingester.py   downloads real GLB models from Poly Pizza, Kenney,
                    glTF-Sample-Models, Poly Haven HDRIs
       ↓
kb_builder.py       scans downloaded assets → generates concept_map.json
       ↓
semantic_parser.py  (startup)  loads sentence-transformer model,
                               encodes concept corpus, stores in memory
```

**Phase B — Per voice command** (the real-time hot path)

```
Voice transcript  (string, from Whisper)
       ↓
semantic_parser.py       embed transcript → cosine similarity → concept buckets
       ↓                 ~15ms on CPU
fallback_engine.py       handle anything below similarity threshold
       ↓                 <5ms
retrieval.py             concept → asset/generator/effect
       ↓                 <5ms
scene_builder.py         deterministic JSON assembly
       ↓                 <20ms
scene_validator.py       in-process Python mirror of validateScene()
       ↓                 ~2ms
repair_loop.py           fix errors without LLM
       ↓                 <5ms
llm_bridge.py            only if repair still fails
       ↓
core/outputs/live_scene.json
```

**Total target latency (hot path, no LLM): under 60ms after transcript arrives.**

### 0.2 The two architectural decisions (read this carefully)

#### Decision 1 — Sentence-transformer embeddings instead of keyword matching

The naive approach is keyword matching: tokenize the transcript, look up each word
in `concept_map.json`. This fails constantly in practice because:

- "A towering glass structure" → no keyword "building" → nothing resolved → LLM triggered
- "Red rocky world" → "world" not in synonym_map → nothing → LLM triggered
- "Some figures standing in a grove" → "figures", "grove" not in map → LLM triggered

Every LLM fallback costs 500–2000ms. The system feels broken.

The fix: instead of matching words exactly, we measure *semantic similarity*.
We use a small language model (`all-MiniLM-L6-v2`, 80MB) that converts any text
to a 384-dimensional vector. Two texts that mean similar things end up as vectors
pointing in roughly the same direction — even if they share no words.

How it works in practice:
- At startup, for every concept in the knowledge base, we build a rich description
  string ("human: person figure character walking standing astronaut soldier alien").
  We encode all these descriptions once → get a matrix of shape (N_concepts × 384).
- At runtime, we encode the whole transcript → one vector of shape (384,).
- We compute cosine similarity between the transcript vector and every concept vector.
- Concepts with similarity above a threshold (0.35) are included.

Result: "towering glass structures" will score high similarity with the "city/building"
concept description even though the word "city" never appeared.
No synonym map needed for this. No LLM needed.

The model runs entirely on CPU with no internet connection after first download.
Inference time is ~10–20ms for a short transcript on a typical laptop CPU.

#### Decision 2 — Python port of validateScene() instead of Node subprocess

The original plan called for `validator_bridge.py` to spawn a Node.js process,
pipe JSON in via stdin, and read the result via stdout. This is the simplest way
to reuse the TypeScript validation code directly.

The problem is process spawn overhead: creating a new OS process takes 100–300ms
on Windows. For a real-time pipeline targeting <60ms total, this is a deal-breaker.

The fix: we port `validateScene()` line-by-line to Python. The TypeScript file
(`gui/lib/sceneFactory.ts`) remains the single source of truth — but we maintain
a Python mirror that applies identical rules. If the schema ever changes, both files
need updating (this is explicitly called out in the Python file with a comment).

The Python port validates in ~1–3ms with no subprocesses, no file I/O, no network.

---

## 1. SCHEMA REFERENCE

This is the complete contract extracted from `gui/lib/sceneFactory.ts`.
The Python validator in Section 5.9 implements every rule listed here.

### 1.1 Root structure

```json
{
  "name":    "optional string",
  "objects": [ ...SceneObject... ],
  "lights":  [ ...LightDef... ],
  "camera":  { ...CameraDef... }
}
```

`objects` is required and must be non-empty — absence triggers a **fatal** error.
`lights` and `camera` are optional — the renderer uses defaults when absent.

### 1.2 SceneObject — complete field list

```json
{
  "id":       "unique_snake_case_string",         REQUIRED, non-empty
  "type":     "primitive" | "mesh",               REQUIRED

  "geometry": {                                   REQUIRED if type="primitive"
    "type": "sphere"|"box"|"cylinder"|"plane"|"ring"|"capsule"|"torus",
    "radius":       1.0,       sphere, capsule, torus   (must be > 0)
    "length":       1.0,       capsule only             (must be > 0)
    "tube":         0.2,       torus only               (must be > 0)
    "innerRadius":  0.5,       ring only                (must be > 0)
    "outerRadius":  1.0,       ring only, must > innerRadius
    "thetaSegments": 64,       ring only, must be >= 3
    "width":        1.0,       box, plane               (must be > 0)
    "height":       1.0,       box, cylinder
    "depth":        1.0,       box
    "from": [x, y, z],        cylinder ONLY — use this to orient cylinders
    "to":   [x, y, z]         cylinder ONLY — do NOT use rotation for cylinders
  },

  "model": "/assets/meshes/category/file.glb",   REQUIRED if type="mesh"

  "position": [x, y, z],     REQUIRED always — 3 finite numbers
  "rotation": [rx, ry, rz],  optional — Euler DEGREES (not radians)
  "scale":    [sx, sy, sz],  optional — include it anyway, use [1,1,1] default

  "parent": "another_object_id",   optional
                                    position/rotation/scale become LOCAL to parent
                                    parent id MUST exist in same scene
                                    no self-reference, no circular chains

  "material": {               REQUIRED always
    "type":              "standard",   REQUIRED — always exactly this string
    "color":             "#rrggbb",    REQUIRED — hex ONLY, not [r,g,b] floats
    "roughness":         0.7,          REQUIRED — 0.0 to 1.0
    "metalness":         0.1,          REQUIRED — 0.0 to 1.0
    "opacity":           0.5,          optional — if < 1.0, you MUST set transparent: true
    "transparent":       true,         optional boolean
    "emissive":          "#ff8800",    optional hex
    "emissiveIntensity": 0.9,          optional, >= 0
    "map":               "/textures/diffuse.jpg",   optional string path
    "normalMap":         "/textures/normal.jpg",    optional string path
    "roughnessMap":      "/textures/rough.jpg",     optional string path
    "metalnessMap":      "/textures/metal.jpg",     optional string path
    "emissiveMap":       "/textures/emit.jpg"       optional string path
  },

  "label": "optional display string",

  "animation": {             optional — omitting is fine; defaults to {type:"none"}
    "type":       "none" | "orbit" | "spin",  REQUIRED if animation block present
    "center":     [x, y, z],    orbit: world-space center point
    "center_ref": "object_id",  orbit: follow another object's live position
    "axis":       [0, 1, 0],    orbit + spin: rotation axis
    "speed":      0.5,          radians/sec
    "phase":      0.0           orbit: starting angle offset in radians
  }
}
```

### 1.3 LightDef

```json
{
  "type":       "ambient" | "directional" | "point" | "spot",  REQUIRED
  "intensity":  1.2,          REQUIRED — must be >= 0
  "color":      "#ffffff",    optional — default "#ffffff"
  "position":   [x, y, z],   optional — not used for ambient
  "castShadow": true          optional boolean
}
```

### 1.4 CameraDef

```json
{
  "position": [x, y, z],   REQUIRED if camera block is present
  "target":   [x, y, z],   REQUIRED if camera block is present
  "fov":      65            optional — valid range 1 to 179
}
```

### 1.5 Fatal vs non-fatal errors

**Fatal** — entire scene is rejected, renderer shows an error:
- Root is not a JSON object
- `objects` field is missing
- `objects` array is empty

**Non-fatal** — that individual object is skipped, valid ones still render:
- Any field-level error on a specific object

### 1.6 Hard rules summary

| Rule | Consequence of breaking it |
|------|---------------------------|
| `primitive` must have `geometry` | object skipped |
| `mesh` must have `model` | object skipped |
| `material.color` must be `"#rrggbb"` hex | object skipped |
| `material.type` must be exactly `"standard"` | object skipped |
| `animation.type` must be `none\|orbit\|spin` | object skipped |
| `geometry.type` must be one of 7 valid types | object skipped |
| `opacity < 1` without `transparent: true` | renders incorrectly |
| `parent` references non-existent id | non-fatal error logged |
| `parent` creates cycle | non-fatal error logged |
| `objects` array is empty | fatal — nothing renders |

---

## 2. WHAT ALREADY EXISTS — DO NOT REBUILD THESE

| File | Status | Action |
|------|--------|--------|
| `voice/recorder.py` | ✅ Works | Keep as-is |
| `voice/transcriber.py` | ✅ Works | Keep as-is |
| `voice/command_parser.py` | ⚠️ Trivial | Route to semantic_parser instead |
| `llm/groq_client.py` | ⚠️ Fix | FALLBACK_SCENE uses wrong schema |
| `llm/prompt_templates.py` | ❌ Broken import | Rewrite (Section 6.2) |
| `llm/scene_schema.py` | ❌ Wrong schema | Rewrite (Section 6.1) |
| `llm/ollama_client.py` | ✅ Works | Reuse in llm_bridge.py |
| `core/assets/examples/*.json` | ✅ Reference | 16 valid scenes — read them to learn the schema |
| `core/outputs/` | ✅ Exists | Write `live_scene.json` here |
| `gui/lib/sceneFactory.ts` | ✅ Schema authority | Never modify |

---

## 3. COMPLETE FILE STRUCTURE TO CREATE

```
pipeline/
  __init__.py
  asset_ingester.py       downloads real GLBs from Poly Pizza / Kenney / glTF-Sample
  kb_builder.py           scans downloaded assets → builds concept_map.json
  semantic_parser.py      sentence-transformer intent extraction (replaces keyword matching)
  retrieval.py            concept → asset/generator/effect lookup
  generators.py           scatter, grid, orbit_cluster — all deterministic
  effects.py              orbit/spin animation templates
  scene_builder.py        assembles final scene dict
  fallback_engine.py      handles low-similarity concepts
  llm_bridge.py           last-resort LLM with schema-locked prompt
  scene_validator.py      Python port of validateScene() — NO subprocess
  repair_loop.py          structural fixes, no LLM
  pipeline_runner.py      main entry point

  knowledge_base/
    synonym_map.json      hand-written linguistics (word → category)
    concept_map.json      GENERATED by kb_builder.py — never hand-edit
    concept_descriptions.json   GENERATED by kb_builder.py — embedding corpus
    assets/               GENERATED sidecar JSONs — one per downloaded GLB

core/assets/
  meshes/
    humans/
    vehicles/
    buildings/
    trees/
    planets/
    satellites/
    abstract/
  hdri/                   Poly Haven environment maps
```

**Files NOT in this list** (removed from earlier version):
- ~~`validate_runner.js`~~ — replaced by `scene_validator.py`
- ~~`validator_bridge.py`~~ — replaced by `scene_validator.py`
- ~~`intent_parser.py`~~ — replaced by `semantic_parser.py`

---

## 4. ASSET INGESTION SYSTEM

### 4.1 Why real assets matter

The system could use only primitive geometry (spheres, boxes, etc.) but it would look
like a 1990s screensaver. Real GLB models from Kenney and Poly Pizza make the scene
look like an actual 3D scene — and they're free, small (under 5MB each), and ship
in the format Three.js natively loads.

More importantly: the knowledge base is built FROM the downloaded assets.
We do not hardcode what concepts exist. The concepts that the system can recognize
are exactly those whose assets were actually downloaded. If you download a "dragon.glb"
with tags ["dragon", "creature", "fantasy"], the system automatically learns that
"dragon" is a valid concept without you writing a single line of code.

### 4.2 Asset sources

| Source | What we get | API / Access |
|--------|------------|--------------|
| **Poly Pizza** | Single GLB models, many categories | REST API at `https://api.poly.pizza/v2/models?q=<query>` |
| **Kenney** | Consistent style, free ZIP packs | Direct ZIP download, no login |
| **glTF-Sample-Models** | Reference models, guaranteed compatible | GitHub raw URLs |
| **Poly Haven** | HDRI environment maps (.hdr) | REST API at `https://api.polyhaven.com/` |

### 4.3 `pipeline/asset_ingester.py`

```python
"""
Downloads real 3D assets from public free sources into core/assets/meshes/.
Also writes sidecar JSON files to pipeline/knowledge_base/assets/.

Run once to seed the knowledge base:
    python -m pipeline.asset_ingester

Run again any time to pick up new assets.
Already-downloaded files are skipped (checks by filename).

Sources:
  Poly Pizza  https://api.poly.pizza/v2/
  Kenney      https://kenney.nl (free ZIP packs, CC0 license)
  glTF-Sample https://github.com/KhronosGroup/glTF-Sample-Models
  Poly Haven  https://api.polyhaven.com/ (HDRIs only)
"""
import io
import json
import time
import zipfile
from pathlib import Path
import requests

_ROOT   = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_HDRI   = _ROOT / "core" / "assets" / "hdri"
_KB     = Path(__file__).parent / "knowledge_base" / "assets"

MAX_BYTES = 10 * 1024 * 1024   # 10 MB hard limit per file

# ── What to fetch from Poly Pizza ────────────────────────────────────────────
# Format: { "folder_name": ["search query 1", "search query 2", ...] }
# Poly Pizza returns GLB files from a public creative-commons library.
# We search multiple queries per category to get diverse results.
POLYPIZZA_QUERIES = {
    "humans":    ["person", "character", "human figure"],
    "vehicles":  ["car", "vehicle", "rocket", "spaceship"],
    "buildings": ["building", "skyscraper", "house", "tower"],
    "trees":     ["tree", "palm tree", "pine tree", "plant"],
    "planets":   ["planet", "sphere", "moon", "asteroid"],
    "satellites":["satellite", "spacecraft", "space station"],
    "abstract":  ["crystal", "gem", "orb", "ring"],
}
POLYPIZZA_PER_QUERY = 3   # max models to download per query

# ── Kenney free packs ─────────────────────────────────────────────────────────
# These are CC0 (public domain). Each is a ZIP containing GLBs.
# Check https://kenney.nl/assets for current URLs — these may change.
KENNEY_PACKS = {
    "vehicles":  "https://kenney.nl/content/assets/Car-Kit.zip",
    "buildings": "https://kenney.nl/content/assets/City-Kit-Roads.zip",
    "trees":     "https://kenney.nl/content/assets/Nature-Kit.zip",
    "abstract":  "https://kenney.nl/content/assets/Mini-Dungeon.zip",
}
KENNEY_MAX_PER_PACK = 6

# ── glTF-Sample-Models ────────────────────────────────────────────────────────
# Specific well-known models from the Khronos reference set.
# License: CC-BY 4.0 for most models.
GLTF_SAMPLES = {
    "humans": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb",
    ],
    "vehicles": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/ToyCar/glTF-Binary/ToyCar.glb",
    ],
    "abstract": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Avocado/glTF-Binary/Avocado.glb",
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Lantern/glTF-Binary/Lantern.glb",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> bool:
    """Stream-download a file. Returns True on success. Enforces MAX_BYTES."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                size += len(chunk)
                if size > MAX_BYTES:
                    dest.unlink(missing_ok=True)
                    print(f"  skip {dest.name}: over {MAX_BYTES//1024//1024}MB limit")
                    return False
                f.write(chunk)
        print(f"  ✓ {dest.name}  ({size // 1024}KB)")
        return True
    except Exception as e:
        print(f"  ✗ {dest.name}: {e}")
        dest.unlink(missing_ok=True)
        return False


def _write_sidecar(glb_path: Path, category: str, tags: list[str],
                   author: str = "", license_str: str = "") -> None:
    """
    Write a small JSON sidecar next to nothing — stored in knowledge_base/assets/.
    This sidecar is what kb_builder.py reads to build concept_map.json.
    Format must stay stable — kb_builder depends on it.
    """
    _KB.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "id":       glb_path.stem,
        "src":      f"/assets/meshes/{category}/{glb_path.name}",
        "category": category,
        "tags":     list(dict.fromkeys(t.lower().strip() for t in tags if t.strip())),
        "author":   author,
        "license":  license_str,
    }
    with open(_KB / f"{glb_path.stem}.json", "w") as f:
        json.dump(sidecar, f, indent=2)


# ── Poly Pizza fetcher ────────────────────────────────────────────────────────

def fetch_polypizza(max_per_query: int = POLYPIZZA_PER_QUERY) -> None:
    print("\n── Poly Pizza ──")
    for category, queries in POLYPIZZA_QUERIES.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for query in queries:
            try:
                resp = requests.get(
                    "https://api.poly.pizza/v2/models",
                    params={"q": query, "limit": max_per_query},
                    timeout=10,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
            except Exception as e:
                print(f"  Poly Pizza search '{query}': {e}")
                continue

            for item in results[:max_per_query]:
                dl_url = item.get("Download") or item.get("download", "")
                if not dl_url.lower().endswith(".glb"):
                    continue
                safe = item.get("Title", query).lower().replace(" ", "_")[:40] + ".glb"
                dest = dest_dir / safe
                if dest.exists():
                    continue
                tags = item.get("Tags", []) + [query, category.rstrip("s")]
                author = item.get("Creator", {}).get("Username", "unknown")
                license_str = item.get("License", "CC-BY")
                if _download_file(dl_url, dest):
                    _write_sidecar(dest, category, tags, author, license_str)
                time.sleep(0.3)


# ── Kenney fetcher ────────────────────────────────────────────────────────────

def fetch_kenney() -> None:
    print("\n── Kenney ──")
    for category, zip_url in KENNEY_PACKS.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"  Downloading {category} pack...")
            resp = requests.get(zip_url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                glb_entries = [
                    n for n in zf.namelist()
                    if n.lower().endswith(".glb") and "__MACOSX" not in n
                ]
                for entry in glb_entries[:KENNEY_MAX_PER_PACK]:
                    fname = Path(entry).name
                    dest  = dest_dir / fname
                    if dest.exists():
                        continue
                    data = zf.read(entry)
                    if len(data) > MAX_BYTES:
                        continue
                    dest.write_bytes(data)
                    stem = dest.stem.lower().replace("-", " ").replace("_", " ")
                    tags = [t for t in stem.split() if not t.isdigit()]
                    tags += [category, category.rstrip("s")]
                    _write_sidecar(dest, category, tags, "Kenney", "CC0")
                    print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ Kenney {category}: {e}")


# ── glTF-Sample fetcher ────────────────────────────────────────────────────────

def fetch_gltf_samples() -> None:
    print("\n── glTF-Sample-Models ──")
    for category, urls in GLTF_SAMPLES.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            fname = Path(url).name
            dest  = dest_dir / fname
            if dest.exists():
                continue
            tags = [Path(url).stem.lower(), category, category.rstrip("s")]
            if _download_file(url, dest):
                _write_sidecar(dest, category, tags, "Khronos Group", "CC-BY 4.0")


# ── Poly Haven HDRI fetcher ───────────────────────────────────────────────────

def fetch_polyhaven_hdri(max_hdris: int = 3) -> None:
    """
    Downloads a small set of HDRI environment maps for realistic scene lighting.
    These are .hdr files used by Three.js for image-based lighting (IBL).
    We fetch 1k resolution only — sufficient for real-time rendering.
    """
    print("\n── Poly Haven HDRIs ──")
    _HDRI.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get("https://api.polyhaven.com/assets?t=hdris", timeout=10)
        resp.raise_for_status()
        slugs = list(resp.json().keys())[:max_hdris * 3]
    except Exception as e:
        print(f"  ✗ Poly Haven listing: {e}")
        return

    count = 0
    for slug in slugs:
        if count >= max_hdris:
            break
        out = _HDRI / f"{slug}_1k.hdr"
        if out.exists():
            count += 1
            continue
        try:
            files = requests.get(
                f"https://api.polyhaven.com/files/{slug}", timeout=10
            ).json()
            url = files.get("hdri", {}).get("1k", {}).get("hdr", {}).get("url")
            if url and _download_file(url, out):
                count += 1
                time.sleep(0.5)
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    print("=== Asset Ingestion ===")
    _MESHES.mkdir(parents=True, exist_ok=True)
    _KB.mkdir(parents=True, exist_ok=True)
    fetch_polypizza()
    fetch_kenney()
    fetch_gltf_samples()
    fetch_polyhaven_hdri()
    total = sum(1 for _ in _MESHES.rglob("*.glb"))
    sidecars = sum(1 for _ in _KB.glob("*.json"))
    print(f"\n=== Done: {total} GLBs downloaded, {sidecars} sidecars written ===")
    print("Next step: python -m pipeline.kb_builder")


if __name__ == "__main__":
    run_ingestion()
```

---

### 4.4 `pipeline/kb_builder.py`

```python
"""
Builds two generated files from ingested assets:

  pipeline/knowledge_base/concept_map.json
      Maps concept names → asset, generator, or effect handlers.
      This is what the semantic parser and retrieval system use.

  pipeline/knowledge_base/concept_descriptions.json
      Maps concept names → rich text description for embedding.
      This is what the sentence-transformer uses at startup.

Both files are GENERATED. Never edit them by hand — run this script instead.

Run after asset_ingester.py:
    python -m pipeline.kb_builder
"""
import json
from pathlib import Path

_ROOT   = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_KB     = Path(__file__).parent / "knowledge_base"
_ASSETS = _KB / "assets"

_OUT_MAP  = _KB / "concept_map.json"
_OUT_DESC = _KB / "concept_descriptions.json"

# ── Category → pipeline type ──────────────────────────────────────────────────
# Maps asset folder names to the concept type used in the pipeline.
# "object" = a single placed instance
# "structure" = a generator that places many instances
# "system" = a generator for relational arrangements (orbits etc.)
# "effect" = an animation modifier
CATEGORY_TYPE = {
    "humans":     "object",
    "vehicles":   "object",
    "buildings":  "object",
    "trees":      "object",
    "planets":    "object",
    "satellites": "object",
    "abstract":   "object",
}

# ── Structure and system generators ──────────────────────────────────────────
# These are added to concept_map regardless of which assets are present.
# asset_category points to whichever folder they should draw instances from.
# The actual asset is resolved at runtime by retrieval.py.
STATIC_ENTRIES = {
    "forest": {
        "type": "structure", "generator": "scatter",
        "asset_category": "trees",
        "count": [8, 16], "radius": 12,
    },
    "city": {
        "type": "structure", "generator": "grid",
        "asset_category": "buildings",
        "count": [9, 16], "spacing": 3.5,
    },
    "crowd": {
        "type": "structure", "generator": "scatter",
        "asset_category": "humans",
        "count": [6, 12], "radius": 8,
    },
    "fleet": {
        "type": "structure", "generator": "scatter",
        "asset_category": "vehicles",
        "count": [4, 8], "radius": 10,
    },
    "solar_system": {
        "type": "system", "generator": "orbit_cluster",
        "central_category": "planets", "satellite_category": "planets",
        "count": [4, 8],
    },
    "atom": {
        "type": "system", "generator": "orbit_cluster",
        "central_category": "abstract", "satellite_category": "abstract",
        "count": [3, 6],
    },
    "orbit":   {"type": "effect", "handler": "anim_orbit"},
    "spin":    {"type": "effect", "handler": "anim_spin"},
    "rotate":  {"type": "effect", "handler": "anim_spin"},
    "revolve": {"type": "effect", "handler": "anim_orbit"},
    "float":   {"type": "effect", "handler": "anim_spin"},
    "hover":   {"type": "effect", "handler": "anim_spin"},
}

# ── Descriptions for embedding ────────────────────────────────────────────────
# These are human-readable descriptions used to build the sentence-transformer
# corpus. The richer the description, the better the similarity matching.
# Key insight: include diverse synonyms and related words so that paraphrases
# in user speech ("towering structures") match the right concept ("city").
# These are SEEDED descriptions. kb_builder enriches them with asset tags
# and synonym_map entries automatically — you don't need to be exhaustive here.
STATIC_DESCRIPTIONS = {
    "forest":       "forest trees woodland grove jungle dense trees plants nature scattered",
    "city":         "city skyline buildings skyscrapers urban downtown metropolis towers structures",
    "crowd":        "crowd group people standing figures gathering many humans",
    "fleet":        "fleet group vehicles cars ships aircraft formation",
    "solar_system": "solar system planets orbiting star sun celestial bodies",
    "atom":         "atom nucleus electrons orbiting particles quantum",
    "orbit":        "orbit orbiting flying around circling revolving moving ellipse",
    "spin":         "spin spinning rotating turning whirling swirling rotation",
    "rotate":       "rotating turning spinning revolving angle",
    "revolve":      "revolving orbiting circling moving around",
    "float":        "floating hovering drifting gentle movement",
    "hover":        "hovering floating staying still gentle bob",
    "humans":       "human person figure character walking standing people",
    "vehicles":     "vehicle car truck rocket ship aircraft spacecraft transport",
    "buildings":    "building house structure skyscraper tower architecture",
    "trees":        "tree forest plant nature palm pine oak vegetation",
    "planets":      "planet sphere world globe rocky terrain celestial body moon mars earth",
    "satellites":   "satellite probe station spacecraft orbiting device",
    "abstract":     "crystal gem ring torus abstract shape decorative object",
}


def _load_sidecar(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _reverse_synonym_map() -> dict[str, list[str]]:
    """Build a map from concept → [all synonyms that map to it]."""
    syn_path = _KB / "synonym_map.json"
    if not syn_path.exists():
        return {}
    syn = json.loads(syn_path.read_text())
    reverse: dict[str, list[str]] = {}
    for word, targets in syn.items():
        if word.startswith("_"):
            continue
        for target in targets:
            reverse.setdefault(target, []).append(word)
    return reverse


def build() -> tuple[dict, dict]:
    """Returns (concept_map, concept_descriptions)."""
    concept_map: dict = {}
    descriptions: dict = {}
    reverse_syn = _reverse_synonym_map()

    # ── Walk every sidecar and register object concepts ──────────────────
    if _ASSETS.exists():
        for sidecar_path in sorted(_ASSETS.glob("*.json")):
            meta = _load_sidecar(sidecar_path)
            if not meta:
                continue
            category     = meta.get("category", "abstract")
            concept_type = CATEGORY_TYPE.get(category, "object")
            tags         = meta.get("tags", [])

            # Register one concept entry per tag
            for tag in tags:
                tag = tag.lower().strip()
                if not tag or tag.isdigit() or len(tag) < 2:
                    continue
                if tag not in concept_map:
                    concept_map[tag] = {
                        "type":       concept_type,
                        "asset_id":   meta["id"],
                        "asset_src":  meta["src"],
                        "category":   category,
                    }

            # Also register the category name itself (e.g. "trees" → object)
            cat_key = category.rstrip("s")
            if cat_key not in concept_map:
                concept_map[cat_key] = {
                    "type":      concept_type,
                    "asset_id":  meta["id"],
                    "asset_src": meta["src"],
                    "category":  category,
                }

    # ── Add static entries ────────────────────────────────────────────────
    for key, entry in STATIC_ENTRIES.items():
        if key not in concept_map:
            concept_map[key] = entry

    # ── Build descriptions ────────────────────────────────────────────────
    # For each concept, combine:
    #   1. Static seed description (if any)
    #   2. All synonym_map keys that point to this concept
    #   3. Asset tags from the chosen sidecar
    #   4. The concept name itself and its category
    for concept, entry in concept_map.items():
        parts = [concept]

        # Seed description
        base_desc = STATIC_DESCRIPTIONS.get(concept, "")
        if not base_desc:
            base_desc = STATIC_DESCRIPTIONS.get(entry.get("category", ""), "")
        if base_desc:
            parts.append(base_desc)

        # Synonyms that map to this concept
        for syn in reverse_syn.get(concept, []):
            parts.append(syn)

        # Asset tags
        asset_id = entry.get("asset_id")
        if asset_id:
            sid = _load_sidecar(_ASSETS / f"{asset_id}.json")
            if sid:
                parts.extend(sid.get("tags", []))

        descriptions[concept] = " ".join(dict.fromkeys(
            w.lower() for w in " ".join(parts).split() if len(w) > 1
        ))

    return concept_map, descriptions


def main() -> None:
    print("Building knowledge base from ingested assets...")
    concept_map, descriptions = build()
    _KB.mkdir(parents=True, exist_ok=True)
    _OUT_MAP.write_text(json.dumps(concept_map, indent=2))
    _OUT_DESC.write_text(json.dumps(descriptions, indent=2))
    print(f"  concept_map.json      → {len(concept_map)} entries")
    print(f"  concept_descriptions  → {len(descriptions)} descriptions")
    by_type: dict[str, int] = {}
    for v in concept_map.values():
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n}")
    print("Next step: python -m pipeline.pipeline_runner \"test prompt\"")


if __name__ == "__main__":
    main()
```

---

### 4.5 `pipeline/knowledge_base/synonym_map.json`

This is the one file you write by hand. It contains only linguistics —
word → category mappings. It does NOT contain colors, geometry values,
or any asset data. Those come from the ingested assets.

The semantic parser uses this as a SUPPLEMENT to embedding similarity.
If a word scores below the embedding threshold, the synonym_map is checked
as a second pass before giving up and calling the LLM.

```json
{
  "_comment": "word → concept category. Categories must match keys in concept_map.json or folder names. Hand-written.",

  "alien":      ["human"],   "astronaut":   ["human"],
  "soldier":    ["human"],   "dragon":      ["human"],
  "monster":    ["human"],   "creature":    ["human"],
  "person":     ["human"],   "robot":       ["human"],
  "android":    ["human"],   "figure":      ["human"],

  "spaceship":  ["vehicle"], "rocket":      ["vehicle"],
  "ufo":        ["vehicle"], "aircraft":    ["vehicle"],
  "jet":        ["vehicle"], "ship":        ["vehicle"],
  "shuttle":    ["vehicle"], "fighter":     ["vehicle"],

  "castle":     ["building"], "tower":      ["building"],
  "house":      ["building"], "skyscraper": ["building"],
  "pyramid":    ["building"], "temple":     ["building"],
  "structure":  ["building"],

  "mars":       ["planet"],  "earth":       ["planet"],
  "jupiter":    ["planet"],  "saturn":      ["planet"],
  "venus":      ["planet"],  "neptune":     ["planet"],
  "moon":       ["planet"],  "world":       ["planet"],
  "globe":      ["planet"],  "sun":         ["planet"],
  "star":       ["planet"],  "asteroid":    ["planet"],
  "comet":      ["planet"],

  "woods":      ["forest"],  "jungle":      ["forest"],
  "trees":      ["forest"],  "grove":       ["forest"],
  "park":       ["forest"],

  "town":       ["city"],    "metropolis":  ["city"],
  "skyline":    ["city"],    "downtown":    ["city"],
  "village":    ["city"],

  "flying":     ["orbit"],   "orbiting":    ["orbit"],
  "circling":   ["orbit"],   "revolving":   ["orbit"],
  "spinning":   ["spin"],    "rotating":    ["spin"],
  "turning":    ["spin"],    "swirling":    ["spin"],

  "probe":      ["satellite"], "station":   ["satellite"],
  "telescope":  ["satellite"],

  "dna":        ["abstract"], "helix":      ["abstract"],
  "crystal":    ["abstract"], "molecule":   ["atom"],
  "nucleus":    ["abstract"], "electron":   ["abstract"]
}
```

---

## 5. MODULE IMPLEMENTATIONS

### 5.1 `pipeline/__init__.py`

```python
# empty
```

---

### 5.2 `pipeline/semantic_parser.py` ← REPLACES keyword matching

#### Why this module exists

This is the heart of the real-time classification system.
Read the WHY in Section 0.2 before reading the implementation.

#### How the corpus is built

At startup, this module loads `concept_descriptions.json` (generated by kb_builder).
Each entry looks like:

```
"planet" → "planet world globe rocky terrain celestial body moon mars earth
             mars earth venus neptune satellite probe asteroid comet"
```

The sentence-transformer model encodes each description into a 384-dimensional
vector and stores the matrix in memory. This only happens once at startup (~2 seconds).

#### How classification works at runtime

Given the transcript `"red rocky world orbiting a massive star"`:

1. The whole transcript is encoded into a single 384-dim vector (~15ms).
2. Cosine similarity is computed between this vector and every concept in the corpus.
3. Concepts scoring above `SIMILARITY_THRESHOLD` (0.35) are selected.
4. Each selected concept is looked up in `concept_map.json` to get its type
   (object / structure / system / effect) and placed in the appropriate bucket.

The result is the same `{"objects":[], "structures":[], "systems":[], "effects":[]}`
dict that the old keyword parser produced — so everything downstream is unchanged.

#### Why 0.35 as the threshold

- Score 0.0 = completely unrelated
- Score 0.5+ = very clearly the same concept
- Score 0.35 = semantically related — good threshold for finding matches
  without picking up noise

You can tune this. If the system is picking up wrong concepts → raise to 0.40.
If the system is missing obvious ones → lower to 0.30.

```python
"""
Semantic intent parser using sentence-transformer embeddings.

Replaces keyword/synonym matching with cosine similarity over a pre-embedded
concept corpus. Handles paraphrases like "towering glass structures" → city
without any synonym map entry for those words.

Dependencies:
    pip install sentence-transformers

Model:
    all-MiniLM-L6-v2 (~80MB, downloads on first run, CPU-capable)
    Inference: ~10–20ms per transcript on CPU

Startup cost:
    ~1–3 seconds the first time (model load + corpus encode).
    After that, the encoded matrix stays in memory — per-call cost is ~15ms.
"""
import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer

_KB   = Path(__file__).parent / "knowledge_base"
_MAP  = _KB / "concept_map.json"
_DESC = _KB / "concept_descriptions.json"

# Similarity score below which a concept is NOT included.
# Tune this if you're getting too many false positives (raise it)
# or too many misses (lower it).
SIMILARITY_THRESHOLD = 0.35

# Model name — all-MiniLM-L6-v2 is 80MB, fast on CPU, good quality.
# Do not change this without testing — the threshold may need re-tuning.
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticParser:
    """
    Loads model + corpus once at construction. Call parse_intent() repeatedly.

    Usage:
        parser = SemanticParser()          # slow (1-3s) — do once at startup
        intent = parser.parse_intent(text) # fast (~15ms) — call per transcript
    """

    def __init__(self) -> None:
        if not _MAP.exists():
            raise RuntimeError(
                "concept_map.json not found. Run: python -m pipeline.kb_builder"
            )
        if not _DESC.exists():
            raise RuntimeError(
                "concept_descriptions.json not found. Run: python -m pipeline.kb_builder"
            )

        print("[semantic_parser] Loading sentence-transformer model...")
        self._model = SentenceTransformer(MODEL_NAME)

        with open(_MAP)  as f: self._concept_map: dict  = json.load(f)
        with open(_DESC) as f: self._descriptions: dict = json.load(f)

        # Build corpus: list of (concept_name, description) in a stable order
        self._concepts: list[str]  = list(self._descriptions.keys())
        corpus: list[str]          = [self._descriptions[c] for c in self._concepts]

        print(f"[semantic_parser] Encoding {len(corpus)} concept descriptions...")
        # Shape: (N_concepts, 384)
        self._corpus_matrix: np.ndarray = self._model.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )
        print("[semantic_parser] Ready.")

    def parse_intent(self, text: str) -> dict:
        """
        Encode transcript → cosine similarity → concept buckets.

        Returns:
            {
                "objects":    ["planet", "satellite", ...],
                "structures": ["city", ...],
                "systems":    [],
                "effects":    ["orbit", ...]
            }

        Example:
            parse_intent("red rocky world with orbiting satellites")
            → {"objects": ["planet", "satellite"], "structures": [],
               "systems": [], "effects": ["orbit"]}
        """
        if not text or not text.strip():
            return {"objects": [], "structures": [], "systems": [], "effects": []}

        # Encode transcript — shape (384,)
        query_vec: np.ndarray = self._model.encode(
            text.lower().strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Cosine similarity — since both are L2-normalized, this is just a dot product.
        # Shape: (N_concepts,)
        scores: np.ndarray = self._corpus_matrix @ query_vec

        objects, structures, systems, effects = [], [], [], []

        for idx, concept in enumerate(self._concepts):
            if scores[idx] < SIMILARITY_THRESHOLD:
                continue
            entry = self._concept_map.get(concept)
            if not entry:
                continue
            t = entry.get("type")
            target = {
                "object":    objects,
                "structure": structures,
                "system":    systems,
                "effect":    effects,
            }.get(t)
            if target is not None and concept not in target:
                target.append(concept)

        # Sort each bucket by score descending so the best matches come first.
        def sort_key(c: str) -> float:
            idx = self._concepts.index(c)
            return -float(scores[idx])

        return {
            "objects":    sorted(objects,    key=sort_key),
            "structures": sorted(structures, key=sort_key),
            "systems":    sorted(systems,    key=sort_key),
            "effects":    sorted(effects,    key=sort_key),
        }

    def top_matches(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        """Debug helper — returns top N concepts with their similarity scores."""
        query_vec = self._model.encode(
            text.lower().strip(), convert_to_numpy=True, normalize_embeddings=True
        )
        scores = self._corpus_matrix @ query_vec
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._concepts[i], float(scores[i])) for i in top_idx]


# ── Module-level singleton ────────────────────────────────────────────────────
# The pipeline_runner creates this once and passes it down.
# Do NOT instantiate SemanticParser multiple times — model load is expensive.

_parser_instance: SemanticParser | None = None

def get_parser() -> SemanticParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = SemanticParser()
    return _parser_instance


def parse_intent(text: str) -> dict:
    """Convenience function for one-off use."""
    return get_parser().parse_intent(text)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "red rocky world with orbiting satellites"
    parser = SemanticParser()

    print(f"\nTranscript: {repr(text)}")
    print("\nTop 10 concept matches:")
    for concept, score in parser.top_matches(text, 10):
        bar = "█" * int(score * 30)
        print(f"  {score:.3f} {bar:<30} {concept}")

    print("\nParsed intent:")
    print(json.dumps(parser.parse_intent(text), indent=2))
```

---

### 5.3 `pipeline/retrieval.py`

```python
"""
Maps resolved intent → concrete component list.

Reads concept_map.json (generated). Checks whether GLB files actually exist
on disk before committing to mesh type — gracefully falls back to primitives.
"""
import json
from pathlib import Path

_ROOT    = Path(__file__).resolve().parents[1]
_MESHES  = _ROOT / "core" / "assets" / "meshes"
_KB      = Path(__file__).parent / "knowledge_base"
_ASSETS  = _KB / "assets"

with open(_KB / "concept_map.json") as f:
    CONCEPT_MAP: dict = json.load(f)


def _load_sidecar(asset_id: str) -> dict | None:
    path = _ASSETS / f"{asset_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _first_asset_in_category(category: str) -> dict | None:
    """Return sidecar for the first GLB found in a category folder."""
    cat_dir = _MESHES / category
    if not cat_dir.exists():
        return None
    for glb in sorted(cat_dir.glob("*.glb")):
        sid = _ASSETS / f"{glb.stem}.json"
        if sid.exists():
            return json.loads(sid.read_text())
    return None


def _glb_on_disk(src: str) -> bool:
    if not src:
        return False
    rel = src.lstrip("/").replace("assets/meshes/", "")
    return (_ROOT / "core" / "assets" / "meshes" / rel).exists()


def retrieve(intent: dict) -> dict:
    """intent = output of semantic_parser.parse_intent()"""
    assets, generators, effects = [], [], []

    for concept in intent.get("objects", []):
        entry     = CONCEPT_MAP.get(concept, {})
        asset_id  = entry.get("asset_id")
        asset_src = entry.get("asset_src", "")
        sidecar   = _load_sidecar(asset_id) if asset_id else None
        if not _glb_on_disk(asset_src):
            asset_src, asset_id = None, None
        assets.append({
            "concept": concept, "asset_id": asset_id, "asset_src": asset_src,
            "sidecar": sidecar, "category": entry.get("category", "abstract"),
            "is_generator": False,
        })

    for concept in intent.get("structures", []):
        entry = CONCEPT_MAP.get(concept, {})
        cat   = entry.get("asset_category") or entry.get("category")
        sid   = _first_asset_in_category(cat) if cat else None
        src   = (sid or {}).get("src")
        assets.append({
            "concept": concept,
            "asset_id": (sid or {}).get("id"),
            "asset_src": src if _glb_on_disk(src or "") else None,
            "sidecar": sid, "category": cat,
            "is_generator": True,
            "generator_type":    entry.get("generator", "scatter"),
            "generator_count":   entry.get("count", [4, 8]),
            "generator_radius":  entry.get("radius", 10),
            "generator_spacing": entry.get("spacing", 3.0),
        })

    for concept in intent.get("systems", []):
        entry = CONCEPT_MAP.get(concept, {})
        cc    = entry.get("central_category",   "abstract")
        sc    = entry.get("satellite_category", "abstract")
        generators.append({
            "concept": concept,
            "generator_type":    entry.get("generator", "orbit_cluster"),
            "central_sidecar":   _first_asset_in_category(cc),
            "satellite_sidecar": _first_asset_in_category(sc),
            "count": entry.get("count", [4, 8]),
        })

    for concept in intent.get("effects", []):
        entry = CONCEPT_MAP.get(concept, {})
        effects.append({"concept": concept, "handler": entry.get("handler", "anim_orbit")})

    return {"assets": assets, "generators": generators, "effects": effects}
```

---

### 5.4 `pipeline/generators.py`

```python
"""
Deterministic generators. All use seeded random — same input = same output.
No hardcoded colors or geometry values. Appearance derived from asset sidecar.
"""
import math, random

_CATEGORY_FALLBACK = {
    "humans":     {"geom": "capsule",  "params": {"radius": 0.3, "length": 1.0}},
    "vehicles":   {"geom": "box",      "params": {"width": 2.0, "height": 0.8, "depth": 3.5}},
    "buildings":  {"geom": "box",      "params": {"width": 1.2, "height": 3.0, "depth": 1.2}},
    "trees":      {"geom": "cylinder", "params": {"from": [0,0,0], "to": [0,2.5,0]}},
    "planets":    {"geom": "sphere",   "params": {"radius": 1.5}},
    "satellites": {"geom": "box",      "params": {"width": 0.8, "height": 0.4, "depth": 0.8}},
    "abstract":   {"geom": "torus",    "params": {"radius": 1.0, "tube": 0.3}},
}

_NEUTRAL_MATERIAL = {"type": "standard", "color": "#888888", "roughness": 0.6, "metalness": 0.1}


def make_object(concept: str, asset_src: str | None, sidecar: dict | None,
                category: str, position: list, scale: list, obj_id: str,
                label: str | None = None, animation: dict | None = None) -> dict:
    mat = dict(_NEUTRAL_MATERIAL)
    if asset_src:
        return {
            "id": obj_id, "type": "mesh", "model": asset_src,
            "position": position, "scale": scale, "material": mat,
            **({"label": label} if label else {}),
            "animation": animation or {"type": "none"},
        }
    fb = _CATEGORY_FALLBACK.get(category, {"geom": "sphere", "params": {"radius": 1.0}})
    return {
        "id": obj_id, "type": "primitive",
        "geometry": {"type": fb["geom"], **fb["params"]},
        "position": position, "scale": scale, "material": mat,
        **({"label": label} if label else {}),
        "animation": animation or {"type": "none"},
    }


def scatter(asset_src, sidecar, category, count, radius, center, base_id,
            y_base=0.0, seed=42):
    rng, objs = random.Random(seed), []
    for i in range(count):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.5, radius)
        sv = rng.uniform(0.8, 1.3)
        obj = make_object(category, asset_src, sidecar, category,
                          [round(r*math.cos(a),3), round(y_base,3), round(r*math.sin(a),3)],
                          [round(sv,3)]*3, f"{base_id}_{i}")
        obj["rotation"] = [0, round(rng.uniform(0,360),1), 0]
        objs.append(obj)
    return objs


def grid(asset_src, sidecar, category, count, spacing, center, base_id,
         y_base=0.0, seed=42):
    rng  = random.Random(seed)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    objs, i = [], 0
    for row in range(rows):
        for col in range(cols):
            if i >= count: break
            x = center[0] + (col - cols/2.0) * spacing
            z = center[2] + (row - rows/2.0) * spacing
            sv = rng.uniform(0.8, 1.4)
            obj = make_object(category, asset_src, sidecar, category,
                              [round(x,3), round(y_base,3), round(z,3)],
                              [round(sv,3)]*3, f"{base_id}_{i}")
            obj["rotation"] = [0, round(rng.uniform(0,360),1), 0]
            objs.append(obj)
            i += 1
    return objs


def orbit_cluster(central_src, central_sid, satellite_src, satellite_sid,
                  count, orbit_radii, central_id, base_id,
                  central_cat="abstract", satellite_cat="abstract", seed=42):
    rng = random.Random(seed)
    central = make_object(central_id, central_src, central_sid, central_cat,
                          [0,0,0], [1,1,1], central_id,
                          central_id.replace("_"," ").title())
    sats = []
    for i in range(count):
        r     = rng.uniform(*orbit_radii)
        phase = rng.uniform(0, 2*math.pi)
        sv    = rng.uniform(0.4, 1.0)
        anim  = {"type":"orbit","center":[0,0,0],"speed":round(rng.uniform(0.2,1.2),3),"phase":phase}
        sats.append(make_object(f"{base_id}_sat", satellite_src, satellite_sid, satellite_cat,
                                [round(r*math.cos(phase),3),0,round(r*math.sin(phase),3)],
                                [round(sv,3)]*3, f"{base_id}_{i}", None, anim))
    return [central] + sats
```

---

### 5.5 `pipeline/effects.py`

```python
EFFECT_TEMPLATES = {
    "anim_orbit": {"type": "orbit", "center": [0,0,0], "axis": [0,1,0], "speed": 0.5},
    "anim_spin":  {"type": "spin",  "axis": [0,1,0], "speed": 1.0},
}

def apply_effect(objects: list[dict], handler: str, center=None, speed=None) -> list[dict]:
    template = dict(EFFECT_TEMPLATES.get(handler, {"type": "none"}))
    if center: template["center"] = center
    if speed is not None: template["speed"] = speed
    for obj in objects:
        if obj.get("animation", {}).get("type", "none") == "none":
            obj["animation"] = dict(template)
    return objects
```

---

### 5.6 `pipeline/scene_builder.py`

```python
"""Assembles scene dict from retrieved components."""
import math, random, json
from pathlib import Path
from pipeline import generators, effects

_KB = Path(__file__).parent / "knowledge_base"
with open(_KB / "concept_map.json") as f:
    _CONCEPT_MAP = json.load(f)

DEFAULT_LIGHTS = [
    {"type": "ambient",     "intensity": 0.4, "color": "#ffffff"},
    {"type": "directional", "intensity": 1.2, "color": "#ffffff",
     "position": [10, 10, 10], "castShadow": True},
]

def _uid(base, used):
    cand, n = base, 0
    while cand in used: n += 1; cand = f"{base}_{n}"
    return cand

def _camera(objects):
    if not objects: return {"position":[0,5,20],"target":[0,0,0],"fov":65}
    xs=[o["position"][0] for o in objects]; ys=[o["position"][1] for o in objects]
    zs=[o["position"][2] for o in objects]
    cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
    spread=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs),10)
    return {"position":[round(cx,2),round(cy+spread*0.5,2),round(cz+spread*1.8,2)],
            "target":[round(cx,2),round(cy,2),round(cz,2)],"fov":65}

def _name(intent):
    parts=(intent.get("objects",[])+intent.get("structures",[])+intent.get("systems",[]))
    return " + ".join(p.replace("_"," ").title() for p in parts[:3]) or "Generated Scene"

def build_scene(components: dict, intent: dict, seed: int = 42) -> dict:
    rng=random.Random(seed); all_objs=[]; used=set()
    solo=[a for a in components.get("assets",[]) if not a.get("is_generator")]
    n=max(len(solo),1)
    for i,item in enumerate(solo):
        c=item["concept"]; oid=_uid(c.lower().replace(" ","_"),used); used.add(oid)
        angle=i*2*math.pi/n; r=rng.uniform(4,8)
        pos=[round(r*math.cos(angle),2),0.0,round(r*math.sin(angle),2)]
        sv=rng.uniform(0.85,1.15)
        obj=generators.make_object(c,item.get("asset_src"),item.get("sidecar"),
                                   item.get("category","abstract"),pos,[round(sv,3)]*3,
                                   oid,c.replace("_"," ").title())
        all_objs.append(obj)
    planet=next((o for o in all_objs if any(t in o["id"]
                 for t in ["planet","mars","earth","moon","world","saturn","jupiter"])),None)
    sc=list(planet["position"]) if planet else [0,0,0]
    for item in components.get("assets",[]):
        if not item.get("is_generator"): continue
        c=item["concept"]; gt=item["generator_type"]
        cnt=rng.randint(*item["generator_count"]); bid=_uid(c.lower(),used)
        if gt=="scatter":
            objs=generators.scatter(item.get("asset_src"),item.get("sidecar"),
                                    item.get("category","abstract"),cnt,
                                    item["generator_radius"],sc,bid,sc[1],rng.randint(0,9999))
        elif gt=="grid":
            objs=generators.grid(item.get("asset_src"),item.get("sidecar"),
                                 item.get("category","abstract"),cnt,
                                 item["generator_spacing"],sc,bid,sc[1],rng.randint(0,9999))
        else: objs=[]
        for obj in objs: obj["id"]=_uid(obj["id"],used); used.add(obj["id"])
        all_objs.extend(objs)
    for spec in components.get("generators",[]):
        cnt=rng.randint(*spec["count"]); cs=spec.get("central_sidecar"); ss=spec.get("satellite_sidecar")
        objs=generators.orbit_cluster(
            (cs or {}).get("src"),cs,(ss or {}).get("src"),ss,cnt,(6,18),
            _uid("center",used),spec["concept"],
            (cs or {}).get("category","abstract"),(ss or {}).get("category","abstract"),
            rng.randint(0,9999))
        for obj in objs: obj["id"]=_uid(obj["id"],used); used.add(obj["id"])
        all_objs.extend(objs)
    for eff in components.get("effects",[]):
        all_objs=effects.apply_effect(all_objs,eff["handler"])
    for obj in all_objs:
        if "animation" not in obj: obj["animation"]={"type":"none"}
    return {"name":_name(intent),"objects":all_objs,"lights":DEFAULT_LIGHTS,"camera":_camera(all_objs)}
```

---

### 5.7 `pipeline/fallback_engine.py`

```python
"""
Handles concepts that scored below the embedding threshold.

The semantic parser already handles paraphrases via embeddings.
This module is a final safety net for:
  - Very unusual words not in the concept corpus
  - Direct synonym_map lookups as a supplement

LEVEL 1 — concept_map has entry           → use it
LEVEL 2 — synonym_map has entry           → remap to known concept
LEVEL 3 — compound word split             → try sub-tokens
LEVEL 4 — signal LLM                      → return as unresolved
"""
import re, json
from pathlib import Path

_KB = Path(__file__).parent / "knowledge_base"
with open(_KB / "concept_map.json") as f:  CONCEPT_MAP = json.load(f)
with open(_KB / "synonym_map.json") as f:
    SYNONYM_MAP = {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _resolve_single(token: str) -> str | None:
    t = token.lower().strip()
    if t in CONCEPT_MAP:              return t
    if t in SYNONYM_MAP:
        for s in SYNONYM_MAP[t]:
            if s in CONCEPT_MAP:     return s
    for part in re.split(r"[-_\s]", t):
        if part in CONCEPT_MAP:      return part
        if part in SYNONYM_MAP:
            for s in SYNONYM_MAP[part]:
                if s in CONCEPT_MAP: return s
    return None


def resolve_intent(intent: dict) -> tuple[dict, list[str]]:
    """
    Resolves anything in intent that didn't make it through embedding.
    Returns (resolved_intent, unresolved_list).
    unresolved_list = concepts for LLM bridge.
    """
    resolved = {"objects": [], "structures": [], "systems": [], "effects": []}
    unresolved: list[str] = []
    for bucket in ["objects", "structures", "systems", "effects"]:
        for concept in intent.get(bucket, []):
            mapped = _resolve_single(concept)
            if mapped:
                entry = CONCEPT_MAP[mapped]
                target = {"object":"objects","structure":"structures",
                          "system":"systems","effect":"effects"}.get(entry["type"], bucket)
                if mapped not in resolved[target]:
                    resolved[target].append(mapped)
            else:
                if concept not in unresolved:
                    unresolved.append(concept)
    return resolved, unresolved
```

---

### 5.8 `pipeline/llm_bridge.py`

```python
"""LLM fallback — last resort, called only for truly unresolvable concepts."""
import json, os

SYSTEM_PROMPT = """\
You are a strict 3D object generator. Output ONLY a raw JSON array. No markdown.

Rules (every rule is non-negotiable):
1. Output format: [ { object }, { object }, ... ]
2. Every object needs: id, type, position, scale, material, animation
3. type="primitive" → must have geometry.type ∈ {sphere,box,cylinder,plane,ring,capsule,torus}
4. type="mesh" → must have model (string path). Only if you are certain the file exists.
5. material.type always "standard". material.color always "#rrggbb" hex (NOT float array).
6. animation.type ∈ {none,orbit,spin}.
   orbit → add: "center":[0,0,0], "speed":0.5
   spin  → add: "axis":[0,1,0], "speed":1.0
7. All numbers finite. position and scale are [x,y,z] arrays. Max 15 objects.
8. Do NOT invent geometry types.
"""


def llm_generate_objects(description: str) -> list[dict] | None:
    prompt = f"Generate 3D objects for: {description}\nOutput only the JSON array."
    raw = _try_groq(prompt) or _try_ollama(prompt)
    return _parse_array(raw) if raw else None


def _try_groq(prompt):
    try:
        import requests
        key = os.getenv("GROQ_API_KEY")
        if not key: return None
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role":"system","content":SYSTEM_PROMPT},
                                {"role":"user",  "content":prompt}],
                  "temperature": 0.2, "max_tokens": 1500},
            timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception: return None


def _try_ollama(prompt):
    try:
        from llm.ollama_client import generate_scene_ollama
        result = generate_scene_ollama(prompt, None)
        if isinstance(result, dict) and "objects" in result:
            return json.dumps(result["objects"])
    except Exception: pass
    return None


def _parse_array(raw):
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```"); raw = parts[1][4:] if parts[1].startswith("json") else parts[1]
        start, end = raw.find("["), raw.rfind("]") + 1
        if start == -1 or end <= start: return None
        return json.loads(raw[start:end])
    except Exception: return None
```

---

### 5.9 `pipeline/scene_validator.py` ← REPLACES Node subprocess

#### Why this module exists

Read Section 0.2 Decision 2 before reading the code.

#### How it mirrors the TypeScript

Every function below has a direct counterpart in `gui/lib/sceneFactory.ts`.
The correspondence is:

| Python function | TypeScript counterpart |
|----------------|----------------------|
| `_is_hex(v)` | `isHex(v)` |
| `_is_vec3(v)` | `isVec3(v)` |
| `_is_num(v, min, max)` | `isNum(v, min, max)` |
| `_validate_material(raw, prefix)` | `validateMaterial(raw, prefix)` |
| `_validate_geometry(raw, prefix)` | `validateGeometry(raw, prefix)` |
| `_validate_animation(raw, prefix)` | `validateAnimation(raw, prefix)` |
| `_validate_object(raw, index)` | `validateObject(raw, index)` |
| `_validate_parent_refs(objects)` | `validateParentRefs(objects)` |
| `validate_scene(raw)` | `validateScene(raw)` |

**If `gui/lib/sceneFactory.ts` is ever modified, this file must be updated too.**
There is a comment at the top of the file listing the exact TypeScript lines
that each Python rule corresponds to, so the update is mechanical.

```python
"""
Python port of gui/lib/sceneFactory.ts → validateScene().

MIRROR STATUS: synchronized with sceneFactory.ts as of 2026-05-07.
If sceneFactory.ts is modified, update this file and the date above.

Corresponding TypeScript lines:
  _is_hex            → TS line 166-168   isHex()
  _is_vec3           → TS line 170-172   isVec3()
  _is_num            → TS line 174-176   isNum()
  _validate_material → TS line 180-222   validateMaterial()
  _validate_geometry → TS line 226-282   validateGeometry()
  _validate_animation→ TS line 286-318   validateAnimation()
  _validate_object   → TS line 322-398   validateObject()
  _validate_parent_refs→TS line 402-432  validateParentRefs()
  validate_scene     → TS line 436-524   validateScene()

Performance: ~1–3ms per call, no subprocess, no I/O.
"""
import re
import math
from typing import Any

# ── Constants — must match sceneFactory.ts exactly ───────────────────────────
VALID_OBJECT_TYPES = {"primitive", "mesh"}
VALID_GEOM_TYPES   = {"sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"}
VALID_ANIM_TYPES   = {"none", "orbit", "spin"}
VALID_LIGHT_TYPES  = {"ambient", "directional", "point", "spot"}

DEFAULT_CAMERA = {"position": [0, 5, 20], "target": [0, 0, 0], "fov": 65}
DEFAULT_LIGHTS = [
    {"type": "ambient",     "intensity": 0.4},
    {"type": "directional", "intensity": 1.2, "position": [10, 10, 10], "castShadow": True},
]

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# ── Helpers (mirrors of TS helper functions) ──────────────────────────────────

def _is_hex(v: Any) -> bool:
    """TS: isHex(v) — line 166"""
    return isinstance(v, str) and bool(_HEX_RE.match(v))


def _is_vec3(v: Any) -> bool:
    """TS: isVec3(v) — line 170"""
    return (
        isinstance(v, list)
        and len(v) == 3
        and all(isinstance(n, (int, float)) and math.isfinite(n) for n in v)
    )


def _is_num(v: Any, min_val: float = -math.inf, max_val: float = math.inf) -> bool:
    """TS: isNum(v, min, max) — line 174"""
    return isinstance(v, (int, float)) and math.isfinite(v) and min_val <= v <= max_val


# ── Material validation ───────────────────────────────────────────────────────

def _validate_material(raw: Any, prefix: str) -> tuple[dict | None, list[str]]:
    """TS: validateMaterial(raw, prefix) — lines 180-222"""
    errors: list[str] = []

    if not isinstance(raw, dict):
        return None, [f"{prefix}: material is required"]

    if raw.get("type") != "standard":
        errors.append(f'{prefix}.material.type must be "standard", got "{raw.get("type")}"')
    if not _is_hex(raw.get("color")):
        errors.append(f'{prefix}.material.color must be "#rrggbb" hex, got "{raw.get("color")}"')
    if not _is_num(raw.get("roughness"), 0, 1):
        errors.append(f"{prefix}.material.roughness must be 0–1, got {raw.get('roughness')}")
    if not _is_num(raw.get("metalness"), 0, 1):
        errors.append(f"{prefix}.material.metalness must be 0–1, got {raw.get('metalness')}")
    if raw.get("opacity") is not None and not _is_num(raw["opacity"], 0, 1):
        errors.append(f"{prefix}.material.opacity must be 0–1")
    if raw.get("transparent") is not None and not isinstance(raw["transparent"], bool):
        errors.append(f"{prefix}.material.transparent must be a boolean")
    if raw.get("emissive") is not None and not _is_hex(raw["emissive"]):
        errors.append(f"{prefix}.material.emissive must be hex")
    if raw.get("emissiveIntensity") is not None and not _is_num(raw["emissiveIntensity"], 0):
        errors.append(f"{prefix}.material.emissiveIntensity must be >= 0")
    for field in ["map", "normalMap", "roughnessMap", "metalnessMap", "emissiveMap"]:
        if raw.get(field) is not None and not isinstance(raw[field], str):
            errors.append(f"{prefix}.material.{field} must be a string path")

    if errors:
        return None, errors

    return {
        "type":              "standard",
        "color":             raw["color"],
        "roughness":         raw["roughness"],
        "metalness":         raw["metalness"],
        "opacity":           raw.get("opacity"),
        "transparent":       raw.get("transparent"),
        "map":               raw.get("map"),
        "normalMap":         raw.get("normalMap"),
        "roughnessMap":      raw.get("roughnessMap"),
        "metalnessMap":      raw.get("metalnessMap"),
        "emissive":          raw.get("emissive"),
        "emissiveMap":       raw.get("emissiveMap"),
        "emissiveIntensity": raw.get("emissiveIntensity"),
    }, []


# ── Geometry validation ───────────────────────────────────────────────────────

def _validate_geometry(raw: Any, prefix: str) -> tuple[dict | None, list[str]]:
    """TS: validateGeometry(raw, prefix) — lines 226-282"""
    if not isinstance(raw, dict):
        return None, [f'{prefix}: geometry is required for type "primitive"']

    t = raw.get("type")
    if t not in VALID_GEOM_TYPES:
        return None, [
            f'{prefix}.geometry.type must be one of '
            f'{"|".join(sorted(VALID_GEOM_TYPES))}, got "{t}"'
        ]

    errors: list[str] = []

    if t in ("sphere", "capsule") and raw.get("radius") is not None:
        if not _is_num(raw["radius"], 0): errors.append(f"{prefix}.geometry.radius must be > 0")
    if t == "capsule" and raw.get("length") is not None:
        if not _is_num(raw["length"], 0): errors.append(f"{prefix}.geometry.length must be > 0")
    if t == "torus":
        if raw.get("radius") is not None and not _is_num(raw["radius"], 0):
            errors.append(f"{prefix}.geometry.radius must be > 0")
        if raw.get("tube") is not None and not _is_num(raw["tube"], 0):
            errors.append(f"{prefix}.geometry.tube must be > 0")
    if t == "ring":
        if raw.get("innerRadius") is not None and not _is_num(raw["innerRadius"], 0):
            errors.append(f"{prefix}.geometry.innerRadius must be > 0")
        if raw.get("outerRadius") is not None and not _is_num(raw["outerRadius"], 0):
            errors.append(f"{prefix}.geometry.outerRadius must be > 0")
        if (_is_num(raw.get("innerRadius"), 0) and _is_num(raw.get("outerRadius"), 0)
                and raw["outerRadius"] <= raw["innerRadius"]):
            errors.append(f"{prefix}.geometry.outerRadius must be greater than innerRadius")
        if raw.get("thetaSegments") is not None and not _is_num(raw["thetaSegments"], 3):
            errors.append(f"{prefix}.geometry.thetaSegments must be >= 3")
    if t in ("box", "plane") and raw.get("width") is not None:
        if not _is_num(raw["width"], 0): errors.append(f"{prefix}.geometry.width must be > 0")
    if raw.get("from") is not None and not _is_vec3(raw["from"]):
        errors.append(f"{prefix}.geometry.from must be [x,y,z]")
    if raw.get("to") is not None and not _is_vec3(raw["to"]):
        errors.append(f"{prefix}.geometry.to must be [x,y,z]")

    if errors:
        return None, errors

    return {
        "type":          t,
        "radius":        raw.get("radius"),
        "length":        raw.get("length"),
        "tube":          raw.get("tube"),
        "innerRadius":   raw.get("innerRadius"),
        "outerRadius":   raw.get("outerRadius"),
        "thetaSegments": raw.get("thetaSegments"),
        "width":         raw.get("width"),
        "height":        raw.get("height"),
        "depth":         raw.get("depth"),
        "from":          raw.get("from"),
        "to":            raw.get("to"),
    }, []


# ── Animation validation ──────────────────────────────────────────────────────

def _validate_animation(raw: Any, prefix: str) -> tuple[dict | None, list[str]]:
    """TS: validateAnimation(raw, prefix) — lines 286-318"""
    if raw is None:
        return {"type": "none"}, []
    if not isinstance(raw, dict):
        return None, [f"{prefix}.animation must be an object"]

    t = raw.get("type")
    if t not in VALID_ANIM_TYPES:
        return None, [
            f'{prefix}.animation.type must be one of '
            f'{"|".join(sorted(VALID_ANIM_TYPES))}, got "{t}"'
        ]

    errors: list[str] = []
    if raw.get("center")     is not None and not _is_vec3(raw["center"]):
        errors.append(f"{prefix}.animation.center must be [x,y,z]")
    if raw.get("axis")       is not None and not _is_vec3(raw["axis"]):
        errors.append(f"{prefix}.animation.axis must be [x,y,z]")
    if raw.get("center_ref") is not None and not isinstance(raw["center_ref"], str):
        errors.append(f"{prefix}.animation.center_ref must be a string")
    if raw.get("speed")      is not None and not _is_num(raw["speed"]):
        errors.append(f"{prefix}.animation.speed must be a finite number")
    if raw.get("phase")      is not None and not _is_num(raw["phase"]):
        errors.append(f"{prefix}.animation.phase must be a finite number")

    if errors:
        return None, errors

    return {
        "type":       t,
        "center":     raw.get("center"),
        "center_ref": raw.get("center_ref"),
        "axis":       raw.get("axis"),
        "speed":      raw.get("speed"),
        "phase":      raw.get("phase"),
    }, []


# ── Object validation ─────────────────────────────────────────────────────────

def _validate_object(raw: Any, index: int) -> tuple[dict | None, list[str]]:
    """TS: validateObject(raw, index) — lines 322-398"""
    errors: list[str] = []

    if not isinstance(raw, dict):
        return None, [f"objects[{index}]: must be an object"]

    prefix = f'objects[{index}](id="{raw.get("id")}")'

    if not isinstance(raw.get("id"), str) or not raw["id"].strip():
        errors.append(f"{prefix}: id must be a non-empty string")

    obj_type = raw.get("type")
    if obj_type not in VALID_OBJECT_TYPES:
        return None, [f'{prefix}: type must be "primitive" or "mesh", got "{obj_type}"']

    if raw.get("parent") is not None:
        if not isinstance(raw["parent"], str) or not raw["parent"].strip():
            errors.append(f"{prefix}: parent must be a non-empty string id")

    if not _is_vec3(raw.get("position")):
        errors.append(f"{prefix}: position must be [x,y,z] of finite numbers")

    geom = None
    if obj_type == "primitive":
        geom, ge = _validate_geometry(raw.get("geometry"), prefix)
        errors.extend(ge)
    else:
        if not isinstance(raw.get("model"), str) or not raw["model"].strip():
            errors.append(f"{prefix}: model (path to .glb/.gltf) is required for type \"mesh\"")

    mat, me = _validate_material(raw.get("material"), prefix)
    errors.extend(me)

    if raw.get("rotation") is not None and not _is_vec3(raw["rotation"]):
        errors.append(f"{prefix}: rotation must be [rx,ry,rz]")
    if raw.get("scale") is not None and not _is_vec3(raw["scale"]):
        errors.append(f"{prefix}: scale must be [sx,sy,sz]")
    if raw.get("label") is not None and not isinstance(raw["label"], str):
        errors.append(f"{prefix}: label must be a string")

    anim, ae = _validate_animation(raw.get("animation"), prefix)
    errors.extend(ae)

    # Fail this object if any required field is invalid
    position_ok = _is_vec3(raw.get("position"))
    type_ok = (
        (obj_type == "primitive" and geom is not None)
        or (obj_type == "mesh" and isinstance(raw.get("model"), str) and raw["model"].strip())
    )
    if not position_ok or mat is None or not type_ok:
        return None, errors

    return {
        "id":       raw["id"],
        "type":     obj_type,
        "parent":   raw["parent"] if isinstance(raw.get("parent"), str) and raw["parent"].strip() else None,
        "geometry": geom,
        "model":    raw.get("model") if obj_type == "mesh" else None,
        "position": raw["position"],
        "rotation": raw["rotation"] if _is_vec3(raw.get("rotation")) else None,
        "scale":    raw["scale"]    if _is_vec3(raw.get("scale"))    else None,
        "material": mat,
        "label":    raw["label"] if isinstance(raw.get("label"), str) else None,
        "animation": anim or {"type": "none"},
    }, errors


# ── Parent reference + cycle check ────────────────────────────────────────────

def _validate_parent_refs(objects: list[dict]) -> list[str]:
    """TS: validateParentRefs(objects) — lines 402-432"""
    errors: list[str] = []
    id_set = {o["id"] for o in objects}

    for obj in objects:
        parent = obj.get("parent")
        if parent is None:
            continue
        if parent not in id_set:
            errors.append(f'objects(id="{obj["id"]}"): parent "{parent}" references an unknown id')
            continue
        if parent == obj["id"]:
            errors.append(f'objects(id="{obj["id"]}"): parent cannot reference self')
            continue
        # Cycle detection: walk parent chain
        visited: set[str] = set()
        cur: str | None = obj["id"]
        parent_map = {o["id"]: o.get("parent") for o in objects}
        while cur is not None:
            if cur in visited:
                errors.append(f'objects(id="{obj["id"]}"): circular parent dependency detected')
                break
            visited.add(cur)
            cur = parent_map.get(cur)

    return errors


# ── Main entry point ──────────────────────────────────────────────────────────

def validate_scene(raw: Any) -> dict:
    """
    Python mirror of TS validateScene(raw) — lines 436-524.

    Returns:
        {
            "scene":  { "name": ..., "objects": [...], "lights": [...], "camera": {...} },
            "errors": [ "non-fatal error string", ... ],
            "fatal":  "fatal error string" | None
        }

    Usage:
        from pipeline.scene_validator import validate_scene
        result = validate_scene(my_scene_dict)
        if result["fatal"]:
            print("Scene rejected:", result["fatal"])
        elif result["errors"]:
            print("Some objects skipped:", result["errors"])
        else:
            print("Scene valid:", len(result["scene"]["objects"]), "objects")
    """
    all_errors: list[str] = []

    # Fatal: root must be a dict
    if not isinstance(raw, dict):
        return {
            "scene": {"objects": [], "lights": DEFAULT_LIGHTS, "camera": DEFAULT_CAMERA},
            "errors": [],
            "fatal": "Scene root must be a JSON object",
        }

    # Objects — required, non-empty
    valid_objects: list[dict] = []
    if not isinstance(raw.get("objects"), list) or len(raw["objects"]) == 0:
        all_errors.append("scene.objects must be a non-empty array")
    else:
        for i, obj_raw in enumerate(raw["objects"]):
            obj, errs = _validate_object(obj_raw, i)
            all_errors.extend(errs)
            if obj:
                valid_objects.append(obj)

    # Parent ref check
    all_errors.extend(_validate_parent_refs(valid_objects))

    # Lights — optional, fall back to defaults
    lights = DEFAULT_LIGHTS
    if raw.get("lights") is not None:
        if not isinstance(raw["lights"], list):
            all_errors.append("scene.lights must be an array")
        else:
            valid_lights = []
            for l in raw["lights"]:
                if not isinstance(l, dict): continue
                if l.get("type") not in VALID_LIGHT_TYPES:
                    all_errors.append(f'light: type must be one of {"|".join(sorted(VALID_LIGHT_TYPES))}, got "{l.get("type")}"')
                    continue
                if not _is_num(l.get("intensity"), 0):
                    all_errors.append(f"light({l['type']}): intensity must be >= 0")
                    continue
                if l.get("color") is not None and not _is_hex(l["color"]):
                    all_errors.append(f"light({l['type']}): color must be \"#rrggbb\" hex")
                    continue
                if l.get("position") is not None and not _is_vec3(l["position"]):
                    all_errors.append(f"light({l['type']}): position must be [x,y,z]")
                    continue
                valid_lights.append({
                    "type":       l["type"],
                    "intensity":  l["intensity"],
                    "color":      l.get("color", "#ffffff"),
                    "position":   l.get("position"),
                    "castShadow": l.get("castShadow", False),
                })
            if valid_lights:
                lights = valid_lights

    # Camera — optional, fall back to defaults
    camera = DEFAULT_CAMERA
    if raw.get("camera") is not None:
        c = raw["camera"]
        if not isinstance(c, dict):
            all_errors.append("scene.camera must be an object")
        elif not _is_vec3(c.get("position")):
            all_errors.append("scene.camera.position must be [x,y,z]")
        elif not _is_vec3(c.get("target")):
            all_errors.append("scene.camera.target must be [x,y,z]")
        else:
            camera = {
                "position": c["position"],
                "target":   c["target"],
                "fov":      c["fov"] if _is_num(c.get("fov"), 1, 179) else DEFAULT_CAMERA["fov"],
            }

    return {
        "scene": {
            "name":    raw.get("name") if isinstance(raw.get("name"), str) else None,
            "objects": valid_objects,
            "lights":  lights,
            "camera":  camera,
        },
        "errors": all_errors,
        "fatal":  None,
    }


def is_valid(result: dict) -> bool:
    return result.get("fatal") is None and len(result.get("scene", {}).get("objects", [])) > 0
```

---

### 5.10 `pipeline/repair_loop.py`

```python
"""Structural repairs without LLM. Max 3 iterations."""
import re

DEMO_FALLBACK = {
    "name": "Fallback",
    "objects": [{"id": "fallback", "type": "primitive",
                 "geometry": {"type": "sphere", "radius": 2.0},
                 "position": [0,0,0], "scale": [1,1,1],
                 "material": {"type":"standard","color":"#ffcc22","roughness":0.3,"metalness":0.0},
                 "animation": {"type":"spin","axis":[0,1,0],"speed":0.5}}],
    "lights":  [{"type":"ambient","intensity":0.4},
                {"type":"directional","intensity":1.2,"position":[10,10,10],"castShadow":True}],
    "camera":  {"position":[0,5,15],"target":[0,0,0],"fov":65},
}


def _fix_colors(s):
    for o in s.get("objects",[]):
        m=o.get("material",{}); c=m.get("color")
        if isinstance(c,list) and len(c)==3:
            r,g,b=[max(0,min(255,int(x*255))) for x in c]; m["color"]=f"#{r:02x}{g:02x}{b:02x}"
        elif not isinstance(c,str) or not c.startswith("#"): m["color"]="#888888"
    return s


def _fix_material_types(s):
    for o in s.get("objects",[]):
        m=o.get("material",{})
        if m.get("type")!="standard": m["type"]="standard"
        if not _is_num_01(m.get("roughness")): m["roughness"]=0.6
        if not _is_num_01(m.get("metalness")): m["metalness"]=0.1
    return s


def _is_num_01(v): return isinstance(v,(int,float)) and 0<=v<=1


def _fix_animations(s):
    valid={"none","orbit","spin"}
    for o in s.get("objects",[]):
        a=o.get("animation")
        if not isinstance(a,dict) or a.get("type") not in valid: o["animation"]={"type":"none"}
    return s


def _fix_geometries(s):
    valid={"sphere","box","cylinder","plane","ring","capsule","torus"}
    for o in s.get("objects",[]):
        if o.get("type")=="primitive":
            g=o.get("geometry",{})
            if g.get("type") not in valid: o["geometry"]={"type":"sphere","radius":1.0}
    return s


def _fix_transparency(s):
    for o in s.get("objects",[]):
        m=o.get("material",{})
        op=m.get("opacity")
        if isinstance(op,(int,float)) and op<1.0: m["transparent"]=True
    return s


def _fix_required(s):
    for o in s.get("objects",[]):
        if not isinstance(o.get("position"),list) or len(o.get("position",[]))!=3: o["position"]=[0,0,0]
        if not isinstance(o.get("scale"),list): o["scale"]=[1,1,1]
        if "material" not in o: o["material"]={"type":"standard","color":"#888888","roughness":0.6,"metalness":0.1}
        if "animation" not in o: o["animation"]={"type":"none"}
    return s


def repair(scene: dict, errors: list[str], max_iterations: int = 3) -> dict:
    current = scene
    for _ in range(max_iterations):
        e = " ".join(errors).lower()
        if "color"          in e: current = _fix_colors(current)
        if "material.type"  in e: current = _fix_material_types(current)
        if "animation.type" in e: current = _fix_animations(current)
        if "geometry.type"  in e: current = _fix_geometries(current)
        if "transparent"    in e: current = _fix_transparency(current)
        current = _fix_required(current)
        if current.get("objects"): return current
    return DEMO_FALLBACK
```

---

### 5.11 `pipeline/pipeline_runner.py`

```python
"""
Main entry point: transcript → valid scene JSON → core/outputs/live_scene.json

Usage:
    python -m pipeline.pipeline_runner "create a city on mars"
    python -m pipeline.pipeline_runner          # uses voice recorder
"""
import json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from dotenv import load_dotenv; load_dotenv()

from pipeline.semantic_parser  import SemanticParser
from pipeline.fallback_engine  import resolve_intent
from pipeline.retrieval        import retrieve
from pipeline.scene_builder    import build_scene
from pipeline.llm_bridge       import llm_generate_objects
from pipeline.scene_validator  import validate_scene, is_valid
from pipeline.repair_loop      import repair, DEMO_FALLBACK

OUTPUT_PATH = _ROOT / "core" / "outputs" / "live_scene.json"

DEFAULT_LIGHTS = [
    {"type": "ambient",     "intensity": 0.4, "color": "#ffffff"},
    {"type": "directional", "intensity": 1.2, "color": "#ffffff",
     "position": [10, 10, 10], "castShadow": True},
]

# Create parser once at startup — model load takes ~2s but only happens once
print("[startup] Loading semantic parser (first run downloads model ~80MB)...")
_PARSER = SemanticParser()
print("[startup] Ready.")


def run_pipeline(transcript: str) -> dict:
    import time
    t0 = time.perf_counter()

    print(f"\n[1] Transcript: {repr(transcript)}")

    # ── Semantic classification (~15ms) ──────────────────────────────────
    raw_intent = _PARSER.parse_intent(transcript)
    print(f"[2] Embedding intent: {raw_intent}  ({(time.perf_counter()-t0)*1000:.0f}ms)")

    # ── Fallback for below-threshold concepts (<5ms) ──────────────────────
    resolved_intent, unresolved = resolve_intent(raw_intent)
    print(f"[3] Resolved: {resolved_intent}")
    if unresolved: print(f"    Unresolved: {unresolved}")

    has_concepts = any(resolved_intent[k] for k in ["objects","structures","systems"])

    if not has_concepts:
        print("[4] Nothing resolved — full LLM fallback")
        llm_objs = llm_generate_objects(transcript)
        scene = {"name": transcript[:50], "objects": llm_objs or [],
                 "lights": DEFAULT_LIGHTS,
                 "camera": {"position":[0,5,20],"target":[0,0,0],"fov":65}}
        if not scene["objects"]: scene = DEMO_FALLBACK
    else:
        # ── Retrieval + scene build (~20ms) ───────────────────────────────
        components = retrieve(resolved_intent)
        scene = build_scene(components, resolved_intent)
        print(f"[4] Built: {len(scene.get('objects',[]))} objects  ({(time.perf_counter()-t0)*1000:.0f}ms)")

        if unresolved:
            print(f"[5] LLM for unresolved: {unresolved}")
            for concept in unresolved:
                extra = llm_generate_objects(concept)
                if extra: scene["objects"].extend(extra)

    # ── In-process validation (~2ms) ─────────────────────────────────────
    vr = validate_scene(scene)
    print(f"[6] Validation — fatal: {vr.get('fatal')}, errors: {len(vr.get('errors',[]))}  ({(time.perf_counter()-t0)*1000:.0f}ms)")

    if vr.get("fatal"):
        scene = repair(scene, [vr["fatal"]])
        vr    = validate_scene(scene)

    if vr.get("errors"):
        scene = repair(scene, vr["errors"])
        vr    = validate_scene(scene)

    final = vr.get("scene") or scene
    if not final.get("objects"): final = DEMO_FALLBACK

    # ── Write output (<1ms) ───────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(final, indent=2))

    elapsed = (time.perf_counter()-t0)*1000
    print(f"[7] Saved → {OUTPUT_PATH}")
    print(f"    Scene: \"{final.get('name')}\" — {len(final.get('objects',[]))} objects — {elapsed:.0f}ms total")
    return final


def run_with_voice() -> dict:
    from voice.recorder import record_audio
    from voice.transcriber import transcribe
    print("\nRecording 5s — SPEAK NOW")
    audio = record_audio(duration=5)
    transcript = transcribe(audio)
    if not transcript:
        print("No speech detected.")
        return DEMO_FALLBACK
    return run_pipeline(transcript)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(" ".join(sys.argv[1:]))
    else:
        run_with_voice()
```

---

## 6. FIXES TO EXISTING FILES

### 6.1 `llm/scene_schema.py` — REWRITE

```python
import re
from typing import Literal, Optional
from pydantic import BaseModel, field_validator

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

class MaterialDef(BaseModel):
    type: Literal["standard"]
    color: str
    roughness: float
    metalness: float
    opacity: Optional[float] = None
    transparent: Optional[bool] = None
    emissive: Optional[str] = None
    emissiveIntensity: Optional[float] = None

    @field_validator("color")
    @classmethod
    def check_color(cls, v):
        if not HEX_RE.match(v): raise ValueError(f"color must be #rrggbb, got {v!r}")
        return v

class GeometryDef(BaseModel):
    type: Literal["sphere","box","cylinder","plane","ring","capsule","torus"]
    radius: Optional[float] = None; length: Optional[float] = None
    tube: Optional[float] = None; innerRadius: Optional[float] = None
    outerRadius: Optional[float] = None; width: Optional[float] = None
    height: Optional[float] = None; depth: Optional[float] = None

class AnimationDef(BaseModel):
    type: Literal["none","orbit","spin"]
    center: Optional[list[float]] = None
    axis: Optional[list[float]] = None
    speed: Optional[float] = None
    phase: Optional[float] = None

class SceneObject(BaseModel):
    id: str; type: Literal["primitive","mesh"]
    geometry: Optional[GeometryDef] = None; model: Optional[str] = None
    position: list[float]; scale: Optional[list[float]] = None
    material: MaterialDef; animation: Optional[AnimationDef] = None

class SceneDef(BaseModel):
    name: Optional[str] = None; objects: list[SceneObject]
    lights: Optional[list] = None; camera: Optional[dict] = None
```

### 6.2 `llm/prompt_templates.py` — Remove broken import

```python
# llm/prompt_templates.py — GUI schema prompts

_RULE = """
Output ONLY valid JSON. Schema:
{ "name": "...", "objects": [ { "id":"...", "type":"primitive",
  "geometry": {"type":"sphere"}, "position":[0,0,0], "scale":[1,1,1],
  "material": {"type":"standard","color":"#ff8800","roughness":0.7,"metalness":0.0},
  "animation": {"type":"none"} } ],
  "lights": [{"type":"ambient","intensity":0.4}],
  "camera": {"position":[0,5,20],"target":[0,0,0]} }
Rules: color="#rrggbb". animation.type∈{none,orbit,spin}.
geometry.type∈{sphere,box,cylinder,plane,ring,capsule,torus}.
"""

def build_system_prompt(): return _RULE
def build_refinement_prompt(prev, cmd):
    import json; return f"{_RULE}\n\nExisting:\n{json.dumps(prev)}\n\nModify: {cmd}"
```

### 6.3 `llm/groq_client.py` — Fix FALLBACK_SCENE only

```python
FALLBACK_SCENE = {
    "name": "Fallback",
    "objects": [{"id":"fallback_sphere","type":"primitive",
                 "geometry":{"type":"sphere","radius":2.0},
                 "position":[0,0,0],"scale":[1,1,1],
                 "material":{"type":"standard","color":"#ffcc22","roughness":0.3,"metalness":0.0},
                 "animation":{"type":"none"}}],
    "lights":[{"type":"ambient","intensity":0.4}],
    "camera":{"position":[0,5,20],"target":[0,0,0],"fov":65},
}
```

### 6.4 `core/state/scene_grammar.py` — Create stub

```python
# Stub — prevents ImportError from legacy code.
```

---

## 7. SETUP AND RUN ORDER

```bash
# Step 1 — Install Python dependencies
pip install pydantic requests python-dotenv sentence-transformers

# Step 2 — Download real assets (~5 min first run, skips already-downloaded)
python -m pipeline.asset_ingester

# Step 3 — Build concept_map from ingested assets
python -m pipeline.kb_builder
# Expected output:
#   concept_map.json → 80–150 entries
#   concept_descriptions.json → same count

# Step 4 — Debug the embedding similarity for your test prompts
#   This shows you the top concept matches and their scores.
#   Useful to verify the threshold is set correctly.
python -m pipeline.semantic_parser "dragon flying over a city on mars with orbiting satellites"
# Expected: planet, city/buildings, human → high scores; orbit → high score

# Step 5 — Run full pipeline with text input
python -m pipeline.pipeline_runner "create a forest on a planet with orbiting satellites"
# Expected: built + validated + saved in <100ms

# Step 6 — Run with voice
python -m pipeline.pipeline_runner
```

---

## 8. TUNING THE EMBEDDING THRESHOLD

If the system is picking up too many irrelevant concepts, raise `SIMILARITY_THRESHOLD`
in `semantic_parser.py` from 0.35 to 0.40.

If the system is missing obvious concepts (e.g. "trees" not matching "forest"),
lower it to 0.30.

Use the debug command to inspect scores before changing the threshold:
```bash
python -m pipeline.semantic_parser "your test prompt here"
```

The output shows all concepts sorted by score. You want a natural break point
between relevant and irrelevant concepts — set the threshold just above the
irrelevant ones.

---

## 9. FINAL TEST — MUST PASS

**Prompt:** `"dragon flying over a city on mars with orbiting satellites"`

Expected trace:
```
[1] Transcript: "dragon flying over a city on mars with orbiting satellites"

[2] Embedding intent:
    objects:    ["human", "planet", "satellite"]   ← dragon→human via similarity
    structures: ["city"]
    systems:    []
    effects:    ["orbit"]
    ~15ms

[3] Resolved: all L1 (already in concept_map), unresolved=[]

[4] Build scene:
    - human GLB or capsule at orbit radius
    - planet GLB or sphere at [0,0,0]
    - grid of building GLBs or boxes centered on planet
    - satellite GLBs or boxes placed
    - orbit effect applied to objects without animation
    - camera auto-fitted
    ~20ms

[6] Python validate_scene():
    fatal=None, errors=[]
    ~2ms

[7] Saved → core/outputs/live_scene.json
    Total: ~40ms
```

**Success criteria:**
1. `live_scene.json` written to disk
2. `validate_scene()` returns `fatal=None`
3. At least 3 objects present
4. Total latency under 100ms (no LLM triggered)
5. GUI renders without error

---

## 10. WHAT NOT TO DO

```
❌ Instantiate SemanticParser more than once — it takes 1-3s to load
❌ Spawn a Node subprocess for validation — use scene_validator.py
❌ Edit concept_map.json by hand — run kb_builder.py instead
❌ Edit concept_descriptions.json — generated file
❌ Hardcode colors, geometry values, or asset paths in any module
❌ Use LLM before attempting semantic matching + fallback engine
❌ Use Python renderer schema (orbit_center, RGB float arrays, size field)
❌ Set material.color to anything other than "#rrggbb" hex
❌ Use animation.type other than none | orbit | spin
❌ Create primitive without geometry; create mesh without model path
❌ Reference a parent id not in the objects array
❌ Set opacity < 1 without transparent: true
❌ Invent geometry types (cone, wedge, etc.)
```

---

*Schema in Section 1 is extracted directly from `gui/lib/sceneFactory.ts`.
Python validator in Section 5.9 mirrors it line-for-line.
If `sceneFactory.ts` changes, both Section 1 and `scene_validator.py` must be updated.*
