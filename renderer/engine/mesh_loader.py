"""renderer/mesh_loader.py
OBJ mesh loader with in-process cache.

Only supports geometry (v / f lines). Normals are computed per face at load
time so draw_mesh() can call glNormal3f without recalculating every frame.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_mesh_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mesh(filepath: str) -> dict | None:
    """Return cached mesh data for filepath, loading it on first call.

    Returns None if the file is missing or unparseable (warning is printed).
    The returned dict has keys:
        "vertices"       np.ndarray (N, 3)  float64
        "faces"          list of (i0, i1, i2) int tuples  (0-based)
        "face_normals"   np.ndarray (F, 3)  float64, unit-length per face
    """
    if filepath not in _mesh_cache:
        mesh = _load_obj(filepath)
        if mesh is None:
            return None
        _mesh_cache[filepath] = mesh
    return _mesh_cache[filepath]


def clear_cache() -> None:
    """Evict all cached meshes (useful for hot-reload during development)."""
    _mesh_cache.clear()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_obj(filepath: str) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"[mesh_loader] WARNING: cannot open '{filepath}': {exc}")
        return None

    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        token = parts[0]

        if token == "v":
            # vertex position
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (IndexError, ValueError):
                print(f"[mesh_loader] WARNING: '{filepath}' line {lineno}: bad vertex, skipping.")

        elif token == "f":
            # face — indices may be "v", "v/vt", "v/vt/vn", or "v//vn"
            raw_indices = parts[1:]
            indices: list[int] = []
            ok = True
            for token_idx in raw_indices:
                try:
                    v_idx = int(token_idx.split("/")[0])
                    # OBJ is 1-based; negative = relative
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx
                    else:
                        v_idx -= 1
                    indices.append(v_idx)
                except (ValueError, IndexError):
                    print(f"[mesh_loader] WARNING: '{filepath}' line {lineno}: bad face index, skipping face.")
                    ok = False
                    break

            if not ok or len(indices) < 3:
                continue

            # Triangulate by fan from first vertex
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))

    if not vertices:
        print(f"[mesh_loader] WARNING: '{filepath}' has no vertices.")
        return None
    if not faces:
        print(f"[mesh_loader] WARNING: '{filepath}' has no usable faces.")
        return None

    verts = np.array(vertices, dtype=np.float64)
    face_normals = _compute_face_normals(verts, faces)

    print(f"[mesh_loader] Loaded '{filepath}': {len(verts)} verts, {len(faces)} tris.")
    return {
        "vertices":     verts,
        "faces":        faces,
        "face_normals": face_normals,
    }


# ---------------------------------------------------------------------------
# Normal computation
# ---------------------------------------------------------------------------

def _compute_face_normals(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
) -> np.ndarray:
    """Return unit normals (F, 3) for each triangle face."""
    n = len(faces)
    normals = np.zeros((n, 3), dtype=np.float64)

    for i, (i0, i1, i2) in enumerate(faces):
        try:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
        except IndexError:
            continue  # leave as (0,0,0) — degenerate face
        cross = np.cross(v1 - v0, v2 - v0)
        length = float(np.linalg.norm(cross))
        if length > 1e-10:
            normals[i] = cross / length

    return normals
