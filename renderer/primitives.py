"""renderer/primitives.py
One draw function per object type for the HoloScript Renderer.
"""

from __future__ import annotations

import math
import numpy as np
from OpenGL.GL import (
    glPushMatrix,
    glPopMatrix,
    glTranslatef,
    glScalef,
    glColor3f,
    glColor4f,
    glEnable,
    glDisable,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    glBegin,
    glEnd,
    glVertex3f,
    glNormal3f,
    glRotatef,
    GL_QUADS,
    GL_TRIANGLES,
    GL_TRIANGLE_STRIP,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    glBlendFunc,
    glMatrixMode,
    glLoadIdentity,
    glOrtho,
    GL_PROJECTION,
    GL_MODELVIEW,
    glGetDoublev,
    glGetIntegerv,
    GL_MODELVIEW_MATRIX,
    GL_PROJECTION_MATRIX,
    GL_VIEWPORT,
)
from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric, gluSphere, gluCylinder, gluDisk, gluProject

from renderer.scene_parser import SceneObject
from renderer.mesh_loader import get_mesh

# ---------------------------------------------------------------------------
# Label cache — pyglet Label objects keyed by display text.
# Created once, repositioned each frame.
# ---------------------------------------------------------------------------

_label_cache: dict[str, object] = {}   # str → pyglet.text.Label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp_color(color: list[float] | tuple[float, float, float], scale: float = 1.0) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, float(c) * scale)) for c in color[:3])


def _rotate_z_to_vector(direction: np.ndarray) -> None:
    """Rotate local +Z axis to align with direction."""
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return

    target = direction / norm
    source = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 0.999999:
        return
    if dot < -0.999999:
        glRotatef(180.0, 1.0, 0.0, 0.0)
        return

    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-8:
        return
    axis = axis / axis_norm
    angle = math.degrees(math.acos(dot))
    glRotatef(angle, float(axis[0]), float(axis[1]), float(axis[2]))


def _get_band_color(bands: list, lat_frac: float) -> list[float]:
    """Return the RGB color for a given latitude fraction."""
    if not bands:
        return [1.0, 1.0, 1.0]

    parsed_bands: list[tuple[list[float], float]] = []
    for entry in bands:
        if not isinstance(entry, dict):
            continue
        color = entry.get("color")
        width = entry.get("width", 0.0)
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        try:
            parsed_color = [float(color[0]), float(color[1]), float(color[2])]
            parsed_width = float(width)
        except (TypeError, ValueError):
            continue
        if parsed_width <= 0.0:
            continue
        parsed_bands.append((parsed_color, parsed_width))

    if not parsed_bands:
        return [1.0, 1.0, 1.0]

    total_width = sum(width for _color, width in parsed_bands)
    if total_width <= 0.0:
        return [1.0, 1.0, 1.0]

    pos = max(0.0, min(1.0, float(lat_frac))) * total_width
    if pos >= total_width:
        pos = total_width - 1e-9

    cumulative = 0.0
    for color, width in parsed_bands:
        cumulative += width
        if pos <= cumulative:
            return color

    return parsed_bands[-1][0]


def _draw_banded_sphere(obj: SceneObject) -> None:
    """Draw a sphere with horizontal latitude bands using triangle strips."""
    glEnable(GL_LIGHTING)

    r = float(obj.size)
    n_stacks = 32
    n_slices = 32

    for i in range(n_stacks):
        phi1 = math.pi * i / n_stacks
        phi2 = math.pi * (i + 1) / n_stacks
        lat_frac = i / n_stacks
        color = _get_band_color(obj.bands, lat_frac)
        glColor3f(float(color[0]), float(color[1]), float(color[2]))

        glBegin(GL_TRIANGLE_STRIP)
        for j in range(n_slices + 1):
            theta = 2.0 * math.pi * j / n_slices

            x1 = math.sin(phi1) * math.cos(theta)
            y1 = math.cos(phi1)
            z1 = math.sin(phi1) * math.sin(theta)

            x2 = math.sin(phi2) * math.cos(theta)
            y2 = math.cos(phi2)
            z2 = math.sin(phi2) * math.sin(theta)

            glNormal3f(x1, y1, z1)
            glVertex3f(x1 * r, y1 * r, z1 * r)
            glNormal3f(x2, y2, z2)
            glVertex3f(x2 * r, y2 * r, z2 * r)
        glEnd()


def _draw_polar_disk(y_offset: float, radius: float, color: tuple[float, float, float], facing_up: bool) -> None:
    glPushMatrix()
    glTranslatef(0.0, y_offset, 0.0)
    glRotatef(-90.0 if facing_up else 90.0, 1.0, 0.0, 0.0)
    glColor3f(*color)
    quadric = gluNewQuadric()
    gluDisk(quadric, 0.0, radius, 24, 1)
    gluDeleteQuadric(quadric)
    glPopMatrix()


def _draw_earth_sphere(obj: SceneObject) -> None:
    """Draw Earth with ocean base, continent patches, and white polar caps."""
    glEnable(GL_LIGHTING)

    quadric = gluNewQuadric()
    glColor3f(*_clamp_color(obj.color))
    gluSphere(quadric, obj.size, 48, 48)

    continent_color = _clamp_color(obj.secondary_color)
    glColor3f(*continent_color)

    continents = [
        (-10.0, -80.0, 0.18),
        (45.0, -100.0, 0.15),
        (50.0, 60.0, 0.22),
        (0.0, 20.0, 0.14),
        (-25.0, 135.0, 0.08),
    ]

    r = float(obj.size)
    for lat_deg, lon_deg, factor in continents:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        cx = r * math.cos(lat) * math.cos(lon)
        cy = r * math.sin(lat)
        cz = r * math.cos(lat) * math.sin(lon)
        patch_size = r * factor

        center = np.array([cx, cy, cz], dtype=float)
        patch_center = center * 1.002
        normal = center / max(float(np.linalg.norm(center)), 1e-8)

        glPushMatrix()
        glTranslatef(float(patch_center[0]), float(patch_center[1]), float(patch_center[2]))
        _rotate_z_to_vector(normal)
        gluDisk(quadric, 0.0, patch_size, 16, 1)
        glPopMatrix()

    cap_radius = r * 0.34
    cap_offset = r * 0.94
    _draw_polar_disk(cap_offset, cap_radius, (1.0, 1.0, 1.0), True)
    _draw_polar_disk(-cap_offset, cap_radius, (1.0, 1.0, 1.0), False)
    gluDeleteQuadric(quadric)


def _draw_polar_caps(obj: SceneObject) -> None:
    """Draw sphere with polar cap disks at top and bottom."""
    glEnable(GL_LIGHTING)

    quadric = gluNewQuadric()
    glColor3f(*_clamp_color(obj.color))
    gluSphere(quadric, obj.size, 32, 32)

    cap_color = _clamp_color(obj.secondary_color)
    _draw_polar_disk(float(obj.size) * 0.85, float(obj.size) * 0.35, cap_color, True)
    _draw_polar_disk(float(obj.size) * -0.85, float(obj.size) * 0.35, cap_color, False)
    gluDeleteQuadric(quadric)


def _draw_saturn_rings(obj: SceneObject) -> None:
    """Draw Saturn as a banded sphere with a tilted, blended ring system."""
    _draw_banded_sphere(obj)

    ring = obj.ring or {}
    inner_r = float(obj.size) * float(ring.get("inner_radius_factor", 1.3))
    outer_r = float(obj.size) * float(ring.get("outer_radius_factor", 2.4))
    ring_color = ring.get("color", [0.8, 0.72, 0.55])
    opacity = float(ring.get("opacity", 0.75))

    base = _clamp_color(ring_color)
    bright = _clamp_color(ring_color, 1.1)
    dark = _clamp_color(ring_color, 0.6)

    quadric = gluNewQuadric()
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)

    glPushMatrix()
    glRotatef(27.0, 0.0, 0.0, 1.0)

    zone_c_inner = float(obj.size) * 1.1
    zone_c_outer = inner_r
    if zone_c_outer > zone_c_inner:
        glColor4f(dark[0], dark[1], dark[2], opacity * 0.4)
        gluDisk(quadric, zone_c_inner, zone_c_outer, 64, 1)

    zone_b_inner = inner_r
    zone_b_outer = inner_r * 1.35
    if zone_b_outer > zone_b_inner:
        glColor4f(base[0], base[1], base[2], opacity)
        gluDisk(quadric, zone_b_inner, zone_b_outer, 64, 1)

    zone_a_inner = max(outer_r * 0.75, inner_r * 1.5)
    zone_a_outer = outer_r
    if zone_a_outer > zone_a_inner:
        glColor4f(bright[0], bright[1], bright[2], opacity)
        gluDisk(quadric, zone_a_inner, zone_a_outer, 64, 1)

    glPopMatrix()

    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)
    gluDeleteQuadric(quadric)


def _draw_emissive_glow(obj: SceneObject) -> None:
    """Draw emissive sun body plus transparent corona layers."""
    quadric = gluNewQuadric()

    glDisable(GL_LIGHTING)
    glColor3f(*_clamp_color(obj.color))
    gluSphere(quadric, obj.size, 64, 64)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    corona_color = _clamp_color(obj.secondary_color)
    layers = [
        (float(obj.size) * 1.05, 0.35),
        (float(obj.size) * 1.12, 0.18),
        (float(obj.size) * 1.22, 0.08),
    ]
    for radius, alpha in layers:
        glColor4f(corona_color[0], corona_color[1], corona_color[2], alpha)
        gluSphere(quadric, radius, 32, 32)

    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)
    gluDeleteQuadric(quadric)


def _setup_color(obj: SceneObject) -> None:
    """Set material color and toggle lighting for emissive vs. lit objects."""
    if obj.emissive:
        glDisable(GL_LIGHTING)
        glColor3f(*obj.color)
    else:
        glEnable(GL_LIGHTING)
        glColor3f(*obj.color)


def _restore_lighting(obj: SceneObject) -> None:
    """Re-enable lighting after drawing an emissive object."""
    if obj.emissive:
        glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------
# draw_sphere
# ---------------------------------------------------------------------------

def draw_sphere(obj: SceneObject) -> None:
    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )

    style = getattr(obj, "surface_style", "plain")

    if style == "banded":
        _draw_banded_sphere(obj)
    elif style == "earth":
        _draw_earth_sphere(obj)
    elif style == "polar_caps":
        _draw_polar_caps(obj)
    elif style == "saturn_rings":
        _draw_saturn_rings(obj)
    elif style == "emissive_glow":
        _draw_emissive_glow(obj)
    else:
        if obj.emissive:
            glDisable(GL_LIGHTING)
            glColor3f(*obj.color)
        else:
            glEnable(GL_LIGHTING)
            glColor3f(*obj.color)
        quadric = gluNewQuadric()
        gluSphere(quadric, obj.size, 32, 32)
        gluDeleteQuadric(quadric)
        if obj.emissive:
            glEnable(GL_LIGHTING)

    glPopMatrix()


# ---------------------------------------------------------------------------
# draw_cube
# ---------------------------------------------------------------------------

def draw_cube(obj: SceneObject) -> None:
    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )
    _setup_color(obj)

    s = obj.size
    glBegin(GL_QUADS)

    # Front (+Z)
    glNormal3f(0, 0, 1)
    glVertex3f(-s, -s, s)
    glVertex3f(s, -s, s)
    glVertex3f(s, s, s)
    glVertex3f(-s, s, s)

    # Back (-Z)
    glNormal3f(0, 0, -1)
    glVertex3f(s, -s, -s)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s, s, -s)
    glVertex3f(s, s, -s)

    # Left (-X)
    glNormal3f(-1, 0, 0)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s, -s, s)
    glVertex3f(-s, s, s)
    glVertex3f(-s, s, -s)

    # Right (+X)
    glNormal3f(1, 0, 0)
    glVertex3f(s, -s, s)
    glVertex3f(s, -s, -s)
    glVertex3f(s, s, -s)
    glVertex3f(s, s, s)

    # Top (+Y)
    glNormal3f(0, 1, 0)
    glVertex3f(-s, s, s)
    glVertex3f(s, s, s)
    glVertex3f(s, s, -s)
    glVertex3f(-s, s, -s)

    # Bottom (-Y)
    glNormal3f(0, -1, 0)
    glVertex3f(-s, -s, -s)
    glVertex3f(s, -s, -s)
    glVertex3f(s, -s, s)
    glVertex3f(-s, -s, s)

    glEnd()
    _restore_lighting(obj)
    glPopMatrix()


# ---------------------------------------------------------------------------
# draw_cylinder
# ---------------------------------------------------------------------------

def draw_cylinder(obj: SceneObject) -> None:
    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )
    _setup_color(obj)

    radius = obj.size
    height = obj.size * 2.0

    # Center cylinder vertically around world_position
    glTranslatef(0.0, -height / 2.0, 0.0)

    quadric = gluNewQuadric()
    # Body
    gluCylinder(quadric, radius, radius, height, 32, 1)
    # Bottom cap
    gluDisk(quadric, 0, radius, 32, 1)
    # Top cap
    glTranslatef(0.0, height, 0.0)
    gluDisk(quadric, 0, radius, 32, 1)
    gluDeleteQuadric(quadric)

    _restore_lighting(obj)
    glPopMatrix()


# ---------------------------------------------------------------------------
# draw_ring
# ---------------------------------------------------------------------------

def draw_ring(obj: SceneObject) -> None:
    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )
    _setup_color(obj)

    outer_radius = obj.size
    inner_radius = obj.size * 0.5

    quadric = gluNewQuadric()
    gluDisk(quadric, inner_radius, outer_radius, 64, 1)
    gluDeleteQuadric(quadric)

    _restore_lighting(obj)
    glPopMatrix()


# ---------------------------------------------------------------------------
# draw_label  (type="label" scene objects — no geometry; text handled by draw_labels_2d)
# ---------------------------------------------------------------------------

def draw_label(obj: SceneObject) -> None:
    pass  # text is rendered as a 2D overlay by draw_labels_2d()


# ---------------------------------------------------------------------------
# draw_labels_2d — billboard text overlay for all labeled scene objects
# ---------------------------------------------------------------------------

def draw_labels_2d(
    label_queue: list[tuple[str, np.ndarray, float]],
    window_w: int,
    window_h: int,
) -> None:
    """Project 3D label positions to screen coords and draw text as a 2D overlay.

    Args:
        label_queue: list of (text, world_pos, y_offset) collected during the
                     3D draw pass.  y_offset = obj.size (or obj.scale for meshes)
                     so the label floats above the object surface.
        window_w/h:  current window dimensions in pixels.

    Must be called while the 3D modelview/projection matrices are still active
    (before pyglet swaps the buffer).
    """
    if not label_queue:
        return

    import pyglet.text  # deferred — avoids import at module load time

    # Snapshot current GL matrices (set by _setup_camera + apply_global)
    mv   = np.zeros(16, dtype=np.float64)
    proj = np.zeros(16, dtype=np.float64)
    vp   = np.zeros(4,  dtype=np.int32)
    glGetDoublev(GL_MODELVIEW_MATRIX, mv)
    glGetDoublev(GL_PROJECTION_MATRIX, proj)
    glGetIntegerv(GL_VIEWPORT, vp)

    # Project every label's world position to 2D screen coords
    screen_items: list[tuple[str, int, int]] = []
    for text, world_pos, y_offset in label_queue:
        px = float(world_pos[0])
        py = float(world_pos[1]) + float(y_offset) + 0.4
        pz = float(world_pos[2])
        try:
            sx, sy, sz = gluProject(px, py, pz, mv, proj, vp)
        except Exception:
            continue
        if not (0.0 < sz < 1.0):
            continue  # behind camera or beyond far plane — skip
        screen_items.append((text, int(sx), int(sy)))

    if not screen_items:
        return

    # --- 2D overlay pass ---
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, window_w, 0, window_h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    for text, sx, sy in screen_items:
        if text not in _label_cache:
            _label_cache[text] = pyglet.text.Label(
                text,
                font_name="Arial",
                font_size=9,
                weight="normal",
                anchor_x="center",
                anchor_y="bottom",
                color=(255, 255, 255, 210),
            )
        lbl = _label_cache[text]
        lbl.x = sx
        lbl.y = sy
        lbl.draw()

    # Restore 3D matrices and GL state
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------
# draw_mesh
# ---------------------------------------------------------------------------

def draw_mesh(obj: SceneObject) -> None:
    mesh = get_mesh(obj.mesh_file)
    if mesh is None:
        return  # already warned by loader

    vertices     = mesh["vertices"]
    faces        = mesh["faces"]
    face_normals = mesh["face_normals"]

    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )
    s = float(obj.scale)
    glScalef(s, s, s)

    glEnable(GL_LIGHTING)
    glColor3f(*_clamp_color(obj.color))

    glBegin(GL_TRIANGLES)
    for face_idx, (i0, i1, i2) in enumerate(faces):
        nx, ny, nz = face_normals[face_idx]
        glNormal3f(float(nx), float(ny), float(nz))
        v = vertices[i0]; glVertex3f(float(v[0]), float(v[1]), float(v[2]))
        v = vertices[i1]; glVertex3f(float(v[0]), float(v[1]), float(v[2]))
        v = vertices[i2]; glVertex3f(float(v[0]), float(v[1]), float(v[2]))
    glEnd()

    glPopMatrix()


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def dispatch(obj: SceneObject) -> None:
    """Call the correct draw function based on obj.type."""
    if obj.type == "sphere":
        draw_sphere(obj)
    elif obj.type == "cube":
        draw_cube(obj)
    elif obj.type == "cylinder":
        draw_cylinder(obj)
    elif obj.type == "ring":
        draw_ring(obj)
    elif obj.type == "label":
        draw_label(obj)
    elif obj.type == "mesh":
        draw_mesh(obj)
    else:
        print(f"[primitives] WARNING: unknown object type {obj.type!r} for id={obj.id!r}, skipping.")
