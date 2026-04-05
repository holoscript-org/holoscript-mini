"""renderer/primitives.py
One draw function per object type for the HoloScript Renderer.
No pyglet import here — only OpenGL/GLU and numpy.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    glPushMatrix, glPopMatrix,
    glTranslatef, glColor3f,
    glEnable, glDisable,
    GL_LIGHTING,
    glBegin, glEnd, glVertex3f, glNormal3f,
    GL_QUADS,
)
from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric, gluSphere, gluCylinder, gluDisk

from renderer.scene_parser import SceneObject


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    _setup_color(obj)
    quadric = gluNewQuadric()
    gluSphere(quadric, obj.size, 32, 32)
    gluDeleteQuadric(quadric)
    _restore_lighting(obj)
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
    glVertex3f(-s, -s,  s)
    glVertex3f( s, -s,  s)
    glVertex3f( s,  s,  s)
    glVertex3f(-s,  s,  s)

    # Back (-Z)
    glNormal3f(0, 0, -1)
    glVertex3f( s, -s, -s)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s,  s, -s)
    glVertex3f( s,  s, -s)

    # Left (-X)
    glNormal3f(-1, 0, 0)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s, -s,  s)
    glVertex3f(-s,  s,  s)
    glVertex3f(-s,  s, -s)

    # Right (+X)
    glNormal3f(1, 0, 0)
    glVertex3f( s, -s,  s)
    glVertex3f( s, -s, -s)
    glVertex3f( s,  s, -s)
    glVertex3f( s,  s,  s)

    # Top (+Y)
    glNormal3f(0, 1, 0)
    glVertex3f(-s,  s,  s)
    glVertex3f( s,  s,  s)
    glVertex3f( s,  s, -s)
    glVertex3f(-s,  s, -s)

    # Bottom (-Y)
    glNormal3f(0, -1, 0)
    glVertex3f(-s, -s, -s)
    glVertex3f( s, -s, -s)
    glVertex3f( s, -s,  s)
    glVertex3f(-s, -s,  s)

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
# draw_label
# ---------------------------------------------------------------------------

def draw_label(obj: SceneObject) -> None:
    if obj.size == 0.0:
        return  # no geometry for zero-size labels

    # Placeholder: tiny white sphere until billboard text is implemented
    glPushMatrix()
    glTranslatef(
        float(obj.world_position[0]),
        float(obj.world_position[1]),
        float(obj.world_position[2]),
    )
    _setup_color(obj)
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.1, 8, 8)
    gluDeleteQuadric(quadric)
    _restore_lighting(obj)
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
    else:
        print(f"[primitives] WARNING: unknown object type {obj.type!r} for id={obj.id!r}, skipping.")
