"""
renderer/animation.py
=====================
Frame-by-frame animation engine for orbital scene objects.

Pipeline position
-----------------
::

    list[RenderObject]  →  AnimationEngine.tick()  →  updated world_positions
                                   ↑
                          SceneState (frozen flag)

This module is *pure kinematics*: no rendering, no I/O, no threading.

Orbit model
-----------
Each :class:`~renderer.scene_builder.RenderObject` may carry an
:class:`~renderer.scene_builder.OrbitSpec` that places it on a circular,
optionally-tilted orbit around a parent object.

The angular position advances linearly with simulation time::

    ω   = speed × 2π / 60          # speed [RPM] → angular velocity [rad s⁻¹]
    θ   = t_sim × ω                 # current angle [rad]

The orbit plane is the XZ plane, optionally **inclined** by ``tilt`` radians
around the X-axis, giving a classical *orbital inclination*::

    x = parent_x  +  r · cos(θ)
    y = parent_y  +  r · sin(θ) · sin(tilt)
    z = parent_z  +  r · sin(θ) · cos(tilt)

When ``tilt = 0`` (equatorial) the orbit is flat in the XZ plane and
``y`` is constant at the parent's ``y``.

Nested chains
-------------
Chains of arbitrary depth (e.g. moon → earth → sun) are supported.
:meth:`AnimationEngine.tick` resolves parent positions before children by
processing objects in parent-first topological order, which is cached across
frames for efficiency and re-built only when the scene changes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from renderer.scene_builder import RenderObject, OrbitSpec

if TYPE_CHECKING:
    # Avoids a circular import: SceneState is only needed for type hints.
    from renderer.scene_state import SceneState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI: float = 2.0 * math.pi
_RPM_TO_RAD_S: float = _TWO_PI / 60.0   # multiply speed [RPM] to get [rad s⁻¹]


# ---------------------------------------------------------------------------
# AnimationEngine
# ---------------------------------------------------------------------------


class AnimationEngine:
    """Advances orbital animations by one time-step each frame.

    The engine is *stateful* only with respect to:

    * ``_sim_time`` — monotonically increasing simulation clock in **seconds**.
    * ``_cached_key`` / ``_processing_order`` — cached topological sort of the
      current scene's object list, invalidated whenever the set of object ids
      changes.

    The engine does **not** modify :class:`~renderer.scene_builder.OrbitSpec`
    instances; it only writes to each object's ``world_position`` array.

    Usage::

        engine = AnimationEngine()

        # Inside the renderer loop (dt in seconds, e.g. 1/25):
        engine.tick(objects, dt, scene_state)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Simulation clock – seconds elapsed since last reset.
        self._sim_time: float = 0.0

        # Cache: tuple of object ids → determines when the scene has changed.
        self._cached_key: tuple[str, ...] | None = None

        # Objects in parent-before-child topological order.
        self._processing_order: list[RenderObject] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulation clock to zero.

        Call this whenever a new scene is loaded so that all animated objects
        start at their t=0 positions.  The topological cache is intentionally
        preserved — it will be rebuilt on the next :meth:`tick` call if the
        new scene's objects differ.
        """
        self._sim_time = 0.0

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds (read-only)."""
        return self._sim_time

    def tick(
        self,
        objects: list[RenderObject],
        dt: float,
        scene_state: "SceneState",
    ) -> None:
        """Advance the simulation by *dt* seconds and update all world positions.

        If ``scene_state.frozen`` is ``True`` the simulation clock is **not**
        advanced and no positions are modified, preserving the last frame
        exactly.

        Parent-before-child ordering is guaranteed: when a moon orbits an
        earth that itself orbits a sun, the earth's ``world_position`` is
        fully updated before the moon's position is computed from it.

        Args:
            objects:     Flat list of :class:`~renderer.scene_builder.RenderObject`
                         instances produced by
                         :meth:`~renderer.scene_builder.SceneBuilder.parse_and_build`.
                         Root objects (no orbit) have their ``world_position``
                         snapped to their authored ``position``.
            dt:          Frame delta-time in **seconds**.  At 25 FPS this is
                         typically ``0.04``.  Must be non-negative.
            scene_state: Shared blackboard; only :attr:`frozen` is read.

        Raises:
            ValueError: If *dt* is negative.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")

        if scene_state.frozen:
            return

        self._sim_time += dt

        # Rebuild topological order if the scene has changed.
        key = tuple(o.id for o in objects)
        if key != self._cached_key:
            self._processing_order = _topological_sort(objects)
            self._cached_key = key

        # Build a fast id → object lookup (O(n) rebuild, O(1) lookup).
        id_map: dict[str, RenderObject] = {o.id: o for o in objects}

        # Advance positions in parent-first order so nested chains converge
        # correctly within a single tick.
        for obj in self._processing_order:
            _update_object(obj, id_map, self._sim_time)


# ---------------------------------------------------------------------------
# Per-object update
# ---------------------------------------------------------------------------


def _update_object(
    obj: RenderObject,
    id_map: dict[str, RenderObject],
    sim_time: float,
) -> None:
    """Compute and write *obj*'s ``world_position`` for *sim_time*.

    Root objects (no :class:`~renderer.scene_builder.OrbitSpec`) have their
    ``world_position`` set to their authored ``position``.

    Orbiting objects are placed using the tilted-plane orbit formula::

        ω = speed_rpm × 2π / 60
        θ = sim_time × ω
        x = parent_x + r · cos(θ)
        y = parent_y + r · sin(θ) · sin(tilt)
        z = parent_z + r · sin(θ) · cos(tilt)

    If the orbit's ``parent_id`` cannot be resolved in *id_map* (e.g. after an
    incomplete scene update), the object falls back to its authored position.

    Args:
        obj:      The object to update.
        id_map:   id → :class:`~renderer.scene_builder.RenderObject` mapping.
        sim_time: Current simulation time in seconds.
    """
    orbit: OrbitSpec | None = obj.orbit

    if orbit is None:
        # Root object: world position equals the authored rest position.
        np.copyto(obj.world_position, obj.position)
        return

    parent = id_map.get(orbit.parent_id)
    if parent is None:
        # Graceful fallback: unresolvable parent → use authored position.
        np.copyto(obj.world_position, obj.position)
        return

    # Angular velocity [rad s⁻¹] from speed [RPM].
    omega: float = orbit.speed * _RPM_TO_RAD_S
    theta: float = sim_time * omega

    r: float = orbit.radius
    tilt: float = orbit.tilt          # already in radians (converted by scene_builder)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    sin_tilt  = math.sin(tilt)
    cos_tilt  = math.cos(tilt)

    obj.world_position[0] = parent.world_position[0] + r * cos_theta
    obj.world_position[1] = parent.world_position[1] + r * sin_theta * sin_tilt
    obj.world_position[2] = parent.world_position[2] + r * sin_theta * cos_tilt


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------


def _topological_sort(objects: list[RenderObject]) -> list[RenderObject]:
    """Return *objects* sorted so every parent appears before its children.

    Uses Kahn's BFS-based algorithm.  Cycles are not expected (the scene
    builder rejects them), but any cycle survivors are appended at the end
    in original order to prevent silent data loss.

    Args:
        objects: Unsorted flat object list from SceneBuilder or a live scene.

    Returns:
        New list in parent-before-child order.
    """
    # Collect the set of ids present in this scene.
    id_set: set[str] = {o.id for o in objects}

    # In-degree: 1 if the object orbits a *known* parent, else 0.
    in_degree: dict[str, int] = {}
    children_of: dict[str, list[RenderObject]] = {o.id: [] for o in objects}

    for obj in objects:
        if obj.orbit is not None and obj.orbit.parent_id in id_set:
            in_degree[obj.id] = 1
            children_of[obj.orbit.parent_id].append(obj)
        else:
            in_degree[obj.id] = 0

    # Start with all roots (in-degree 0).
    queue: list[RenderObject] = [o for o in objects if in_degree[o.id] == 0]
    sorted_list: list[RenderObject] = []

    while queue:
        node = queue.pop(0)
        sorted_list.append(node)
        for child in children_of[node.id]:
            in_degree[child.id] -= 1
            if in_degree[child.id] == 0:
                queue.append(child)

    # Append any survivors (cycle members or orphaned objects) in stable order.
    visited_ids = {o.id for o in sorted_list}
    for obj in objects:
        if obj.id not in visited_ids:
            sorted_list.append(obj)

    return sorted_list
