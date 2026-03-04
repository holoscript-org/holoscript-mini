"""
renderer/scene_builder.py
=========================
Converts a ``scene_json`` dictionary into a typed, linked graph of
:class:`RenderObject` instances ready for the animation engine and renderer.

Pipeline position
-----------------
::

    scene_json  →  SceneBuilder.parse_and_build()  →  list[RenderObject]
                         ↓
              animation_engine (next stage)

This module is *pure data processing*: no OpenGL, no threading, no I/O.

Scene JSON schema
-----------------
.. code-block:: json

    {
      "scene_name": "My Scene",
      "background_color": "#1A1A2E",
      "objects": [
        {
          "id": "earth",
          "type": "sphere",
          "position": [0.0, 0.0, 0.0],
          "size": 1.0,
          "color": "#2E86AB",
          "emissive": false,
          "orbit": {
            "parent_id": "sun",
            "radius": 5.0,
            "speed": 1.0,
            "tilt": 0.0
          }
        }
      ]
    }

Orbit convention
----------------
*tilt* tilts the orbit plane away from the horizontal (XZ) plane, rotating
around the Z-axis.  At animation time ``t = 0`` the orbiting body starts at
angle ``0``, placing it at:

    | x = parent.x  +  radius · cos(tilt)
    | y = parent.y  +  radius · sin(tilt)
    | z = parent.z
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OBJECT_TYPES: frozenset[str] = frozenset(
    {"sphere", "cube", "cylinder", "ring", "label"}
)

_HEX_COLOR_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SceneBuildError(ValueError):
    """Raised when the scene JSON fails validation or graph construction."""


# ---------------------------------------------------------------------------
# OrbitSpec
# ---------------------------------------------------------------------------


@dataclass
class OrbitSpec:
    """Orbital parameters for objects that revolve around a parent.

    Attributes:
        parent_id: ``id`` of the :class:`RenderObject` this body orbits.
        radius:    Distance from the parent centre to this body (world units).
        speed:     Angular velocity in radians per second.  Positive values
                   rotate counter-clockwise when viewed from above.
        tilt:      Tilt of the orbit plane away from the horizontal (XZ)
                   plane, in radians.  ``0.0`` = equatorial orbit.
    """

    parent_id: str
    radius: float
    speed: float
    tilt: float

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise SceneBuildError("orbit.parent_id must be a non-empty string")
        if self.radius <= 0.0:
            raise SceneBuildError(
                f"orbit.radius must be positive, got {self.radius}"
            )


# ---------------------------------------------------------------------------
# RenderObject
# ---------------------------------------------------------------------------


class RenderObject:
    """A single renderable entity in the scene graph.

    Attributes:
        id:             Unique string identifier from the scene JSON.
        type:           One of ``VALID_OBJECT_TYPES``.
        position:       Author-specified rest position as a ``(3,)`` float64
                        array ``[x, y, z]``.  Immutable after construction.
        size:           Uniform scale from the scene JSON (positive float).
        color:          Normalised RGB colour as ``(r, g, b)`` floats in
                        ``[0.0, 1.0]``.
        emissive:       Whether the object emits light (affects shading).
        orbit:          :class:`OrbitSpec` if this object orbits a parent,
                        otherwise ``None``.
        children:       Objects that orbit *this* object (populated by
                        :class:`SceneBuilder` during graph linking).
        parent:         The :class:`RenderObject` this object orbits, or
                        ``None`` for root objects.
        world_position: Current animated world-space position as a mutable
                        ``(3,)`` float64 array.  Initialised to the t=0
                        orbit position by :class:`SceneBuilder`; updated
                        each frame by the animation engine.
    """

    __slots__ = (
        "id",
        "type",
        "position",
        "size",
        "color",
        "emissive",
        "orbit",
        "children",
        "parent",
        "world_position",
    )

    def __init__(
        self,
        *,
        id: str,
        type: str,
        position: np.ndarray,
        size: float,
        color: Tuple[float, float, float],
        emissive: bool,
        orbit: Optional[OrbitSpec],
    ) -> None:
        self.id: str = id
        self.type: str = type
        self.position: np.ndarray = position          # immutable rest position
        self.size: float = size
        self.color: Tuple[float, float, float] = color
        self.emissive: bool = emissive
        self.orbit: Optional[OrbitSpec] = orbit
        self.children: List["RenderObject"] = []
        self.parent: Optional["RenderObject"] = None
        self.world_position: np.ndarray = position.copy()  # updated by anim engine

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        wp = self.world_position
        return (
            f"RenderObject(id={self.id!r}, type={self.type!r}, "
            f"world_pos=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}), "
            f"size={self.size}, orbit={'yes' if self.orbit else 'no'})"
        )


# ---------------------------------------------------------------------------
# SceneBuilder
# ---------------------------------------------------------------------------


class SceneBuilder:
    """Stateless converter that turns a raw ``scene_json`` dict into a typed,
    linked :class:`RenderObject` graph.

    Usage::

        builder = SceneBuilder()
        objects = builder.parse_and_build(scene_json)

    The returned list preserves the ordering from ``scene_json["objects"]``
    so that the renderer can iterate predictably.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_and_build(self, scene_json: dict) -> list[RenderObject]:
        """Parse *scene_json* and return a linked list of :class:`RenderObject`.

        Steps performed:

        1. Validate the top-level structure.
        2. Parse each object entry, including hex-colour conversion and type
           validation.
        3. Build orbit relationships: resolve ``parent_id`` references,
           set bidirectional ``parent``/``children`` links, and initialise
           ``world_position`` to the t=0 orbit placement.

        Args:
            scene_json: A dictionary matching the scene JSON schema described
                in this module's docstring.

        Returns:
            List of :class:`RenderObject` instances in scene-JSON order.
            Objects that are children in an orbit hierarchy are *included* in
            the flat list as well as in their parent's ``children`` attribute.

        Raises:
            SceneBuildError: If required fields are missing, types are wrong,
                colours are malformed, ``parent_id`` references are unresolvable,
                or circular orbit dependencies are detected.
            TypeError: If *scene_json* is not a :class:`dict`.
        """
        if not isinstance(scene_json, dict):
            raise TypeError(
                f"scene_json must be a dict, got {type(scene_json).__name__!r}"
            )

        raw_objects = self._validate_top_level(scene_json)
        objects = [self._parse_object(raw, index) for index, raw in enumerate(raw_objects)]
        id_map = self._build_id_map(objects)
        self._link_orbits(objects, id_map)
        return objects

    # ------------------------------------------------------------------
    # Top-level validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_top_level(scene_json: dict) -> list:
        """Validate required top-level keys and return the raw objects list."""
        if "objects" not in scene_json:
            raise SceneBuildError("scene_json is missing required key 'objects'")
        raw_objects = scene_json["objects"]
        if not isinstance(raw_objects, list):
            raise SceneBuildError(
                f"scene_json['objects'] must be a list, got {type(raw_objects).__name__!r}"
            )
        return raw_objects

    # ------------------------------------------------------------------
    # Per-object parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_object(raw: dict, index: int) -> RenderObject:
        """Parse and validate a single raw object dict into a :class:`RenderObject`.

        Args:
            raw:   The raw dict from ``scene_json["objects"][index]``.
            index: Zero-based position; used in error messages only.

        Returns:
            A fully initialised :class:`RenderObject` (``parent`` and
            ``children`` not yet linked; ``world_position`` == ``position``).

        Raises:
            SceneBuildError: On any validation failure.
        """
        ctx = f"objects[{index}]"

        if not isinstance(raw, dict):
            raise SceneBuildError(f"{ctx}: each object must be a dict, got {type(raw).__name__!r}")

        # ---- id ----------------------------------------------------------
        obj_id = raw.get("id", "")
        if not isinstance(obj_id, str) or not obj_id.strip():
            raise SceneBuildError(f"{ctx}: 'id' must be a non-empty string")
        obj_id = obj_id.strip()

        # ---- type --------------------------------------------------------
        obj_type = raw.get("type", "")
        if obj_type not in VALID_OBJECT_TYPES:
            raise SceneBuildError(
                f"{ctx} (id={obj_id!r}): 'type' must be one of "
                f"{sorted(VALID_OBJECT_TYPES)}, got {obj_type!r}"
            )

        # ---- position ----------------------------------------------------
        position = SceneBuilder._parse_position(raw.get("position", [0.0, 0.0, 0.0]), obj_id)

        # ---- size --------------------------------------------------------
        raw_size = raw.get("size", 1.0)
        try:
            size = float(raw_size)
        except (TypeError, ValueError):
            raise SceneBuildError(
                f"{ctx} (id={obj_id!r}): 'size' must be numeric, got {raw_size!r}"
            )
        if size <= 0.0:
            raise SceneBuildError(
                f"{ctx} (id={obj_id!r}): 'size' must be positive, got {size}"
            )

        # ---- color -------------------------------------------------------
        raw_color = raw.get("color", "#FFFFFF")
        color = SceneBuilder._parse_hex_color(raw_color, obj_id)

        # ---- emissive ----------------------------------------------------
        raw_emissive = raw.get("emissive", False)
        if not isinstance(raw_emissive, bool):
            raise SceneBuildError(
                f"{ctx} (id={obj_id!r}): 'emissive' must be a bool, got {type(raw_emissive).__name__!r}"
            )

        # ---- orbit (optional) --------------------------------------------
        orbit: Optional[OrbitSpec] = None
        raw_orbit = raw.get("orbit")
        if raw_orbit is not None:
            orbit = SceneBuilder._parse_orbit(raw_orbit, obj_id)

        return RenderObject(
            id=obj_id,
            type=obj_type,
            position=position,
            size=size,
            color=color,
            emissive=raw_emissive,
            orbit=orbit,
        )

    @staticmethod
    def _parse_position(raw: object, obj_id: str) -> np.ndarray:
        """Validate and convert a raw position value to a float64 ndarray.

        Args:
            raw:    The raw value from the JSON field.
            obj_id: Object id string, used only in error messages.

        Returns:
            Shape-``(3,)`` float64 array.

        Raises:
            SceneBuildError: If *raw* is not a 3-element numeric sequence.
        """
        try:
            arr = np.array(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SceneBuildError(
                f"id={obj_id!r}: 'position' could not be converted to float array: {exc}"
            ) from exc
        if arr.shape != (3,):
            raise SceneBuildError(
                f"id={obj_id!r}: 'position' must have exactly 3 components, got shape {arr.shape}"
            )
        return arr

    @staticmethod
    def _parse_hex_color(raw: object, obj_id: str) -> Tuple[float, float, float]:
        """Convert a ``#RRGGBB`` hex string to a normalised RGB float triple.

        Args:
            raw:    The raw value from the JSON ``"color"`` field.
            obj_id: Used in error messages only.

        Returns:
            ``(r, g, b)`` where each component is in ``[0.0, 1.0]``.

        Raises:
            SceneBuildError: If *raw* is not a valid ``#RRGGBB`` string.
        """
        if not isinstance(raw, str):
            raise SceneBuildError(
                f"id={obj_id!r}: 'color' must be a '#RRGGBB' string, got {type(raw).__name__!r}"
            )
        match = _HEX_COLOR_RE.match(raw)
        if not match:
            raise SceneBuildError(
                f"id={obj_id!r}: 'color' must be a '#RRGGBB' hex string, got {raw!r}"
            )
        hex_digits = match.group(1)
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
        return (r, g, b)

    @staticmethod
    def _parse_orbit(raw: object, obj_id: str) -> OrbitSpec:
        """Parse and validate a raw orbit dict into an :class:`OrbitSpec`.

        Args:
            raw:    The raw value of the ``"orbit"`` field.
            obj_id: Used in error messages only.

        Returns:
            A validated :class:`OrbitSpec` instance.

        Raises:
            SceneBuildError: On missing fields, wrong types, or invalid values.
        """
        if not isinstance(raw, dict):
            raise SceneBuildError(
                f"id={obj_id!r}: 'orbit' must be a dict, got {type(raw).__name__!r}"
            )

        # parent_id
        parent_id = raw.get("parent_id", "")
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise SceneBuildError(
                f"id={obj_id!r}: orbit.parent_id must be a non-empty string"
            )

        # radius
        raw_radius = raw.get("radius")
        if raw_radius is None:
            raise SceneBuildError(f"id={obj_id!r}: orbit.radius is required")
        try:
            radius = float(raw_radius)
        except (TypeError, ValueError) as exc:
            raise SceneBuildError(
                f"id={obj_id!r}: orbit.radius must be numeric: {exc}"
            ) from exc
        if radius <= 0.0:
            raise SceneBuildError(
                f"id={obj_id!r}: orbit.radius must be positive, got {radius}"
            )

        # speed
        raw_speed = raw.get("speed", 0.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError) as exc:
            raise SceneBuildError(
                f"id={obj_id!r}: orbit.speed must be numeric: {exc}"
            ) from exc

        # tilt  (degrees in JSON → radians internally)
        raw_tilt = raw.get("tilt", 0.0)
        try:
            tilt_deg = float(raw_tilt)
        except (TypeError, ValueError) as exc:
            raise SceneBuildError(
                f"id={obj_id!r}: orbit.tilt must be numeric: {exc}"
            ) from exc
        tilt_rad = math.radians(tilt_deg)

        return OrbitSpec(
            parent_id=parent_id.strip(),
            radius=radius,
            speed=speed,
            tilt=tilt_rad,
        )

    # ------------------------------------------------------------------
    # Graph linking
    # ------------------------------------------------------------------

    @staticmethod
    def _build_id_map(objects: list[RenderObject]) -> dict[str, RenderObject]:
        """Build an id → object lookup dict, checking for duplicate ids.

        Args:
            objects: Flat list of freshly parsed :class:`RenderObject` instances.

        Returns:
            Dict mapping each object's ``id`` to the object.

        Raises:
            SceneBuildError: If any two objects share the same ``id``.
        """
        id_map: dict[str, RenderObject] = {}
        for obj in objects:
            if obj.id in id_map:
                raise SceneBuildError(
                    f"Duplicate object id {obj.id!r} found in scene"
                )
            id_map[obj.id] = obj
        return id_map

    @staticmethod
    def _link_orbits(
        objects: list[RenderObject],
        id_map: dict[str, RenderObject],
    ) -> None:
        """Resolve orbit ``parent_id`` references and set parent/child links.

        For every object that has an :class:`OrbitSpec`, this method:

        * Looks up the parent by ``parent_id``.
        * Sets ``obj.parent`` and appends ``obj`` to ``parent.children``.
        * Computes ``obj.world_position`` for animation time ``t = 0``.

        Orbit position at t=0 (angle θ=0, tilted plane)::

            x = parent.world_position[0] + radius · cos(tilt)
            y = parent.world_position[1] + radius · sin(tilt)
            z = parent.world_position[2]

        Args:
            objects: All :class:`RenderObject` instances in scene order.
            id_map:  id → object mapping produced by :meth:`_build_id_map`.

        Raises:
            SceneBuildError: If a ``parent_id`` cannot be resolved or a
                circular orbit dependency is detected.
        """
        # Topological sort so parents are linked before children.
        # Build a dependency graph first.
        for obj in objects:
            if obj.orbit is None:
                continue  # root object – world_position already == position

            parent_id = obj.orbit.parent_id
            if parent_id == obj.id:
                raise SceneBuildError(
                    f"Object {obj.id!r} cannot orbit itself"
                )
            if parent_id not in id_map:
                raise SceneBuildError(
                    f"Object {obj.id!r} references unknown parent_id {parent_id!r}"
                )

        # Detect cycles via DFS before committing any mutations.
        SceneBuilder._check_orbit_cycles(objects, id_map)

        # Process in topological order so parent.world_position is set
        # before we compute the child's world_position.
        processing_order = SceneBuilder._topological_sort(objects, id_map)

        for obj in processing_order:
            if obj.orbit is None:
                continue  # world_position == position (set in __init__)

            parent = id_map[obj.orbit.parent_id]
            obj.parent = parent
            parent.children.append(obj)

            # t=0 world position in the tilted orbit plane
            r = obj.orbit.radius
            tilt = obj.orbit.tilt
            obj.world_position = np.array(
                [
                    parent.world_position[0] + r * math.cos(tilt),
                    parent.world_position[1] + r * math.sin(tilt),
                    parent.world_position[2],
                ],
                dtype=np.float64,
            )

    @staticmethod
    def _check_orbit_cycles(
        objects: list[RenderObject],
        id_map: dict[str, RenderObject],
    ) -> None:
        """Raise :class:`SceneBuildError` if any cycle exists in the orbit graph.

        Uses iterative DFS with three-colour marking (white/grey/black).

        Args:
            objects: All objects in the scene.
            id_map:  id → object mapping.

        Raises:
            SceneBuildError: If a cycle is detected.
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {obj.id: WHITE for obj in objects}

        def visit(start_id: str) -> None:
            stack: list[tuple[str, bool]] = [(start_id, False)]
            while stack:
                node_id, leaving = stack.pop()
                if leaving:
                    colour[node_id] = BLACK
                    continue
                if colour[node_id] == BLACK:
                    continue
                if colour[node_id] == GREY:
                    raise SceneBuildError(
                        f"Circular orbit dependency detected involving object {node_id!r}"
                    )
                colour[node_id] = GREY
                stack.append((node_id, True))  # schedule back-edge marking
                obj = id_map[node_id]
                if obj.orbit is not None and obj.orbit.parent_id in id_map:
                    stack.append((obj.orbit.parent_id, False))

        for obj in objects:
            if colour[obj.id] == WHITE:
                visit(obj.id)

    @staticmethod
    def _topological_sort(
        objects: list[RenderObject],
        id_map: dict[str, RenderObject],
    ) -> list[RenderObject]:
        """Return *objects* sorted so every parent appears before its children.

        A standard iterative Kahn's algorithm is used.

        Args:
            objects: All objects in the scene (cycle-free, pre-validated).
            id_map:  id → object mapping.

        Returns:
            New list with parents preceding children.
        """
        # in-degree: number of orbit-parents each object has (0 or 1)
        in_degree: dict[str, int] = {obj.id: 0 for obj in objects}
        for obj in objects:
            if obj.orbit is not None:
                in_degree[obj.id] += 1

        # Start with roots (objects with no orbit parent)
        queue: list[RenderObject] = [obj for obj in objects if in_degree[obj.id] == 0]
        sorted_order: list[RenderObject] = []

        # Build adjacency: parent_id → list of children
        children_map: dict[str, list[RenderObject]] = {obj.id: [] for obj in objects}
        for obj in objects:
            if obj.orbit is not None:
                children_map[obj.orbit.parent_id].append(obj)

        while queue:
            node = queue.pop(0)
            sorted_order.append(node)
            for child in children_map[node.id]:
                in_degree[child.id] -= 1
                if in_degree[child.id] == 0:
                    queue.append(child)

        return sorted_order
