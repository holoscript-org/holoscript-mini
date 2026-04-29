"""llm/parametric_generator.py

Stage 2 of the Member 1 pipeline: convert an abstract ScenePlan into concrete
positions, rotations, scales, and hierarchy hints using deterministic math.

This stage does NOT build final renderer JSON. It only computes the geometry
plan that a later builder stage can convert into strict scene objects.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from math import ceil, cos, pi, sin, sqrt
from typing import Any

from core.utils.logger import get_logger
from llm.planner import ScenePlan

logger = get_logger("parametric_generator")

GOLDEN_ANGLE = pi * (3.0 - sqrt(5.0))
DEFAULT_SCALE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class PlacementSpec:
    """Concrete placement for a single planned scene element."""

    role: str
    index: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = DEFAULT_SCALE
    parent: str | None = None
    orbit_center: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParametricScene:
    """Output of the parametric stage."""

    scene_type: str
    placements: list[PlacementSpec]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "placements": [placement.to_dict() for placement in self.placements],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (float(x), float(y), float(z))


def _circle_point(radius: float, angle: float, y: float = 0.0) -> tuple[float, float, float]:
    return _vec(radius * cos(angle), y, radius * sin(angle))


def _grid_position(index: int, count: int, spacing: float = 2.8) -> tuple[float, float, float]:
    cols = max(1, ceil(sqrt(max(1, count))))
    row = index // cols
    col = index % cols
    x = (col - (cols - 1) / 2.0) * spacing
    z = (row - (ceil(count / cols) - 1) / 2.0) * spacing
    return _vec(x, 0.0, z)


def _ordered_roles(plan: ScenePlan) -> list[tuple[str, int]]:
    """Expand unique components and repeat counts into deterministic roles."""
    roles: list[tuple[str, int]] = []
    seen: set[str] = set()

    for component in plan.components:
        count = max(1, int(plan.repeat_counts.get(component, 1)))
        seen.add(component)
        if count == 1:
            roles.append((component, 0))
        else:
            for idx in range(count):
                roles.append((f"{component}_{idx}", idx))

    for component, count in plan.repeat_counts.items():
        if component in seen:
            continue
        count = max(1, int(count))
        if count == 1:
            roles.append((component, 0))
        else:
            for idx in range(count):
                roles.append((f"{component}_{idx}", idx))

    return roles


def _find_roles(placements: list[PlacementSpec], prefix: str) -> list[PlacementSpec]:
    return [placement for placement in placements if placement.role == prefix or placement.role.startswith(f"{prefix}_")]


# ---------------------------------------------------------------------------
# Scene-specific builders
# ---------------------------------------------------------------------------

def _build_atom(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)

    core_placed = False
    orbit_index = 0

    for role, idx in roles:
        lower = role.lower()
        if not core_placed and any(token in lower for token in ("proton", "nucleus", "core", "sun", "center", "oxygen")):
            placements.append(
                PlacementSpec(
                    role=role,
                    index=idx,
                    position=_vec(0.0, 0.0, 0.0),
                    metadata={"kind": "core"},
                )
            )
            core_placed = True
            continue

        if "ring" in lower:
            placements.append(
                PlacementSpec(
                    role=role,
                    index=idx,
                    position=_vec(0.0, 0.0, 0.0),
                    rotation=_vec(pi / 2.0, 0.0, 0.0),
                    metadata={"kind": "ring", "radius": 3.0},
                )
            )
            continue

        radius = 2.8 + orbit_index * 0.45
        angle = orbit_index * GOLDEN_ANGLE
        y = 0.15 if orbit_index % 2 else -0.15
        placements.append(
            PlacementSpec(
                role=role,
                index=idx,
                position=_circle_point(radius, angle, y=y),
                orbit_center=_vec(0.0, 0.0, 0.0),
                metadata={"kind": "orbiting", "orbit_radius": radius},
            )
        )
        orbit_index += 1

    if not core_placed:
        placements.insert(
            0,
            PlacementSpec(role="core", index=0, position=_vec(0.0, 0.0, 0.0), metadata={"kind": "core"}),
        )

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["atom layout"])


def _build_molecule(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)

    # Prefer a centered oxygen-like core if available.
    for role, idx in roles:
        if role.lower().startswith(("oxygen", "core", "center", "nucleus")):
            placements.append(
                PlacementSpec(role=role, index=idx, position=_vec(0.0, 0.0, 0.0), metadata={"kind": "core"})
            )
            break
    else:
        if roles:
            role, idx = roles[0]
            placements.append(
                PlacementSpec(role=role, index=idx, position=_vec(0.0, 0.0, 0.0), metadata={"kind": "core"})
            )

    # Water-like bent geometry is a strong default for molecules.
    hydrogens = [role for role, _ in roles if "hydrogen" in role.lower()]
    if len(hydrogens) >= 2:
        angle = 52.25 * (pi / 180.0)
        radius = 1.15
        placements.append(
            PlacementSpec(
                role=hydrogens[0],
                index=0,
                position=_vec(radius * cos(angle), radius * sin(angle), 0.0),
                orbit_center=_vec(0.0, 0.0, 0.0),
                metadata={"kind": "bonded"},
            )
        )
        placements.append(
            PlacementSpec(
                role=hydrogens[1],
                index=1,
                position=_vec(-radius * cos(angle), radius * sin(angle), 0.0),
                orbit_center=_vec(0.0, 0.0, 0.0),
                metadata={"kind": "bonded"},
            )
        )
        for role, idx in roles:
            if role in hydrogens[:2] or role.lower().startswith("oxygen") or role.lower().startswith("core"):
                continue
            placements.append(
                PlacementSpec(
                    role=role,
                    index=idx,
                    position=_circle_point(2.0 + idx * 0.2, idx * GOLDEN_ANGLE, y=0.0),
                    orbit_center=_vec(0.0, 0.0, 0.0),
                    metadata={"kind": "auxiliary"},
                )
            )
    else:
        # Generic molecule: distribute around a tight ring.
        count = max(1, len(roles))
        for i, (role, idx) in enumerate(roles):
            if role.lower().startswith(("oxygen", "core", "center", "nucleus")):
                continue
            placements.append(
                PlacementSpec(
                    role=role,
                    index=idx,
                    position=_circle_point(1.25 + i * 0.15, i * (2.0 * pi / max(1, count - 1)), y=0.0),
                    orbit_center=_vec(0.0, 0.0, 0.0),
                    metadata={"kind": "bonded"},
                )
            )

    # Pad with small support elements if the plan asked for more objects than
    # were explicitly described by the role set.
    filler_index = 0
    while len(placements) < plan.num_objects:
        placements.append(
            PlacementSpec(
                role=f"support_{filler_index}",
                index=filler_index,
                position=_vec(0.0, 0.35 * (filler_index + 1), 0.75 + 0.15 * filler_index),
                metadata={"kind": "support"},
            )
        )
        filler_index += 1

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["molecule layout"])


def _build_solar_system(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)

    planet_count = max(1, int(plan.repeat_counts.get("planet", 0)) or 0)
    moon_count = max(0, int(plan.repeat_counts.get("moon", 0)) or 0)

    sun_role = next((role for role, _ in roles if role.lower().startswith(("sun", "core", "center"))), "sun")
    placements.append(PlacementSpec(role=sun_role, index=0, position=_vec(0.0, 0.0, 0.0), metadata={"kind": "star"}))

    planet_roles = [role for role, _ in roles if role.lower().startswith("planet") or role.lower() in {"earth", "mars", "venus", "mercury", "jupiter", "saturn", "uranus", "neptune"}]
    if not planet_roles and planet_count:
        planet_roles = [f"planet_{i}" for i in range(planet_count)]
    elif planet_count and len(planet_roles) < planet_count:
        planet_roles.extend(f"planet_{i}" for i in range(len(planet_roles), planet_count))

    planet_positions: list[tuple[str, tuple[float, float, float]]] = []
    for i, role in enumerate(planet_roles):
        radius = 4.0 + i * 2.35
        angle = i * 0.55
        position = _circle_point(radius, angle, y=0.0)
        planet_positions.append((role, position))
        placements.append(
            PlacementSpec(
                role=role,
                index=i,
                position=position,
                orbit_center=_vec(0.0, 0.0, 0.0),
                parent=sun_role if plan.hierarchy_needed else None,
                metadata={"kind": "planet", "orbit_radius": radius},
            )
        )

    moon_roles = [role for role, _ in roles if role.lower().startswith("moon")]
    if moon_count and len(moon_roles) < moon_count:
        moon_roles.extend(f"moon_{i}" for i in range(len(moon_roles), moon_count))

    for i, role in enumerate(moon_roles):
        parent_role, parent_pos = planet_positions[i % max(1, len(planet_positions))]
        moon_radius = 0.7 + 0.05 * i
        moon_angle = i * 1.9
        moon_position = _vec(
            parent_pos[0] + moon_radius * cos(moon_angle),
            parent_pos[1] + 0.15 * sin(moon_angle),
            parent_pos[2] + moon_radius * sin(moon_angle),
        )
        placements.append(
            PlacementSpec(
                role=role,
                index=i,
                position=moon_position,
                orbit_center=parent_pos,
                parent=parent_role if plan.hierarchy_needed else None,
                metadata={"kind": "moon", "orbit_radius": moon_radius},
            )
        )

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["solar system layout"])


def _build_organic(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)
    lower_roles = {role.lower() for role, _ in roles}

    if {"strand_left", "strand_right"}.issubset(lower_roles) or "base_pair" in lower_roles:
        requested_pairs = int(plan.repeat_counts.get("base_pair", 0)) or max(4, plan.num_objects // 2)
        pair_count = max(2, min(requested_pairs, max(1, plan.num_objects // 3)))
        turns = 2.0
        pitch = 0.85
        radius = 1.15

        for i in range(pair_count):
            t = i * (2.0 * pi / pair_count) * turns
            y = i * pitch
            left = _vec(radius * cos(t), y, radius * sin(t))
            right = _vec(radius * cos(t + pi), y, radius * sin(t + pi))
            placements.append(
                PlacementSpec(
                    role=f"strand_left_{i}",
                    index=i,
                    position=left,
                    orbit_center=_vec(0.0, y, 0.0),
                    metadata={"kind": "strand", "side": "left", "helix_turn": i},
                )
            )
            placements.append(
                PlacementSpec(
                    role=f"strand_right_{i}",
                    index=i,
                    position=right,
                    orbit_center=_vec(0.0, y, 0.0),
                    metadata={"kind": "strand", "side": "right", "helix_turn": i},
                )
            )
            placements.append(
                PlacementSpec(
                    role=f"base_pair_{i}",
                    index=i,
                    position=_vec(0.0, y, 0.0),
                    rotation=_vec(0.0, 0.0, pi / 2.0),
                    metadata={"kind": "connector", "pair_index": i},
                )
            )

        # Trim or pad to the requested object count so the plan remains stable.
        placements = placements[: plan.num_objects]

        if len(placements) < plan.num_objects:
            filler_index = 0
            while len(placements) < plan.num_objects:
                y = (filler_index + pair_count) * pitch
                placements.append(
                    PlacementSpec(
                        role=f"support_{filler_index}",
                        index=filler_index,
                        position=_vec(0.0, y, 0.0),
                        metadata={"kind": "support"},
                    )
                )
                filler_index += 1

        # Add any non-helix support roles only if the plan still has room.
        for role, idx in roles:
            if len(placements) >= plan.num_objects:
                break
            if role.lower().startswith(("strand_left", "strand_right", "base_pair")):
                continue
            placements.append(
                PlacementSpec(
                    role=role,
                    index=idx,
                    position=_vec(0.0, idx * pitch * 0.5, 0.0),
                    metadata={"kind": "support"},
                )
            )

        return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["dna helix layout"])

    # Generic organic: use a vertical curve with gentle offsets.
    for i, (role, idx) in enumerate(roles):
        y = i * 0.95
        x = 0.65 * sin(i * 0.8)
        z = 0.65 * cos(i * 0.8)
        placements.append(
            PlacementSpec(
                role=role,
                index=idx,
                position=_vec(x, y, z),
                metadata={"kind": "organic_segment"},
            )
        )

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["generic organic layout"])


def _build_grid(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)
    count = max(1, plan.num_objects)

    for i in range(count):
        if i < len(roles):
            role, idx = roles[i]
        else:
            role, idx = (f"cell_{i}", i)

        position = _grid_position(i, count, spacing=2.8 if plan.complexity == "low" else 3.3)
        placements.append(
            PlacementSpec(
                role=role,
                index=idx,
                position=position,
                metadata={"kind": "grid_cell", "cell_index": i},
            )
        )

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["grid layout"])


def _build_fallback(plan: ScenePlan) -> ParametricScene:
    placements: list[PlacementSpec] = []
    roles = _ordered_roles(plan)
    radius = 2.0 + 0.25 * len(roles)

    target_count = max(1, plan.num_objects)
    for i in range(target_count):
        if i < len(roles):
            role, idx = roles[i]
        else:
            role, idx = (f"fallback_{i}", i)
        placements.append(
            PlacementSpec(
                role=role,
                index=idx,
                position=_circle_point(radius, i * GOLDEN_ANGLE, y=(i % 3) * 0.5),
                metadata={"kind": "fallback"},
            )
        )

    return ParametricScene(scene_type=plan.scene_type, placements=placements, notes=["fallback radial layout"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_parametric_scene(plan: ScenePlan) -> ParametricScene:
    """Convert a ScenePlan into deterministic placements."""

    scene_type = plan.scene_type.lower()
    if scene_type == "atom":
        result = _build_atom(plan)
    elif scene_type == "molecule":
        result = _build_molecule(plan)
    elif scene_type in {"solar_system", "astronomical"}:
        result = _build_solar_system(plan)
    elif scene_type == "organic":
        result = _build_organic(plan)
    elif scene_type in {"mechanical", "system", "structure", "geometric", "crystalline", "abstract", "diagram", "vehicle", "landscape"}:
        result = _build_grid(plan)
    else:
        result = _build_fallback(plan)

    if len(result.placements) != plan.num_objects:
        logger.info(
            "parametric_generator: placement count=%d differs from plan num_objects=%d for scene_type=%s",
            len(result.placements),
            plan.num_objects,
            plan.scene_type,
        )

    return result
