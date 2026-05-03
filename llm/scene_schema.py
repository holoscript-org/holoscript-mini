from typing import Literal, Optional, Any
from pydantic import BaseModel, field_validator


VALID_TYPES = ("primitive", "mesh")
VALID_GEOMETRY_TYPES = ("sphere", "box", "cylinder", "ring", "torus", "plane", "capsule")
VALID_ANIMATIONS = ("none", "orbit", "spin")
MIN_OBJECTS = 1
MAX_OBJECTS = 20
POSITION_LENGTH = 3
SCALE_LENGTH = 3


class GeometryObject(BaseModel):
    """Geometry definition for primitive objects."""
    type: str  # sphere, box, cylinder, ring, torus, plane, capsule
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None
    innerRadius: Optional[float] = None
    outerRadius: Optional[float] = None
    tube: Optional[float] = None
    thetaSegments: Optional[int] = None


class MaterialObject(BaseModel):
    """Material definition for objects."""
    type: Literal["standard", "phong", "basic"] = "standard"
    color: str  # hex color #rrggbb
    roughness: Optional[float] = 0.5
    metalness: Optional[float] = 0.0
    opacity: Optional[float] = 1.0
    transparent: Optional[bool] = False
    emissive: Optional[str] = None
    emissiveIntensity: Optional[float] = 0.0

    @field_validator("color")
    @classmethod
    def validate_color_hex(cls, v: str) -> str:
        if not (isinstance(v, str) and v.startswith("#") and len(v) == 7):
            raise ValueError(f"color must be hex format #rrggbb, got {v!r}")
        return v


class AnimationObject(BaseModel):
    """Animation definition for objects."""
    type: Literal["none", "orbit", "spin"] = "none"
    center: Optional[list[float]] = None
    speed: Optional[float] = None
    axis: Optional[list[float]] = None
    phase: Optional[float] = None

    @field_validator("center")
    @classmethod
    def validate_center(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 3:
            raise ValueError(f"center must have 3 floats, got {len(v)}")
        return v


class CameraObject(BaseModel):
    """Camera definition."""
    position: list[float]
    target: list[float]
    fov: float = 60.0

    @field_validator("position", "target")
    @classmethod
    def validate_position_target(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"position/target must have 3 floats, got {len(v)}")
        return v


class LightObject(BaseModel):
    """Light definition."""
    type: Literal["ambient", "point", "directional"]
    color: str  # hex #rrggbb
    intensity: float = 1.0
    position: Optional[list[float]] = None
    castShadow: Optional[bool] = False

    @field_validator("color")
    @classmethod
    def validate_color_hex(cls, v: str) -> str:
        if not (isinstance(v, str) and v.startswith("#") and len(v) == 7):
            raise ValueError(f"color must be hex format #rrggbb, got {v!r}")
        return v


class SceneObject(BaseModel):
    """Object definition (primitive or mesh)."""
    id: str
    type: Literal["primitive", "mesh"]
    
    # For primitives
    geometry: Optional[GeometryObject] = None
    
    # For meshes
    model: Optional[str] = None
    
    # Common fields
    position: list[float]
    scale: list[float] = [1, 1, 1]
    rotation: Optional[list[float]] = None
    material: Optional[MaterialObject] = None
    animation: Optional[AnimationObject] = None
    label: Optional[str] = None
    parent: Optional[str] = None
    
    # Legacy/backward-compat fields (for old format)
    color: Optional[list[float]] = None
    orbit_center: Optional[list[float]] = None
    orbit_speed: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: list[float]) -> list[float]:
        if len(v) != POSITION_LENGTH:
            raise ValueError(f"position must have exactly {POSITION_LENGTH} floats")
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: list[float]) -> list[float]:
        if len(v) != SCALE_LENGTH:
            raise ValueError(f"scale must have exactly {SCALE_LENGTH} floats")
        for s in v:
            if s <= 0.0:
                raise ValueError(f"scale values must be positive, got {s}")
        return v

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 3:
            raise ValueError(f"rotation must have 3 floats, got {len(v)}")
        return v


class SceneSchema(BaseModel):
    """Root scene schema (Member1 contract)."""
    name: Optional[str] = None
    camera: Optional[CameraObject] = None
    lights: Optional[list[LightObject]] = None
    objects: list[SceneObject]

    @field_validator("objects")
    @classmethod
    def validate_objects_count(cls, v: list[SceneObject]) -> list[SceneObject]:
        if not (MIN_OBJECTS <= len(v) <= MAX_OBJECTS):
            raise ValueError(f"objects list must have between {MIN_OBJECTS} and {MAX_OBJECTS} entries")
        return v

