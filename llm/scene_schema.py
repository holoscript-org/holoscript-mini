from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


VALID_TYPES = ("sphere", "cube", "cylinder", "ring", "label")
VALID_ANIMATIONS = ("none", "orbit")
VALID_SURFACE_STYLES = ("plain", "emissive_glow", "polar_caps", "earth", "banded", "saturn_rings")
MIN_OBJECTS = 1
MAX_OBJECTS = 20
POSITION_LENGTH = 3
COLOR_LENGTH = 3
ORBIT_CENTER_LENGTH = 3
MIN_SIZE = 0.0
MIN_RING_FACTOR = 0.0
MIN_BAND_WIDTH = 0.0
MIN_OPACITY = 0.0
MAX_OPACITY = 1.0


class Band(BaseModel):
    color: list[float]
    width: float

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: list[float]) -> list[float]:
        if len(v) != COLOR_LENGTH:
            raise ValueError(f"band color must have exactly {COLOR_LENGTH} floats")
        return v

    @field_validator("width")
    @classmethod
    def validate_width(cls, v: float) -> float:
        if v <= MIN_BAND_WIDTH:
            raise ValueError("band width must be greater than 0")
        return v


class Ring(BaseModel):
    inner_radius_factor: float
    outer_radius_factor: float
    color: list[float]
    opacity: float

    @field_validator("inner_radius_factor")
    @classmethod
    def validate_inner_radius_factor(cls, v: float) -> float:
        if v <= MIN_RING_FACTOR:
            raise ValueError("inner_radius_factor must be greater than 0")
        return v

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: list[float]) -> list[float]:
        if len(v) != COLOR_LENGTH:
            raise ValueError(f"ring color must have exactly {COLOR_LENGTH} floats")
        return v

    @field_validator("opacity")
    @classmethod
    def validate_opacity(cls, v: float) -> float:
        if not (MIN_OPACITY <= v <= MAX_OPACITY):
            raise ValueError("opacity must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_radius_order(self):
        if self.outer_radius_factor <= self.inner_radius_factor:
            raise ValueError("outer_radius_factor must be greater than inner_radius_factor")
        return self


class SceneObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["sphere", "cube", "cylinder", "ring", "label"]
    position: list[float]
    color: list[float]
    secondary_color: list[float] | None = None
    size: float | None = None
    surface_style: Literal["plain", "emissive_glow", "polar_caps", "earth", "banded", "saturn_rings"] | None = None
    bands: list[Band] | None = None
    ring: Ring | None = None
    animation: Literal["none", "orbit"]
    orbit_center: list[float]
    orbit_speed: float

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: list[float]) -> list[float]:
        if len(v) != POSITION_LENGTH:
            raise ValueError(f"position must have exactly {POSITION_LENGTH} floats")
        return v

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: list[float]) -> list[float]:
        if len(v) != COLOR_LENGTH:
            raise ValueError(f"color must have exactly {COLOR_LENGTH} floats")
        return v

    @field_validator("secondary_color")
    @classmethod
    def validate_secondary_color(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != COLOR_LENGTH:
            raise ValueError(f"secondary_color must have exactly {COLOR_LENGTH} floats")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: float | None) -> float | None:
        if v is not None and v <= MIN_SIZE:
            raise ValueError("size must be greater than 0")
        return v

    @field_validator("orbit_center")
    @classmethod
    def validate_orbit_center(cls, v: list[float]) -> list[float]:
        if len(v) != ORBIT_CENTER_LENGTH:
            raise ValueError(f"orbit_center must have exactly {ORBIT_CENTER_LENGTH} floats")
        return v


class SceneSchema(BaseModel):
    objects: list[SceneObject]

    @field_validator("objects")
    @classmethod
    def validate_objects_count(cls, v: list[SceneObject]) -> list[SceneObject]:
        if not (MIN_OBJECTS <= len(v) <= MAX_OBJECTS):
            raise ValueError(f"objects list must have between {MIN_OBJECTS} and {MAX_OBJECTS} entries")
        return v
