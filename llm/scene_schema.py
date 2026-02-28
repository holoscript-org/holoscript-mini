from typing import Literal
from pydantic import BaseModel, field_validator


VALID_TYPES = ("sphere", "cube", "cylinder", "ring", "label")
VALID_ANIMATIONS = ("none", "orbit")
MIN_OBJECTS = 1
MAX_OBJECTS = 20
POSITION_LENGTH = 3
COLOR_LENGTH = 3
ORBIT_CENTER_LENGTH = 3


class SceneObject(BaseModel):
    id: str
    type: Literal["sphere", "cube", "cylinder", "ring", "label"]
    position: list[float]
    color: list[float]
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
