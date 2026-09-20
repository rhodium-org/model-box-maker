"""The dimensions that shape a box, with their defaults and validation.

Implements model-box-maker REQ-0003 (every dimension is an option with a stated
default; bad values are refused before any geometry) and the lattice options of
REQ-0015 and the tie band of REQ-0004.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

NOZZLE_LINE = 0.4  # mm; walls and skirts must be at least two of these (CON-0001)
WALL_PATTERNS = ("solid", "hex", "diamond")


class SpecError(ValueError):
    """A dimension option holds a value the tool refuses.

    ``option`` is the command-line spelling so the message can name it.
    """

    def __init__(self, option: str, message: str):
        super().__init__(f"{option}: {message}")
        self.option = option


@dataclass(frozen=True)
class BoxSpec:
    """Every dimension in millimetres. Field names match the CLI options."""

    clearance: float = 1.0
    wall: float = 2.4
    floor: float = 3.0
    top_space: float = 2.0
    lip: float = 6.0
    lid_plate: float = 2.0
    skirt: float = 1.6
    fit: float = 0.2
    corner_radius: float = 3.0
    pitch: float = 0.25
    size_tolerance: float = 5.0  # percent (REQ-0004)
    walls: str = "solid"  # solid | hex | diamond (REQ-0015)
    max_hole: float = 6.0
    band: float = 4.0

    def __post_init__(self) -> None:
        self.validate()

    @staticmethod
    def option_name(field_name: str) -> str:
        return "--" + field_name.replace("_", "-")

    def validate(self) -> None:
        positive = (
            "clearance", "wall", "floor", "top_space", "lip", "lid_plate",
            "skirt", "fit", "pitch", "max_hole", "band",
        )
        for name in positive:
            value = getattr(self, name)
            if not (value > 0):
                raise SpecError(self.option_name(name), f"must be greater than zero, got {value}")
        if self.corner_radius < 0:
            raise SpecError("--corner-radius", f"must be zero or more, got {self.corner_radius}")
        if self.size_tolerance < 0:
            raise SpecError("--size-tolerance", f"must be zero or more, got {self.size_tolerance}")
        minimum = 2 * NOZZLE_LINE
        for name in ("wall", "skirt"):
            value = getattr(self, name)
            if value < minimum - 1e-9:
                raise SpecError(
                    self.option_name(name),
                    f"must be at least two nozzle lines ({minimum} mm), got {value}",
                )
        if self.fit >= self.lip:
            raise SpecError("--fit", f"must be smaller than the lip height ({self.lip} mm), got {self.fit}")
        if self.walls not in WALL_PATTERNS:
            raise SpecError("--walls", f"must be one of {', '.join(WALL_PATTERNS)}, got {self.walls}")

    @property
    def body_offset(self) -> float:
        """How far the wall below the lip stands outside the lip (REQ-0013)."""
        return self.skirt + self.fit

    @property
    def lip_radius(self) -> float:
        """Corner radius of the lip; the closed box keeps ``corner_radius`` (REQ-0012)."""
        return max(self.corner_radius - self.body_offset, 0.0)

    @property
    def skirt_height(self) -> float:
        """The skirt stops the fit gap short of the shoulder so the rim is the stop."""
        return self.lip - self.fit

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def dimension_names() -> list[str]:
    """The dimension fields in the order REQ-0003 lists them."""
    return [
        "clearance", "wall", "floor", "top_space", "lip", "lid_plate",
        "skirt", "fit", "corner_radius", "pitch",
    ]
