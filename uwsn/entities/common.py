"""Shared entity types and input validation."""

from enum import Enum
from math import isfinite

Position3D = tuple[float, float, float]


class OverflowPolicy(str, Enum):
    """Select which packets to discard when the CH buffer is full."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


class AUVStatus(str, Enum):
    """Execution states tracked by an AUV."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    COLLECTING = "collecting"
    RETURNING = "returning"
    WAITING_SINK = "waiting_sink"
    UPLOADING = "uploading"


def _require_int(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _nonnegative_float(value: float, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _positive_float(value: float, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _validate_position(position: Position3D) -> Position3D:
    if not isinstance(position, tuple) or len(position) != 3:
        raise ValueError("position must be a tuple of three coordinates")
    x, y, z = position
    return (
        _finite_float(x, "position x"),
        _finite_float(y, "position y"),
        _finite_float(z, "position z"),
    )
