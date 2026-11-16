"""Straight-line travel in meters and seconds."""

from dataclasses import dataclass
from math import dist, isclose

from ..entities.common import Position3D, _finite_float, _positive_float, _validate_position


@dataclass(frozen=True)
class TravelLeg:
    """A travel destination and its distance and duration from departure."""

    destination: Position3D
    distance_m: float
    travel_time_s: float


def distance_m(start: Position3D, end: Position3D) -> float:
    """Return the three-dimensional Euclidean distance in meters."""
    return _finite_float(dist(_validate_position(start), _validate_position(end)), "distance_m")


def in_communication_range(position: Position3D, center: Position3D, radius_m: float) -> bool:
    """Apply a floating-point tolerance when checking a computed boundary point."""
    radius_m = _positive_float(radius_m, "radius_m")
    separation = distance_m(position, center)
    return separation <= radius_m or isclose(separation, radius_m, rel_tol=1e-12, abs_tol=1e-9)


def plan_visit(start: Position3D, center: Position3D, radius_m: float, speed_mps: float) -> TravelLeg:
    """Plan travel to the closest point in the CH communication sphere."""
    start = _validate_position(start)
    center = _validate_position(center)
    radius_m = _positive_float(radius_m, "radius_m")
    speed_mps = _positive_float(speed_mps, "speed_mps")
    separation = distance_m(start, center)
    if separation <= radius_m:
        return TravelLeg(start, 0.0, 0.0)
    x, y, z = start
    cx, cy, cz = center
    destination = (
        cx + radius_m * ((x - cx) / separation),
        cy + radius_m * ((y - cy) / separation),
        cz + radius_m * ((z - cz) / separation),
    )
    travel_distance = separation - radius_m
    return TravelLeg(destination, travel_distance, _finite_float(travel_distance / speed_mps, "travel_time_s"))


def plan_return(start: Position3D, sink_position: Position3D, speed_mps: float) -> TravelLeg:
    """Plan travel to the fixed Sink position."""
    destination = _validate_position(sink_position)
    speed_mps = _positive_float(speed_mps, "speed_mps")
    travel_distance = distance_m(start, destination)
    return TravelLeg(destination, travel_distance, _finite_float(travel_distance / speed_mps, "travel_time_s"))
