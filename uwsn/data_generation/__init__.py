"""Piecewise Poisson packet generation for the simulation's CH entities."""

from .generation_manager import (
    GenerationEvent,
    GenerationManager,
    GenerationResult,
    GenerationStats,
    PacketParameters,
)
from .poisson_process import PoissonProcess
from .rate_schedule import RateSchedule, RateSegment

__all__ = [
    "GenerationEvent",
    "GenerationManager",
    "GenerationResult",
    "GenerationStats",
    "PacketParameters",
    "PoissonProcess",
    "RateSchedule",
    "RateSegment",
]
