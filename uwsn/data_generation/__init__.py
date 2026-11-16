"""Poisson and MMPP packet generation for the simulation's CH entities."""

from .generation_manager import (
    GenerationEvent,
    GenerationManager,
    GenerationResult,
    GenerationStats,
    PacketParameters,
)
from .poisson_process import PoissonProcess
from .mmpp_process import GenerationMode, MMPPConfig, MMPPEvent, MMPPProcess, MMPPStats
from .rate_schedule import RateSchedule, RateSegment

__all__ = [
    "GenerationMode",
    "MMPPConfig",
    "MMPPEvent",
    "MMPPProcess",
    "MMPPStats",
    "GenerationEvent",
    "GenerationManager",
    "GenerationResult",
    "GenerationStats",
    "PacketParameters",
    "PoissonProcess",
    "RateSchedule",
    "RateSegment",
]
