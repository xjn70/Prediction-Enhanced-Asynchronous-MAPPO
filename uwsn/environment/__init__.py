"""Public entry points for the discrete-event simulation core."""

from .event_queue import EventQueue
from .events import EventType, SimulationEvent
from .simulator import CHArrival, CollectionResult, EventLogEntry, Simulator
from .scenarios import make_scenario

__all__ = [
    "CHArrival", "CollectionResult", "EventLogEntry", "EventQueue", "EventType", "SimulationEvent",
    "Simulator", "make_scenario",
]
