"""Public entry points for the discrete-event simulation core."""

from .event_queue import EventQueue
from .events import EventType, SimulationEvent
from .simulator import CHArrival, CollectionResult, EventLogEntry, Simulator
from .scenarios import make_scenario
from .actions import ReturnAction, VisitAction
from .observation import LocalObservation, build_local_observation
from .action_rules import ReturnRuleConfig

__all__ = [
    "CHArrival", "CollectionResult", "EventLogEntry", "EventQueue", "EventType", "SimulationEvent",
    "Simulator", "make_scenario", "LocalObservation", "build_local_observation",
    "ReturnAction", "VisitAction", "ReturnRuleConfig",
]
