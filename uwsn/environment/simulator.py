"""Run CH generation and independent AUV milestones on a single event clock."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from ..data_generation import GenerationEvent, GenerationManager, GenerationResult
from ..entities import AUV, CHReport, ClusterHead, Sink
from ..entities.common import _nonnegative_float, _positive_float, _require_int
from .event_queue import EventQueue
from .events import EventType, SimulationEvent


@dataclass(frozen=True)
class EventLogEntry:
    """Simulation-only execution record, including the handler result."""

    time_s: float
    event_type: EventType
    entity_id: int | None
    result: GenerationResult | str


class Simulator:
    """A finite event run with a private copy of the initial scenario.

    Provide an uninitialized manager with the same [0, mission_duration_s)
    window. By default, cluster_heads are taken from its registrations. Reset
    restores entities and RNG state and clears manually scheduled AUV events.
    Read live entities from the simulator and schedule AUV milestones via
    schedule_event. The queue and log are simulation diagnostics.
    """

    def __init__(
        self,
        mission_duration_s: float,
        *,
        generation_manager: GenerationManager | None = None,
        cluster_heads: Iterable[ClusterHead] | None = None,
        auvs: Iterable[AUV] = (),
        sink: Sink | None = None,
    ) -> None:
        self.mission_duration_s = _positive_float(mission_duration_s, "mission_duration_s")
        manager = generation_manager
        if manager is None:
            manager = GenerationManager(self.mission_duration_s)
        if manager.start_time_s != 0 or manager.mission_deadline_s != self.mission_duration_s:
            raise ValueError("generation manager must use the simulator mission window")
        registered = manager.registered_cluster_heads
        heads = registered if cluster_heads is None else tuple(cluster_heads)
        vehicles = tuple(auvs)
        heads_by_id = {ch.ch_id: ch for ch in heads}
        vehicles_by_id = {auv.auv_id: auv for auv in vehicles}
        if len(heads_by_id) != len(heads) or len(vehicles_by_id) != len(vehicles):
            raise ValueError("CH and AUV IDs must be unique within each entity type")
        if any(heads_by_id.get(ch.ch_id) is not ch for ch in registered):
            raise ValueError("cluster_heads must contain the same entities registered for generation")
        self._initial_scenario = deepcopy((heads_by_id, vehicles_by_id, sink, manager))
        self.reset()

    @property
    def current_time_s(self) -> float:
        return self.event_queue.current_time_s

    @property
    def decision_ready(self) -> frozenset[int]:
        """Return the IDs of AUVs that are ready for a new decision."""
        return frozenset(self._decision_ready)

    def create_ch_report(self, ch_id: int, *, include_generation_counter: bool = False) -> CHReport:
        """Read a live CH snapshot at the simulation clock after processing due events."""
        _require_int(ch_id, "ch_id")
        ch = self.cluster_heads[ch_id]
        next_event = self.event_queue.peek()
        if (
            next_event is not None
            and next_event.time_s == self.current_time_s
            and next_event.event_type is EventType.CH_DATA_GENERATION
        ):
            raise RuntimeError("process generation events at the current time before reading a CH report")
        return ch.create_report(self.current_time_s, include_generation_counter=include_generation_counter)

    def reset(self) -> None:
        """Restore the initial scenario and schedule its first events at time zero."""
        self.cluster_heads, self.auvs, self.sink, self.generation_manager = deepcopy(self._initial_scenario)
        self.event_queue = EventQueue()
        self.event_log: list[EventLogEntry] = []
        self._decision_ready: set[int] = set()
        self._finished = False
        self.generation_manager.initialize(synchronize_clock=True)
        self._schedule_next_generation()
        self.event_queue.push(self.mission_duration_s, EventType.MISSION_END)

    def schedule_event(self, time_s: float, event_type: EventType, entity_id: int) -> SimulationEvent:
        """Schedule AUV decision readiness or wait completion.

        Scheduling a wait clears only that AUV's readiness. Milestones at the
        deadline can execute before MISSION_END. Later ones are discarded there.
        """
        if self._finished:
            raise RuntimeError("the mission has ended")
        if event_type not in (EventType.AUV_DECISION_READY, EventType.WAIT_COMPLETE):
            raise ValueError("only AUV decision-ready and wait-complete events can be scheduled")
        _require_int(entity_id, "entity_id")
        if entity_id not in self.auvs:
            raise ValueError(f"unknown AUV ID {entity_id}")
        event = self.event_queue.push(time_s, event_type, entity_id)
        if event_type is EventType.WAIT_COMPLETE:
            self._decision_ready.discard(entity_id)
        return event

    def step_event(self) -> EventLogEntry | None:
        """Process the earliest event, or return None after mission end."""
        event = self.event_queue.peek()
        if self._finished or event is None:
            return None
        self.generation_manager.advance_to(event.time_s)
        self.event_queue.pop()
        result = self._handle_event(event)
        entry = EventLogEntry(event.time_s, event.event_type, event.entity_id, result)
        self.event_log.append(entry)
        return entry

    def run_until(self, time_s: float) -> tuple[EventLogEntry, ...]:
        """Process events through time_s inclusive, capped at the mission deadline."""
        limit = min(_nonnegative_float(time_s, "time_s"), self.mission_duration_s)
        if limit < self.current_time_s:
            raise ValueError("cannot run the simulator backwards")
        first_record = len(self.event_log)
        while not self._finished and (event := self.event_queue.peek()) is not None and event.time_s <= limit:
            self.step_event()
        self.generation_manager.advance_to(limit)
        self.event_queue.advance_to(limit)
        return tuple(self.event_log[first_record:])

    def run(self) -> tuple[EventLogEntry, ...]:
        """Run through the mission deadline and return the newly processed records."""
        return self.run_until(self.mission_duration_s)

    def _schedule_next_generation(self) -> None:
        # The manager retains one candidate per CH; enqueue only its global next.
        pending = self.generation_manager.peek_next_event()
        if pending is not None and pending.time_s < self.mission_duration_s:
            self.event_queue.push(pending.time_s, EventType.CH_DATA_GENERATION, pending.ch_id, pending)

    def _handle_event(self, event: SimulationEvent) -> GenerationResult | str:
        if event.event_type is EventType.CH_DATA_GENERATION:
            payload = event.payload
            if not isinstance(payload, GenerationEvent) or event.time_s >= self.mission_duration_s:
                raise RuntimeError("invalid scheduled CH generation event")
            result = self.generation_manager.process_event(payload)
            self._schedule_next_generation()
            return result
        if event.event_type in (EventType.AUV_DECISION_READY, EventType.WAIT_COMPLETE):
            assert event.entity_id is not None
            self._decision_ready.add(event.entity_id)
            return "decision_ready"
        if event.event_type is EventType.MISSION_END:
            self._finished = True
            self.event_queue.clear()
            return "mission_ended"
        raise NotImplementedError(f"no simulator handler for {event.event_type.value}")
