"""A deterministic, forward-only event queue implemented with a min-heap."""

from heapq import heappop, heappush
from itertools import count

from ..entities.common import _nonnegative_float
from .events import EventType, SimulationEvent


class EventQueue:
    """Schedule fresh events and track the clock without replaying popped entries."""

    def __init__(self) -> None:
        self._heap: list[tuple[tuple[float, int, int], SimulationEvent]] = []
        self._sequence_ids = count()
        self._current_time_s = 0.0
        self._last_popped_key: tuple[float, int, int] | None = None

    @property
    def current_time_s(self) -> float:
        return self._current_time_s

    def push(
        self, time_s: float, event_type: EventType, entity_id: int | None = None, payload: object = None
    ) -> SimulationEvent:
        """Schedule a new event. Same-time insertion must preserve processed order."""
        event = SimulationEvent(time_s, event_type, entity_id, next(self._sequence_ids), payload)
        if event.time_s < self._current_time_s:
            raise ValueError("cannot schedule an event in the past")
        key = event.sort_key
        if self._last_popped_key is not None and key < self._last_popped_key:
            raise ValueError("cannot schedule an earlier priority after processing that time")
        heappush(self._heap, (key, event))
        return event

    def peek(self) -> SimulationEvent | None:
        """Return the next event without modifying the heap or clock."""
        return self._heap[0][1] if self._heap else None

    def pop(self) -> SimulationEvent | None:
        """Remove the next event and move the clock to its time."""
        if not self._heap:
            return None
        key, event = heappop(self._heap)
        self._current_time_s = event.time_s
        self._last_popped_key = key
        return event

    def advance_to(self, time_s: float) -> None:
        """Advance through an idle interval without skipping any pending event."""
        time_s = _nonnegative_float(time_s, "time_s")
        if time_s < self._current_time_s:
            raise ValueError("cannot move the event clock backwards")
        next_event = self.peek()
        if next_event is not None and next_event.time_s < time_s:
            raise ValueError("cannot advance past a pending event")
        self._current_time_s = time_s

    def is_empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        """Discard pending events while retaining the current clock and sequence."""
        self._heap.clear()
