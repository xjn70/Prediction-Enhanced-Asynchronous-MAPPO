"""Event types and records for the UWSN simulation clock."""

from dataclasses import dataclass
from enum import Enum

from ..entities.common import _nonnegative_float, _require_int


class EventType(str, Enum):
    """Kinds of simulation events, including reserved service milestones."""

    CH_DATA_GENERATION = "ch_data_generation"
    AUV_DECISION_READY = "auv_decision_ready"
    AUV_ARRIVAL_CH = "auv_arrival_ch"
    COLLECTION_COMPLETE = "collection_complete"
    AUV_ARRIVAL_SINK = "auv_arrival_sink"
    UPLOAD_START = "upload_start"
    UPLOAD_COMPLETE = "upload_complete"
    WAIT_COMPLETE = "wait_complete"
    MISSION_END = "mission_end"

    @property
    def priority(self) -> int:
        """Process generation first, other milestones next, and mission end last."""
        if self is EventType.CH_DATA_GENERATION:
            return 0
        if self is EventType.MISSION_END:
            return 2
        return 1


@dataclass(frozen=True)
class SimulationEvent:
    """An event identified by a sequence number assigned by the queue."""

    time_s: float
    event_type: EventType
    entity_id: int | None
    sequence_id: int
    payload: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _nonnegative_float(self.time_s, "time_s"))
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if self.entity_id is not None:
            _require_int(self.entity_id, "entity_id")
        _require_int(self.sequence_id, "sequence_id")

    @property
    def sort_key(self) -> tuple[float, int, int]:
        """Return the deterministic heap ordering key."""
        return self.time_s, self.event_type.priority, self.sequence_id
