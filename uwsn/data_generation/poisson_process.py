"""Incremental Poisson sampling by inversion of integrated intensity."""

from dataclasses import dataclass, field
from math import inf, nextafter
from random import Random

from ..entities.common import _nonnegative_float, _require_int
from .rate_schedule import RateSchedule


@dataclass
class PoissonProcess:
    """An independent random stream on [start_time_s, end_time_s).

    Each arrival consumes a single Exp(1) integrated-intensity budget. Crossing
    a rate boundary subtracts the intensity already traversed from that budget.
    Gaps and zero-rate segments consume time but no intensity. Calls produce one
    event at a time, returning None permanently after the process is exhausted.
    """

    schedule: RateSchedule
    end_time_s: float
    seed: int
    start_time_s: float = 0.0
    _rng: Random = field(init=False, repr=False)
    _time_s: float = field(init=False, repr=False)
    _segment_index: int = field(init=False, default=0, repr=False)
    _exhausted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, RateSchedule):
            raise ValueError("schedule must be a RateSchedule")
        self.start_time_s = _nonnegative_float(self.start_time_s, "start_time_s")
        self.end_time_s = _nonnegative_float(self.end_time_s, "end_time_s")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")
        _require_int(self.seed, "seed")
        self._rng = Random(self.seed)
        self._time_s = self.start_time_s

    def next_time_s(self) -> float | None:
        """Sample the next time, strictly after the previous one and before the end.

        If floating-point addition rounds an interval to zero, advance by one
        representable float. Never emit that rounded time at a segment's end.
        """
        if self._exhausted:
            return None
        remaining_intensity = self._rng.expovariate(1.0)
        while remaining_intensity == 0.0:
            remaining_intensity = self._rng.expovariate(1.0)

        segments = self.schedule.segments
        while self._segment_index < len(segments):
            segment = segments[self._segment_index]
            start_s = max(self._time_s, segment.start_time_s)
            end_s = min(self.end_time_s, segment.end_time_s)
            if start_s >= self.end_time_s:
                break
            if start_s >= end_s or segment.rate_pps == 0.0:
                self._segment_index += 1
                continue

            available_intensity = segment.rate_pps * (end_s - start_s)
            if remaining_intensity < available_intensity:
                event_time_s = start_s + remaining_intensity / segment.rate_pps
                if event_time_s <= self._time_s:
                    event_time_s = nextafter(self._time_s, inf)
                if event_time_s < end_s:
                    self._time_s = event_time_s
                    return event_time_s
                # A rounded boundary belongs to the next active segment.
                remaining_intensity = 0.0
            else:
                remaining_intensity -= available_intensity
            self._segment_index += 1

        self._exhausted = True
        return None
