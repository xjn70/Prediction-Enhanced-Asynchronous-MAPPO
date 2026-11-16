"""Finite, piecewise-constant packet generation rates in packets/second."""

from bisect import bisect_right
from dataclasses import dataclass, field
from math import fsum

from ..entities.common import _nonnegative_float


@dataclass(frozen=True)
class RateSegment:
    """A constant rate on the half-open interval [start_time_s, end_time_s)."""

    start_time_s: float
    end_time_s: float
    rate_pps: float

    def __post_init__(self) -> None:
        for name in ("start_time_s", "end_time_s", "rate_pps"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")


@dataclass(frozen=True)
class RateSchedule:
    """Ordered, nonoverlapping segments; uncovered times have zero rate.

    Segment boundaries and the end of the schedule are right-exclusive.
    Inputs are copied into a tuple so later changes to an input list have no effect.
    """

    segments: tuple[RateSegment, ...]
    _starts: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            segments = tuple(self.segments)
        except TypeError as error:
            raise ValueError("segments must be an iterable of RateSegment objects") from error
        if not segments or any(not isinstance(segment, RateSegment) for segment in segments):
            raise ValueError("segments must contain at least one RateSegment")
        for previous, current in zip(segments, segments[1:]):
            if current.start_time_s < previous.end_time_s:
                raise ValueError("segments must be ordered and must not overlap")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "_starts", tuple(segment.start_time_s for segment in segments))

    @classmethod
    def fixed(cls, rate_pps: float, end_time_s: float, start_time_s: float = 0.0) -> "RateSchedule":
        """Build a schedule with one constant rate over a finite interval."""
        return cls((RateSegment(start_time_s, end_time_s, rate_pps),))

    def rate_at(self, time_s: float) -> float:
        """Return the rate at a time, including zero in gaps and outside the schedule."""
        time_s = _nonnegative_float(time_s, "time_s")
        index = bisect_right(self._starts, time_s) - 1
        if index < 0:
            return 0.0
        segment = self.segments[index]
        if time_s >= segment.end_time_s:
            return 0.0
        return segment.rate_pps

    def integrated_intensity(self, start_time_s: float, end_time_s: float) -> float:
        """Return the expected number of arrivals over the requested interval."""
        start_time_s = _nonnegative_float(start_time_s, "start_time_s")
        end_time_s = _nonnegative_float(end_time_s, "end_time_s")
        if end_time_s < start_time_s:
            raise ValueError("end_time_s cannot precede start_time_s")
        return fsum(
            segment.rate_pps
            * max(0.0, min(end_time_s, segment.end_time_s) - max(start_time_s, segment.start_time_s))
            for segment in self.segments
        )
