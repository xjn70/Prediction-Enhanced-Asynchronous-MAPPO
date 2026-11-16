"""Exact continuous-time two-state MMPP sampling with cached lookahead."""

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from enum import Enum
from math import inf, nextafter
from random import Random

from ..entities.common import _nonnegative_float, _require_int


class GenerationMode(str, Enum):
    """Two physical modes and a stationary initialization option."""

    NORMAL = "normal"
    BURST = "burst"
    STATIONARY = "stationary"


@dataclass(frozen=True)
class MMPPConfig:
    """Packet rates and CTMC transition rates, all expressed per second.

    STATIONARY is used only to choose the initial physical mode. Explicit
    NORMAL or BURST is required when both transition rates are zero.
    Pass random_seed explicitly for each CH.
    """

    lambda_normal_pps: float
    lambda_burst_pps: float
    q_normal_to_burst_per_s: float
    q_burst_to_normal_per_s: float
    initial_mode: GenerationMode = GenerationMode.NORMAL
    random_seed: int = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in (
            "lambda_normal_pps", "lambda_burst_pps",
            "q_normal_to_burst_per_s", "q_burst_to_normal_per_s",
        ):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        object.__setattr__(self, "initial_mode", GenerationMode(self.initial_mode))
        _require_int(self.random_seed, "random_seed")
        if self.initial_mode is GenerationMode.STATIONARY:
            self.stationary_probabilities()

    def stationary_probabilities(self) -> tuple[float, float]:
        """Return (pi_normal, pi_burst), using scaled rates to avoid overflow."""
        scale = max(self.q_normal_to_burst_per_s, self.q_burst_to_normal_per_s)
        if scale == 0:
            raise ValueError("STATIONARY requires a positive transition rate; specify NORMAL or BURST")
        q01 = self.q_normal_to_burst_per_s / scale
        q10 = self.q_burst_to_normal_per_s / scale
        return q10 / (q01 + q10), q01 / (q01 + q10)


@dataclass(frozen=True)
class MMPPEvent:
    """A generated packet's time and hidden mode, for internal diagnostics only."""

    time_s: float
    mode: GenerationMode


@dataclass(frozen=True)
class MMPPStats:
    """Diagnostics committed through accounted_until_s, not a global simulation clock.

    Packet counts include all emitted events, including packets later rejected by
    a CH buffer. Once next_event returns None, durations cover the full window.
    """

    normal_mode_duration_s: float = 0.0
    burst_mode_duration_s: float = 0.0
    normal_to_burst_transitions: int = 0
    burst_to_normal_transitions: int = 0
    packets_generated_in_normal: int = 0
    packets_generated_in_burst: int = 0
    accounted_until_s: float = 0.0


@dataclass(frozen=True)
class _PendingOutcome:
    event: MMPPEvent | None
    mode: GenerationMode
    stats: MMPPStats
    switches: tuple[MMPPStats, ...]


@dataclass
class MMPPProcess:
    """Competing exponential clocks on [start_time_s, end_time_s).

    peek_next_event samples and caches one outcome, including intervening mode
    switches. Repeated peeks leave the RNG and committed state unchanged.
    next_event commits that outcome; next_time_s offers the consuming interface
    used by PoissonProcess. All modes and statistics are simulation diagnostics.
    """

    config: MMPPConfig
    end_time_s: float
    start_time_s: float = 0.0
    _rng: Random = field(init=False, repr=False)
    _mode: GenerationMode = field(init=False, repr=False)
    _stats: MMPPStats = field(init=False, repr=False)
    _pending: _PendingOutcome | None = field(init=False, default=None, repr=False)
    _exhausted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, MMPPConfig):
            raise ValueError("config must be an MMPPConfig")
        self.start_time_s = _nonnegative_float(self.start_time_s, "start_time_s")
        self.end_time_s = _nonnegative_float(self.end_time_s, "end_time_s")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")
        self._rng = Random(self.config.random_seed)
        self._mode = self.config.initial_mode
        if self._mode is GenerationMode.STATIONARY:
            pi_normal, _ = self.config.stationary_probabilities()
            self._mode = GenerationMode.NORMAL if self._rng.random() < pi_normal else GenerationMode.BURST
        self._stats = MMPPStats(accounted_until_s=self.start_time_s)

    @property
    def current_mode(self) -> GenerationMode:
        """Return the committed mode at the diagnostic accounting time."""
        return self._mode

    def get_stats(self) -> MMPPStats:
        """Return an immutable diagnostic snapshot, excluding an unconsumed packet."""
        return self._stats

    def stats_at(self, time_s: float) -> MMPPStats:
        """Project diagnostics within the cached interval without consuming it.

        Switches through time_s are counted. A pending packet at time_s is
        counted only when consumed. The projection keeps sampling independent
        of how often the simulation clock or diagnostics are updated.
        """
        time_s = _nonnegative_float(time_s, "time_s")
        if time_s == self._stats.accounted_until_s:
            return self._stats
        pending = self._pending
        if pending is None or not self._stats.accounted_until_s < time_s <= pending.stats.accounted_until_s:
            raise ValueError("time_s must be within the cached MMPP interval")
        count = bisect_right(pending.switches, time_s, key=lambda stats: stats.accounted_until_s)
        anchor = pending.switches[count - 1] if count else self._stats
        in_normal = (self._mode is GenerationMode.NORMAL) == (count % 2 == 0)
        duration = time_s - anchor.accounted_until_s
        return replace(
            anchor,
            normal_mode_duration_s=anchor.normal_mode_duration_s + (duration if in_normal else 0.0),
            burst_mode_duration_s=anchor.burst_mode_duration_s + (0.0 if in_normal else duration),
            accounted_until_s=time_s,
        )

    def peek_next_event(self) -> MMPPEvent | None:
        """Return the cached next packet without committing mode or counter updates."""
        if self._exhausted:
            return None
        if self._pending is None:
            self._pending = self._sample_outcome()
        return self._pending.event

    def next_event(self) -> MMPPEvent | None:
        """Consume one packet event, or commit the final tail through the deadline."""
        if self._exhausted:
            return None
        event = self.peek_next_event()
        pending = self._pending
        assert pending is not None
        self._mode = pending.mode
        self._stats = pending.stats
        self._pending = None
        self._exhausted = event is None
        return event

    def next_time_s(self) -> float | None:
        """Consume the next packet and return its time, or None after exhaustion."""
        event = self.next_event()
        return None if event is None else event.time_s

    def _wait_s(self, rate: float) -> float:
        if rate == 0:
            return inf
        wait_s = self._rng.expovariate(rate)
        while wait_s == 0:
            wait_s = self._rng.expovariate(rate)
        return wait_s

    def _sample_outcome(self) -> _PendingOutcome:
        mode = self._mode
        stats = self._stats
        time_s = stats.accounted_until_s
        normal_duration = stats.normal_mode_duration_s
        burst_duration = stats.burst_mode_duration_s
        normal_to_burst = stats.normal_to_burst_transitions
        burst_to_normal = stats.burst_to_normal_transitions
        normal_packets = stats.packets_generated_in_normal
        burst_packets = stats.packets_generated_in_burst
        event = None
        switches = []

        while time_s < self.end_time_s:
            if mode is GenerationMode.NORMAL:
                arrival_rate = self.config.lambda_normal_pps
                switch_rate = self.config.q_normal_to_burst_per_s
            else:
                arrival_rate = self.config.lambda_burst_pps
                switch_rate = self.config.q_burst_to_normal_per_s
            arrival_wait = self._wait_s(arrival_rate)
            switch_wait = self._wait_s(switch_rate)
            next_time = time_s + min(arrival_wait, switch_wait)
            if next_time <= time_s:
                next_time = nextafter(time_s, inf)
            next_time = min(next_time, self.end_time_s)
            duration = next_time - time_s
            if mode is GenerationMode.NORMAL:
                normal_duration += duration
            else:
                burst_duration += duration
            time_s = next_time
            if time_s >= self.end_time_s:
                break

            # Tied clocks (probability zero in continuous time) prioritize switching.
            if arrival_wait < switch_wait:
                event = MMPPEvent(time_s, mode)
                if mode is GenerationMode.NORMAL:
                    normal_packets += 1
                else:
                    burst_packets += 1
                break
            if mode is GenerationMode.NORMAL:
                mode = GenerationMode.BURST
                normal_to_burst += 1
            else:
                mode = GenerationMode.NORMAL
                burst_to_normal += 1
            switches.append(replace(
                stats,
                normal_mode_duration_s=normal_duration,
                burst_mode_duration_s=burst_duration,
                normal_to_burst_transitions=normal_to_burst,
                burst_to_normal_transitions=burst_to_normal,
                accounted_until_s=time_s,
            ))

        return _PendingOutcome(event, mode, MMPPStats(
            normal_mode_duration_s=normal_duration,
            burst_mode_duration_s=burst_duration,
            normal_to_burst_transitions=normal_to_burst,
            burst_to_normal_transitions=burst_to_normal,
            packets_generated_in_normal=normal_packets,
            packets_generated_in_burst=burst_packets,
            accounted_until_s=time_s,
        ), tuple(switches))
