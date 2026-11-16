"""Coordinate independent CH generation streams and feed the entity buffers."""

from dataclasses import dataclass, field

from ..entities import BufferInsertResult, ClusterHead, DataPacket
from ..entities.common import _nonnegative_float, _require_int
from .mmpp_process import GenerationMode, MMPPConfig, MMPPProcess, MMPPStats
from .poisson_process import PoissonProcess
from .rate_schedule import RateSchedule


@dataclass(frozen=True)
class PacketParameters:
    """Constant packet size and value parameters for one CH's generated packets."""

    size_bytes: int
    base_value: float
    decay_rate_per_s: float

    def __post_init__(self) -> None:
        _require_int(self.size_bytes, "size_bytes", minimum=1)
        for name in ("base_value", "decay_rate_per_s"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))


@dataclass(frozen=True)
class GenerationEvent:
    """One pending generation time with an optional simulation-only hidden mode."""

    ch_id: int
    time_s: float
    mode: GenerationMode | None = None


@dataclass(frozen=True)
class GenerationResult:
    """The packet generated at an event and its buffer insertion outcome."""

    packet: DataPacket
    buffer_result: BufferInsertResult


@dataclass(frozen=True)
class GenerationStats:
    """Counts since initialization plus the CH's live buffer size.

    inserted_packets includes packets subsequently evicted. overflow_dropped_packets
    includes both DROP_OLDEST evictions and DROP_NEWEST rejections. Oversized packets
    appear only in oversized_rejected_packets. CH counters provide generated totals.
    """

    ch_id: int
    generated_packets: int
    inserted_packets: int
    overflow_dropped_packets: int
    oversized_rejected_packets: int
    generated_bytes: int
    buffer_size_bytes: int


@dataclass
class _CHGeneration:
    ch: ClusterHead
    process: PoissonProcess | MMPPProcess
    packet_parameters: PacketParameters
    initial_generated_packets: int = 0
    initial_generated_bytes: int = 0
    inserted_packets: int = 0
    overflow_dropped_packets: int = 0
    oversized_rejected_packets: int = 0
    pending_event: GenerationEvent | None = None


@dataclass
class GenerationManager:
    """Generate packets for registered CHs over a finite mission interval.

    Register CHs, initialize once, then repeatedly peek_next_event and process_event.
    One pending time per CH is held and scanned in O(number of CHs) time. Equal
    times are ordered by ch_id, independently of registration order.

    The explicit schedule or MMPP config drives the process; the scalar generation_rate_pps
    is left untouched. Generation after initialization should pass through this
    manager so overflow statistics and the CH's generated totals cover the same work.
    IDs are unique across this manager's CHs. For continuing an existing mission,
    first_packet_id must exceed IDs previously used outside the registered buffers.
    """

    mission_deadline_s: float
    start_time_s: float = 0.0
    first_packet_id: int = 0
    _registrations: dict[int, _CHGeneration] = field(init=False, default_factory=dict, repr=False)
    _initialized: bool = field(init=False, default=False, repr=False)
    _next_packet_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.start_time_s = _nonnegative_float(self.start_time_s, "start_time_s")
        self.mission_deadline_s = _nonnegative_float(self.mission_deadline_s, "mission_deadline_s")
        if self.mission_deadline_s <= self.start_time_s:
            raise ValueError("mission_deadline_s must be greater than start_time_s")
        _require_int(self.first_packet_id, "first_packet_id")
        self._next_packet_id = self.first_packet_id

    def register_ch(
        self,
        ch: ClusterHead,
        schedule: RateSchedule,
        packet_parameters: PacketParameters,
        *,
        seed: int,
    ) -> None:
        """Register an entity with its own schedule, packet parameters and RNG seed."""
        self._validate_registration(ch, packet_parameters)
        process = PoissonProcess(schedule, self.mission_deadline_s, seed, self.start_time_s)
        self._registrations[ch.ch_id] = _CHGeneration(ch, process, packet_parameters)

    def register_mmpp_ch(
        self, ch: ClusterHead, config: MMPPConfig, packet_parameters: PacketParameters
    ) -> None:
        """Register a CH's MMPP with the independent seed stored in config."""
        self._validate_registration(ch, packet_parameters)
        process = MMPPProcess(config, self.mission_deadline_s, self.start_time_s)
        self._registrations[ch.ch_id] = _CHGeneration(ch, process, packet_parameters)

    def _validate_registration(self, ch: ClusterHead, packet_parameters: PacketParameters) -> None:
        if self._initialized:
            raise RuntimeError("register CHs before initializing generation")
        if not isinstance(ch, ClusterHead):
            raise ValueError("ch must be a ClusterHead")
        if ch.ch_id in self._registrations:
            raise ValueError(f"CH {ch.ch_id} is already registered")
        if not isinstance(packet_parameters, PacketParameters):
            raise ValueError("packet_parameters must be PacketParameters")

    def _sample_pending_event(self, ch_id: int, registration: _CHGeneration) -> None:
        process = registration.process
        registration.pending_event = None
        if isinstance(process, MMPPProcess):
            event = process.peek_next_event()
            if event is None:
                # Commit the final no-packet tail so full-run mode durations are complete.
                process.next_event()
            else:
                registration.pending_event = GenerationEvent(ch_id, event.time_s, event.mode)
        else:
            time_s = process.next_time_s()
            if time_s is not None:
                registration.pending_event = GenerationEvent(ch_id, time_s)

    def initialize(self) -> None:
        """Sample each CH's first event and remember its existing generation counters.

        Existing buffered IDs are checked and skipped by the shared ID allocator.
        Reinitialization is rejected to prevent replaying events into existing state.
        """
        if self._initialized:
            raise RuntimeError("generation is already initialized; use a new manager for a new run")
        existing_ids: set[int] = set()
        for registration in self._registrations.values():
            for packet in registration.ch.buffer:
                if packet.packet_id in existing_ids:
                    raise ValueError("registered CH buffers contain duplicate global packet IDs")
                existing_ids.add(packet.packet_id)
        self._next_packet_id = max(self.first_packet_id, max(existing_ids, default=-1) + 1)

        for ch_id, registration in self._registrations.items():
            registration.initial_generated_packets = registration.ch.total_generated_packets
            registration.initial_generated_bytes = registration.ch.total_generated_bytes
            self._sample_pending_event(ch_id, registration)
        self._initialized = True

    def peek_next_event(self) -> GenerationEvent | None:
        """Return the earliest pending event without consuming it or drawing randomness."""
        if not self._initialized:
            raise RuntimeError("initialize generation before querying events")
        events = (
            registration.pending_event
            for registration in self._registrations.values()
            if registration.pending_event is not None
        )
        return min(events, key=lambda event: (event.time_s, event.ch_id), default=None)

    def process_event(self, event: GenerationEvent) -> GenerationResult:
        """Generate the next pending event's packet and return its insertion outcome.

        Stale, fabricated, or out-of-order events fail before changing any state.
        The CH alone decides which packets to retain or discard.
        """
        if not isinstance(event, GenerationEvent):
            raise ValueError("event must be a GenerationEvent")
        if event != self.peek_next_event():
            raise ValueError("event must match the next pending generation event")
        registration = self._registrations[event.ch_id]
        parameters = registration.packet_parameters
        packet = DataPacket(
            packet_id=self._next_packet_id,
            source_ch_id=event.ch_id,
            generation_time_s=event.time_s,
            size_bytes=parameters.size_bytes,
            base_value=parameters.base_value,
            decay_rate_per_s=parameters.decay_rate_per_s,
        )
        result = registration.ch.generate_packet(packet)
        if isinstance(registration.process, MMPPProcess):
            registration.process.next_event()
        self._next_packet_id += 1
        registration.inserted_packets += int(result.inserted)
        if packet.size_bytes > registration.ch.buffer_capacity_bytes:
            registration.oversized_rejected_packets += 1
        else:
            registration.overflow_dropped_packets += len(result.dropped_packets)

        self._sample_pending_event(event.ch_id, registration)
        return GenerationResult(packet, result)

    def get_mode_stats(self, ch_id: int) -> MMPPStats:
        """Return simulation-only MMPP diagnostics with their accounting time.

        Counts cover processed generation events. Durations cover through the
        last processed event, or through the deadline once a stream is exhausted.
        These snapshots must not be used as AUV observations or predictor inputs.
        """
        if not self._initialized:
            raise RuntimeError("initialize generation before querying statistics")
        _require_int(ch_id, "ch_id")
        if ch_id not in self._registrations:
            raise KeyError(f"CH {ch_id} is not registered")
        process = self._registrations[ch_id].process
        if not isinstance(process, MMPPProcess):
            raise ValueError(f"CH {ch_id} uses Poisson generation and has no MMPP mode statistics")
        return process.get_stats()

    def get_stats(self, ch_id: int) -> GenerationStats:
        """Return generated counter differences and insertion/drop counts for a CH."""
        if not self._initialized:
            raise RuntimeError("initialize generation before querying statistics")
        _require_int(ch_id, "ch_id")
        if ch_id not in self._registrations:
            raise KeyError(f"CH {ch_id} is not registered")
        registration = self._registrations[ch_id]
        ch = registration.ch
        return GenerationStats(
            ch_id=ch_id,
            generated_packets=ch.total_generated_packets - registration.initial_generated_packets,
            inserted_packets=registration.inserted_packets,
            overflow_dropped_packets=registration.overflow_dropped_packets,
            oversized_rejected_packets=registration.oversized_rejected_packets,
            generated_bytes=ch.total_generated_bytes - registration.initial_generated_bytes,
            buffer_size_bytes=ch.buffer_size_bytes,
        )
