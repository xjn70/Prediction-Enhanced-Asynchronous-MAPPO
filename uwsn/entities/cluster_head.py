"""Cluster head buffering, state reports, and exclusive service leases."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .common import (
    OverflowPolicy,
    Position3D,
    _finite_float,
    _nonnegative_float,
    _positive_float,
    _require_int,
    _validate_position,
)
from .data_packet import DataPacket, _remove_packet_batch


@dataclass(frozen=True)
class CHReport:
    """Local cache volume and generation times, with an opt-in local counter."""

    ch_id: int
    report_time_s: float
    buffer_size_bytes: int
    packet_timestamps: tuple[float, ...]
    total_generated_packets: int | None = None


@dataclass(frozen=True)
class BufferInsertResult:
    """Outcome of generating a packet, including all packets lost to overflow."""

    inserted: bool
    dropped_packets: tuple[DataPacket, ...] = ()


@dataclass
class ClusterHead:
    """A fixed CH with a bounded FIFO buffer and one service lease."""

    ch_id: int
    position: Position3D
    buffer_capacity_bytes: int
    communication_radius_m: float
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_NEWEST
    total_generated_packets: int = field(init=False, default=0)
    total_generated_bytes: int = field(init=False, default=0)
    service_lock_owner: int | None = field(init=False, default=None)
    service_lock_expiry_s: float | None = field(init=False, default=None)
    _buffer: dict[int, DataPacket] = field(init=False, default_factory=dict, repr=False)
    _buffer_size_bytes: int = field(init=False, default=0, repr=False)
    _latest_generation_time_s: float = field(init=False, default=0.0, repr=False)
    _reserved_packet_id: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        _require_int(self.ch_id, "ch_id")
        self.position = _validate_position(self.position)
        _require_int(self.buffer_capacity_bytes, "buffer_capacity_bytes")
        self.communication_radius_m = _positive_float(
            self.communication_radius_m, "communication_radius_m"
        )
        self.overflow_policy = OverflowPolicy(self.overflow_policy)

    @property
    def buffer(self) -> tuple[DataPacket, ...]:
        """Return a snapshot of buffered packets in FIFO order."""
        return tuple(self._buffer.values())

    @property
    def buffer_size_bytes(self) -> int:
        """Return the current number of bytes stored at the CH."""
        return self._buffer_size_bytes

    @property
    def reserved_packet(self) -> DataPacket | None:
        """Return the FIFO head in transmission, still counted in the CH buffer."""
        if self._reserved_packet_id is None:
            return None
        return self._buffer[self._reserved_packet_id]

    def generate_packet(self, packet: DataPacket) -> BufferInsertResult:
        """Count a generated packet and apply the configured overflow policy."""
        if not isinstance(packet, DataPacket):
            raise ValueError("packet must be a DataPacket")
        if packet.source_ch_id != self.ch_id:
            raise ValueError("packet source_ch_id must match this CH")
        if packet.packet_id in self._buffer:
            raise ValueError(f"packet_id {packet.packet_id} is already buffered")

        self.total_generated_packets += 1
        self.total_generated_bytes += packet.size_bytes
        self._latest_generation_time_s = max(self._latest_generation_time_s, packet.generation_time_s)
        reserved = self.reserved_packet
        reserved_bytes = reserved.size_bytes if reserved is not None else 0
        if packet.size_bytes + reserved_bytes > self.buffer_capacity_bytes:
            return BufferInsertResult(False, (packet,))

        free_bytes = self.buffer_capacity_bytes - self._buffer_size_bytes
        if packet.size_bytes > free_bytes and self.overflow_policy is OverflowPolicy.DROP_NEWEST:
            return BufferInsertResult(False, (packet,))

        dropped = []
        while packet.size_bytes > free_bytes:
            oldest_id = next(packet_id for packet_id in self._buffer if packet_id != self._reserved_packet_id)
            oldest = self._buffer.pop(oldest_id)
            dropped.append(oldest)
            free_bytes += oldest.size_bytes
        self._buffer[packet.packet_id] = packet
        self._buffer_size_bytes = self.buffer_capacity_bytes - free_bytes + packet.size_bytes
        return BufferInsertResult(True, tuple(dropped))

    def peek_packets(self, max_bytes: int) -> tuple[DataPacket, ...]:
        """Return the longest whole-packet FIFO prefix within the byte budget."""
        _require_int(max_bytes, "max_bytes")
        packets = []
        remaining_bytes = max_bytes
        for packet in self._buffer.values():
            if packet.size_bytes > remaining_bytes:
                break
            packets.append(packet)
            remaining_bytes -= packet.size_bytes
        return tuple(packets)

    def remove_packets(self, packet_ids: Iterable[int]) -> tuple[DataPacket, ...]:
        """Remove the selected packets after validating the entire request."""
        if not isinstance(packet_ids, Iterable):
            raise ValueError("packet_ids must be an iterable of integers")
        packet_ids = tuple(packet_ids)
        if self._reserved_packet_id is not None and self._reserved_packet_id in packet_ids:
            raise RuntimeError("cannot remove the packet reserved for transmission")
        removed = _remove_packet_batch(self._buffer, packet_ids)
        self._buffer_size_bytes -= sum(packet.size_bytes for packet in removed)
        return removed

    def reserve_next_packet(self, auv_id: int, max_bytes: int) -> DataPacket | None:
        """Reserve the FIFO head for the service owner if the whole packet fits."""
        _require_int(auv_id, "auv_id")
        _require_int(max_bytes, "max_bytes")
        if self.service_lock_owner != auv_id or self._reserved_packet_id is not None:
            raise RuntimeError("reservation requires ownership of a service lock and no active packet")
        packet = next(iter(self._buffer.values()), None)
        if packet is None or packet.size_bytes > max_bytes:
            return None
        self._reserved_packet_id = packet.packet_id
        return packet

    def complete_reserved_packet(self, auv_id: int, packet_id: int) -> DataPacket:
        """Remove a fully transmitted packet from the CH after verifying ownership."""
        _require_int(auv_id, "auv_id")
        _require_int(packet_id, "packet_id")
        if self.service_lock_owner != auv_id or self._reserved_packet_id != packet_id:
            raise RuntimeError("packet is not reserved for this AUV")
        packet = self._buffer.pop(packet_id)
        self._buffer_size_bytes -= packet.size_bytes
        self._reserved_packet_id = None
        return packet

    def create_report(
        self, report_time_s: float, include_generation_counter: bool = False
    ) -> CHReport:
        """Return a snapshot with the cumulative generation count only on request."""
        report_time_s = _nonnegative_float(report_time_s, "report_time_s")
        if not isinstance(include_generation_counter, bool):
            raise ValueError("include_generation_counter must be a bool")
        if report_time_s < self._latest_generation_time_s:
            raise ValueError("report_time_s cannot precede the latest packet generation")
        stamps = tuple(packet.generation_time_s for packet in self._buffer.values())
        counter = self.total_generated_packets if include_generation_counter else None
        return CHReport(self.ch_id, report_time_s, self.buffer_size_bytes, stamps, counter)

    def try_acquire_service_lock(
        self, auv_id: int, now_s: float, lease_duration_s: float
    ) -> bool:
        """Acquire a free or expired lease, or renew the caller's lease."""
        _require_int(auv_id, "auv_id")
        now_s = _nonnegative_float(now_s, "now_s")
        lease_duration_s = _positive_float(lease_duration_s, "lease_duration_s")
        expiry_s = self.service_lock_expiry_s
        if expiry_s is not None and now_s < expiry_s and self.service_lock_owner != auv_id:
            return False
        new_expiry_s = _finite_float(now_s + lease_duration_s, "service_lock_expiry_s")
        if new_expiry_s <= now_s:
            raise ValueError("lease_duration_s is too short to advance the clock")
        if self.service_lock_owner != auv_id:
            self._reserved_packet_id = None
        self.service_lock_owner = auv_id
        self.service_lock_expiry_s = new_expiry_s
        return True

    def release_service_lock(self, auv_id: int) -> bool:
        """Release the caller's lease, leaving any unfinished packet in the CH."""
        _require_int(auv_id, "auv_id")
        if self.service_lock_owner != auv_id:
            return False
        self.service_lock_owner = None
        self.service_lock_expiry_s = None
        self._reserved_packet_id = None
        return True
