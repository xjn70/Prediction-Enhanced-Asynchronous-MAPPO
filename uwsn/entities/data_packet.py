"""Immutable data packets and their value on delivery to the Sink."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp

from .common import _nonnegative_float, _require_int


@dataclass(frozen=True)
class DataPacket:
    """A packet with an immutable source, generation time, and VoI parameters."""

    packet_id: int
    source_ch_id: int
    generation_time_s: float
    size_bytes: int
    base_value: float
    decay_rate_per_s: float

    def __post_init__(self) -> None:
        _require_int(self.packet_id, "packet_id")
        _require_int(self.source_ch_id, "source_ch_id")
        _require_int(self.size_bytes, "size_bytes", minimum=1)
        for name in ("generation_time_s", "base_value", "decay_rate_per_s"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))

    def voi_at_delivery(self, delivery_time_s: float, mission_deadline_s: float) -> float:
        """Return the packet value at delivery, including at the exact deadline."""
        delivery_time_s = _nonnegative_float(delivery_time_s, "delivery_time_s")
        mission_deadline_s = _nonnegative_float(mission_deadline_s, "mission_deadline_s")
        if delivery_time_s < self.generation_time_s:
            raise ValueError("delivery_time_s cannot precede generation_time_s")
        if delivery_time_s > mission_deadline_s:
            return 0.0
        age_s = delivery_time_s - self.generation_time_s
        return self.base_value * exp(-self.decay_rate_per_s * age_s)


def _packet_batch(packets: Iterable[DataPacket]) -> tuple[DataPacket, ...]:
    if not isinstance(packets, Iterable):
        raise ValueError("packets must be an iterable of DataPacket objects")
    batch = tuple(packets)
    if any(not isinstance(packet, DataPacket) for packet in batch):
        raise ValueError("packets must contain only DataPacket objects")
    return batch


def _remove_packet_batch(
    packets_by_id: dict[int, DataPacket], packet_ids: Iterable[int]
) -> tuple[DataPacket, ...]:
    """Validate every ID before removing any packet, preserving request order."""
    if not isinstance(packet_ids, Iterable):
        raise ValueError("packet_ids must be an iterable of integers")
    ids = tuple(packet_ids)
    for packet_id in ids:
        _require_int(packet_id, "packet_id")
    if len(set(ids)) != len(ids):
        raise ValueError("packet_ids must be unique")
    for packet_id in ids:
        if packet_id not in packets_by_id:
            raise KeyError(f"packet_id {packet_id} is not present")
    return tuple(packets_by_id.pop(packet_id) for packet_id in ids)
