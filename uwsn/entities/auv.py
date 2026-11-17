"""AUV state and bounded packet cargo."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .common import AUVStatus, Position3D, _positive_float, _require_int, _validate_position
from .data_packet import DataPacket, _packet_batch, _remove_packet_batch


@dataclass
class AUV:
    """Track an AUV's current state and its cargo, initially empty."""

    auv_id: int
    position: Position3D
    speed_mps: float
    storage_capacity_bytes: int
    status: AUVStatus = AUVStatus.IDLE
    target_ch_id: int | None = None
    _cargo: dict[int, DataPacket] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _require_int(self.auv_id, "auv_id")
        self.position = _validate_position(self.position)
        self.speed_mps = _positive_float(self.speed_mps, "speed_mps")
        _require_int(self.storage_capacity_bytes, "storage_capacity_bytes")
        self.status = AUVStatus(self.status)
        if self.target_ch_id is not None:
            _require_int(self.target_ch_id, "target_ch_id")

    @property
    def cargo(self) -> tuple[DataPacket, ...]:
        """Return a snapshot of cargo in loading order."""
        return tuple(self._cargo.values())

    @property
    def storage_used_bytes(self) -> int:
        """Return the byte size of the current cargo."""
        return sum(packet.size_bytes for packet in self._cargo.values())

    @property
    def available_storage_bytes(self) -> int:
        """Return the available cargo space in bytes."""
        return self.storage_capacity_bytes - self.storage_used_bytes

    def get_packet(self, packet_id: int) -> DataPacket | None:
        """Return a cargo packet by ID without changing its storage or order."""
        _require_int(packet_id, "packet_id")
        return self._cargo.get(packet_id)

    def can_load_packets(self, packets: Iterable[DataPacket]) -> bool:
        """Check if the entire packet batch fits and contains only new IDs."""
        return self._load_error(_packet_batch(packets)) is None

    def load_packets(self, packets: Iterable[DataPacket]) -> None:
        """Load the entire batch or raise ValueError before changing the cargo."""
        batch = _packet_batch(packets)
        error = self._load_error(batch)
        if error is not None:
            raise ValueError(error)
        self._cargo.update((packet.packet_id, packet) for packet in batch)

    def remove_packets(self, packet_ids: Iterable[int]) -> tuple[DataPacket, ...]:
        """Remove the selected cargo after validating the entire request."""
        return _remove_packet_batch(self._cargo, packet_ids)

    def _load_error(self, batch: tuple[DataPacket, ...]) -> str | None:
        packet_ids = {packet.packet_id for packet in batch}
        if len(packet_ids) != len(batch) or not packet_ids.isdisjoint(self._cargo):
            return "packet IDs must be unique within the batch and absent from the cargo"
        if sum(packet.size_bytes for packet in batch) > self.available_storage_bytes:
            return "packet batch exceeds the available AUV storage"
        return None
