"""Single-channel FCFS uploads and packet-level deliveries at the Sink."""

from bisect import insort_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import fsum, ulp

from .common import (
    Position3D,
    _finite_float,
    _nonnegative_float,
    _positive_float,
    _require_int,
    _validate_position,
)
from .data_packet import DEFAULT_VOLUME_WEIGHT, DataPacket, _packet_batch, _validate_volume_weight


@dataclass(frozen=True)
class UploadRequest:
    """An AUV arrival with packets in the requested upload order."""

    auv_id: int
    packets: tuple[DataPacket, ...]
    arrival_time_s: float


@dataclass(frozen=True)
class PacketCompletion:
    """The scheduled completion of one packet in an upload."""

    packet: DataPacket
    completion_time_s: float

    @property
    def packet_id(self) -> int:
        """Return the ID to use when processing the packet completion."""
        return self.packet.packet_id


@dataclass(frozen=True)
class UploadJob:
    """A request assigned to the channel and its packet completion schedule."""

    request: UploadRequest
    start_time_s: float
    packet_completions: tuple[PacketCompletion, ...]

    @property
    def finish_time_s(self) -> float:
        """Return the completion time of the final packet."""
        return self.packet_completions[-1].completion_time_s


@dataclass(frozen=True)
class DeliveryRecord:
    """The delivered packet, the AUV that uploaded it, and the realized VoI."""

    auv_id: int
    packet: DataPacket
    delivery_time_s: float
    voi: float

    @property
    def packet_id(self) -> int:
        """Return the ID to remove from the delivering AUV's cargo."""
        return self.packet.packet_id


@dataclass
class Sink:
    """Serve one upload at a time and track completed packets."""

    position: Position3D
    upload_rate_bytes_per_s: float
    volume_weight: float = DEFAULT_VOLUME_WEIGHT
    _waiting_queue: list[UploadRequest] = field(init=False, default_factory=list, repr=False)
    _active_upload: UploadJob | None = field(init=False, default=None)
    _delivered_records: list[DeliveryRecord] = field(init=False, default_factory=list, repr=False)
    _delivered_packet_ids: set[int] = field(init=False, default_factory=set, repr=False)
    _pending_packet_ids: set[int] = field(init=False, default_factory=set, repr=False)
    _next_completion_index: int = field(init=False, default=0, repr=False)
    _channel_released_at_s: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.position = _validate_position(self.position)
        self.upload_rate_bytes_per_s = _positive_float(
            self.upload_rate_bytes_per_s, "upload_rate_bytes_per_s"
        )
        self.volume_weight = _validate_volume_weight(self.volume_weight)

    @property
    def waiting_queue(self) -> tuple[UploadRequest, ...]:
        """Return the requests waiting for service in FCFS order."""
        return tuple(self._waiting_queue)

    @property
    def active_upload(self) -> UploadJob | None:
        """Return the current upload, if the channel is occupied."""
        return self._active_upload

    @property
    def delivered_records(self) -> tuple[DeliveryRecord, ...]:
        """Return the recorded deliveries in completion order."""
        return tuple(self._delivered_records)

    @property
    def delivered_packet_ids(self) -> frozenset[int]:
        """Return the IDs already delivered to the Sink."""
        return frozenset(self._delivered_packet_ids)

    @property
    def total_voi(self) -> float:
        """Sum the realized values of recorded deliveries."""
        return fsum(record.voi for record in self._delivered_records)

    def enqueue_upload(
        self, auv_id: int, packets: Iterable[DataPacket], arrival_time_s: float
    ) -> UploadRequest:
        """Queue a nonempty upload, preserving insertion order for tied arrivals."""
        _require_int(auv_id, "auv_id")
        arrival_time_s = _nonnegative_float(arrival_time_s, "arrival_time_s")
        batch = _packet_batch(packets)
        if not batch:
            raise ValueError("an upload request must contain at least one packet")

        packet_ids = {packet.packet_id for packet in batch}
        if len(packet_ids) != len(batch):
            raise ValueError("packet IDs must be unique within an upload request")
        if not packet_ids.isdisjoint(self._pending_packet_ids):
            raise ValueError("an upload packet_id is already queued or in service")
        if not packet_ids.isdisjoint(self._delivered_packet_ids):
            raise ValueError("an upload packet_id has already been delivered")
        if any(packet.generation_time_s > arrival_time_s for packet in batch):
            raise ValueError("arrival_time_s cannot precede an upload packet's generation")

        request = UploadRequest(auv_id, batch, arrival_time_s)
        insort_right(self._waiting_queue, request, key=lambda item: item.arrival_time_s)
        self._pending_packet_ids.update(packet_ids)
        return request

    def start_next_upload(self, current_time_s: float) -> UploadJob | None:
        """Start the first arrived request if the channel is free."""
        current_time_s = _nonnegative_float(current_time_s, "current_time_s")
        if current_time_s < self._channel_released_at_s:
            raise ValueError("current_time_s cannot precede the last channel release")
        if self._active_upload is not None or not self._waiting_queue:
            return None
        request = self._waiting_queue[0]
        if request.arrival_time_s > current_time_s:
            return None

        cumulative_bytes = 0
        completions = []
        previous_completion_s = current_time_s
        for packet in request.packets:
            cumulative_bytes += packet.size_bytes
            try:
                upload_time_s = cumulative_bytes / self.upload_rate_bytes_per_s
            except OverflowError as error:
                raise ValueError("upload duration is outside the finite time range") from error
            completion_time_s = _finite_float(
                current_time_s + upload_time_s, "completion_time_s"
            )
            if completion_time_s <= previous_completion_s:
                raise ValueError("packet upload duration is too short to advance the clock")
            completions.append(PacketCompletion(packet, completion_time_s))
            previous_completion_s = completion_time_s

        job = UploadJob(request, current_time_s, tuple(completions))
        self._waiting_queue.pop(0)
        self._active_upload = job
        self._next_completion_index = 0
        return job

    def record_packet_delivery(
        self, packet_id: int, delivery_time_s: float, mission_deadline_s: float
    ) -> DeliveryRecord:
        """Record the next scheduled packet and realize its value at completion."""
        _require_int(packet_id, "packet_id")
        delivery_time_s = _nonnegative_float(delivery_time_s, "delivery_time_s")
        if packet_id in self._delivered_packet_ids:
            raise RuntimeError(f"packet_id {packet_id} has already been delivered")
        job = self._active_upload
        if job is None:
            raise RuntimeError("there is no active upload")

        if self._next_completion_index == len(job.packet_completions):
            raise KeyError(f"packet_id {packet_id} is not in the active upload")
        completion = job.packet_completions[self._next_completion_index]
        if packet_id != completion.packet_id:
            if any(item.packet_id == packet_id for item in job.packet_completions):
                raise RuntimeError("packet deliveries must follow the upload order")
            raise KeyError(f"packet_id {packet_id} is not in the active upload")

        expected_time_s = completion.completion_time_s
        if (
            delivery_time_s < completion.packet.generation_time_s
            or abs(delivery_time_s - expected_time_s) > 4 * ulp(expected_time_s)
        ):
            raise ValueError(f"delivery_time_s must match scheduled time {expected_time_s}")
        # Use the scheduled time for the deadline test even if the caller rounded it.
        voi = completion.packet.voi_at_delivery(
            expected_time_s, mission_deadline_s, volume_weight=self.volume_weight
        )
        record = DeliveryRecord(job.request.auv_id, completion.packet, expected_time_s, voi)
        self._delivered_records.append(record)
        self._delivered_packet_ids.add(packet_id)
        self._pending_packet_ids.remove(packet_id)
        self._next_completion_index += 1
        return record

    def finish_active_upload(self, current_time_s: float) -> UploadJob | None:
        """Release a completed channel and start the next arrived request."""
        current_time_s = _nonnegative_float(current_time_s, "current_time_s")
        job = self._active_upload
        if job is None:
            raise RuntimeError("there is no active upload")
        if current_time_s < job.finish_time_s:
            raise RuntimeError("cannot finish an upload before its scheduled finish time")
        if self._next_completion_index != len(job.packet_completions):
            raise RuntimeError("cannot finish an upload with undelivered packets")
        self._active_upload = None
        self._channel_released_at_s = current_time_s
        return self.start_next_upload(current_time_s)
