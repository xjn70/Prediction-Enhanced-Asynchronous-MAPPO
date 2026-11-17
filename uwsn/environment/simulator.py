"""Run CH generation, AUV navigation and packet transfers on one event clock."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from math import inf, nextafter
from typing import Literal

from ..data_generation import GenerationEvent, GenerationManager, GenerationResult
from ..entities import (
    AUV, AUVStatus, CHReport, ClusterHead, DataPacket, DeliveryRecord, PacketCompletion, Sink, UploadJob,
)
from ..entities.common import _nonnegative_float, _positive_float, _require_int
from ..navigation import TravelLeg, in_communication_range, plan_return, plan_visit
from .event_queue import EventQueue
from .events import EventType, SimulationEvent


@dataclass(frozen=True)
class CHArrival:
    """The CH report received on arrival and the result of the service request."""

    report: CHReport
    outcome: Literal["collection_started", "nothing_to_collect", "service_conflict", "deadline"]


@dataclass(frozen=True)
class CollectionResult:
    """Internal record of a packet transfer or its interruption at the deadline."""

    packet: DataPacket
    completed: bool


@dataclass(frozen=True)
class _AUVPlanEvent:
    token: int
    data: TravelLeg | DataPacket | PacketCompletion | None = None


EventResult = GenerationResult | CHArrival | CollectionResult | UploadJob | DeliveryRecord | str


@dataclass(frozen=True)
class EventLogEntry:
    """Simulation-only execution record, including the handler result."""

    time_s: float
    event_type: EventType
    entity_id: int | None
    result: EventResult


class Simulator:
    """A finite event run with a private copy of the initial scenario.

    Provide an uninitialized manager with the same [0, mission_duration_s)
    window. By default, cluster_heads are taken from its registrations. Reset
    restores entities and RNG state and clears manually scheduled AUV events.
    Provide a fresh Sink for the mission. Its queue and ledger, the event queue
    and the event log are simulation diagnostics.
    """

    def __init__(
        self,
        mission_duration_s: float,
        *,
        generation_manager: GenerationManager | None = None,
        cluster_heads: Iterable[ClusterHead] | None = None,
        auvs: Iterable[AUV] = (),
        sink: Sink | None = None,
        collection_rate_bytes_per_s: float = 1000.0,
        include_generation_counter: bool = False,
        record_log: bool = True,
    ) -> None:
        self.mission_duration_s = _positive_float(mission_duration_s, "mission_duration_s")
        self.collection_rate_bytes_per_s = _positive_float(
            collection_rate_bytes_per_s, "collection_rate_bytes_per_s"
        )
        if not isinstance(include_generation_counter, bool):
            raise ValueError("include_generation_counter must be a bool")
        if not isinstance(record_log, bool):
            raise ValueError("record_log must be a bool")
        self.include_generation_counter = include_generation_counter
        self.record_log = record_log
        manager = generation_manager
        if manager is None:
            manager = GenerationManager(self.mission_duration_s)
        if manager.start_time_s != 0 or manager.mission_deadline_s != self.mission_duration_s:
            raise ValueError("generation manager must use the simulator mission window")
        registered = manager.registered_cluster_heads
        heads = registered if cluster_heads is None else tuple(cluster_heads)
        vehicles = tuple(auvs)
        heads_by_id = {ch.ch_id: ch for ch in heads}
        vehicles_by_id = {auv.auv_id: auv for auv in vehicles}
        if len(heads_by_id) != len(heads) or len(vehicles_by_id) != len(vehicles):
            raise ValueError("CH and AUV IDs must be unique within each entity type")
        if any(heads_by_id.get(ch.ch_id) is not ch for ch in registered):
            raise ValueError("cluster_heads must contain the same entities registered for generation")
        if sink is not None and (
            sink.active_upload is not None or sink.waiting_queue or sink.delivered_records or sink.interrupted_uploads
        ):
            raise ValueError("the simulator requires a fresh Sink without upload history")
        self._initial_scenario = deepcopy((heads_by_id, vehicles_by_id, sink, manager))
        self.reset()

    @property
    def current_time_s(self) -> float:
        return self.event_queue.current_time_s

    @property
    def decision_ready(self) -> frozenset[int]:
        """Return the ready IDs. Submit simultaneous decisions in sorted ID order."""
        return frozenset(self._decision_ready)

    def create_ch_report(self, ch_id: int, *, include_generation_counter: bool = False) -> CHReport:
        """Read a live CH snapshot for simulation diagnostics."""
        _require_int(ch_id, "ch_id")
        ch = self.cluster_heads[ch_id]
        next_event = self.event_queue.peek()
        if (
            next_event is not None
            and next_event.time_s == self.current_time_s
            and next_event.event_type is EventType.CH_DATA_GENERATION
        ):
            raise RuntimeError("process generation events at the current time before reading a CH report")
        return ch.create_report(self.current_time_s, include_generation_counter=include_generation_counter)

    def get_auv_ch_report(self, auv_id: int, ch_id: int) -> CHReport | None:
        """Return the AUV's most recent in-range report, or None before its first visit."""
        _require_int(auv_id, "auv_id")
        _require_int(ch_id, "ch_id")
        if auv_id not in self.auvs or ch_id not in self.cluster_heads:
            raise ValueError("unknown AUV or CH ID")
        return self._auv_reports.get((auv_id, ch_id))

    def reset(self) -> None:
        """Restore the initial scenario and schedule its first events at time zero."""
        self.cluster_heads, self.auvs, self.sink, self.generation_manager = deepcopy(self._initial_scenario)
        self.event_queue = EventQueue()
        self.event_log: list[EventLogEntry] = []
        self._decision_ready: set[int] = set()
        self._plan_token = dict.fromkeys(self.auvs, 0)
        self._auv_reports: dict[tuple[int, int], CHReport] = {}
        self._finished = False
        initial_packet_ids = [packet.packet_id for ch in self.cluster_heads.values() for packet in ch.buffer]
        initial_packet_ids.extend(packet.packet_id for auv in self.auvs.values() for packet in auv.cargo)
        if len(set(initial_packet_ids)) != len(initial_packet_ids):
            raise ValueError("initial CH buffers and AUV cargo must have unique packet IDs")
        self.generation_manager.first_packet_id = max(
            self.generation_manager.first_packet_id, max(initial_packet_ids, default=-1) + 1
        )
        self.generation_manager.initialize(synchronize_clock=True)
        self._schedule_next_generation()
        self.event_queue.push(self.mission_duration_s, EventType.MISSION_END)

    def schedule_event(self, time_s: float, event_type: EventType, entity_id: int) -> SimulationEvent:
        """Schedule AUV decision readiness or wait completion.

        Scheduling replaces that AUV's previous plan and clears its readiness. Milestones at the
        deadline can execute before MISSION_END. Later ones are discarded there.
        """
        if self._finished:
            raise RuntimeError("the mission has ended")
        if event_type not in (EventType.AUV_DECISION_READY, EventType.WAIT_COMPLETE):
            raise ValueError("only AUV decision-ready and wait-complete events can be scheduled")
        _require_int(entity_id, "entity_id")
        if entity_id not in self.auvs:
            raise ValueError(f"unknown AUV ID {entity_id}")
        if self.auvs[entity_id].status is not AUVStatus.IDLE:
            raise RuntimeError("cannot schedule decision readiness for a busy AUV")
        event = self._schedule_auv_event(time_s, event_type, entity_id, replace_plan=True)
        self._decision_ready.discard(entity_id)
        return event

    def visit(self, auv_id: int, ch_id: int) -> TravelLeg:
        """Start a visit by planning to the CH's communication sphere using geometry alone."""
        auv = self._require_decision_ready(auv_id)
        _require_int(ch_id, "ch_id")
        ch = self.cluster_heads[ch_id]
        leg = plan_visit(auv.position, ch.position, ch.communication_radius_m, auv.speed_mps)
        self._schedule_arrival(auv_id, EventType.AUV_ARRIVAL_CH, leg)
        auv.target_ch_id = ch_id
        auv.status = AUVStatus.NAVIGATING
        self._decision_ready.remove(auv_id)
        return leg

    def return_to_sink(self, auv_id: int) -> TravelLeg:
        """Start a straight-line return followed by FCFS upload of the current cargo."""
        auv = self._require_decision_ready(auv_id)
        if self.sink is None:
            raise RuntimeError("the simulator has no Sink")
        leg = plan_return(auv.position, self.sink.position, auv.speed_mps)
        self._schedule_arrival(auv_id, EventType.AUV_ARRIVAL_SINK, leg)
        auv.target_ch_id = None
        auv.status = AUVStatus.RETURNING
        self._decision_ready.remove(auv_id)
        return leg

    def _require_decision_ready(self, auv_id: int) -> AUV:
        _require_int(auv_id, "auv_id")
        if self._finished or self.current_time_s >= self.mission_duration_s:
            raise RuntimeError("the mission has ended")
        auv = self.auvs[auv_id]
        if auv_id not in self._decision_ready or auv.status is not AUVStatus.IDLE:
            raise RuntimeError("AUV is not decision-ready")
        return auv

    def _schedule_arrival(self, auv_id: int, kind: EventType, leg: TravelLeg) -> None:
        arrival_s = self.current_time_s + leg.travel_time_s
        if leg.distance_m > 0:
            arrival_s = max(arrival_s, nextafter(self.current_time_s, inf))
        if arrival_s <= self.mission_duration_s:
            self._schedule_auv_event(arrival_s, kind, auv_id, leg, replace_plan=True)
        else:
            self._plan_token[auv_id] += 1

    def _schedule_auv_event(
        self, time_s: float, kind: EventType, auv_id: int,
        data: TravelLeg | DataPacket | PacketCompletion | None = None, *, replace_plan: bool = False,
    ) -> SimulationEvent:
        token = self._plan_token[auv_id] + int(replace_plan)
        event = self.event_queue.push(time_s, kind, auv_id, _AUVPlanEvent(token, data))
        self._plan_token[auv_id] = token
        return event

    def step_event(self) -> EventLogEntry | None:
        """Process the earliest event, or return None after mission end."""
        event = self.event_queue.peek()
        if self._finished or event is None:
            return None
        self.generation_manager.advance_to(event.time_s)
        self.event_queue.pop()
        result = self._handle_event(event)
        entry = EventLogEntry(event.time_s, event.event_type, event.entity_id, result)
        if self.record_log:
            self.event_log.append(entry)
        return entry

    def run_until(self, time_s: float) -> tuple[EventLogEntry, ...]:
        """Process events through time_s inclusive, capped at the mission deadline."""
        limit = min(_nonnegative_float(time_s, "time_s"), self.mission_duration_s)
        if limit < self.current_time_s:
            raise ValueError("cannot run the simulator backwards")
        records = []
        while not self._finished and (event := self.event_queue.peek()) is not None and event.time_s <= limit:
            entry = self.step_event()
            assert entry is not None
            records.append(entry)
        self.generation_manager.advance_to(limit)
        self.event_queue.advance_to(limit)
        return tuple(records)

    def run(self) -> tuple[EventLogEntry, ...]:
        """Run through the mission deadline and return the newly processed records."""
        return self.run_until(self.mission_duration_s)

    def _schedule_next_generation(self) -> None:
        # The manager retains one candidate per CH; enqueue only its global next.
        pending = self.generation_manager.peek_next_event()
        if pending is not None and pending.time_s < self.mission_duration_s:
            self.event_queue.push(pending.time_s, EventType.CH_DATA_GENERATION, pending.ch_id, pending)

    def _handle_event(self, event: SimulationEvent) -> EventResult:
        if event.event_type is EventType.CH_DATA_GENERATION:
            payload = event.payload
            if not isinstance(payload, GenerationEvent) or event.time_s >= self.mission_duration_s:
                raise RuntimeError("invalid scheduled CH generation event")
            result = self.generation_manager.process_event(payload)
            self._schedule_next_generation()
            return result
        if event.event_type is EventType.MISSION_END:
            if self.sink is not None:
                self.sink.interrupt_uploads(self.current_time_s)
                for upload in self.sink.interrupted_uploads:
                    self.auvs[upload.auv_id].status = AUVStatus.WAITING_SINK
            self._finished = True
            self.event_queue.clear()
            return "mission_ended"
        if not isinstance(event.payload, _AUVPlanEvent) or event.entity_id is None:
            raise NotImplementedError(f"no simulator handler for {event.event_type.value}")
        if event.payload.token != self._plan_token[event.entity_id]:
            return "stale_ignored"

        auv = self.auvs[event.entity_id]
        data = event.payload.data
        if event.event_type in (EventType.AUV_DECISION_READY, EventType.WAIT_COMPLETE):
            if auv.status is not AUVStatus.IDLE:
                return "decision_not_ready"
            self._decision_ready.add(event.entity_id)
            return "decision_ready"
        if event.event_type is EventType.AUV_ARRIVAL_CH:
            assert isinstance(data, TravelLeg)
            return self._arrive_at_ch(auv, data)
        if event.event_type is EventType.COLLECTION_COMPLETE:
            assert isinstance(data, DataPacket)
            return self._complete_packet(auv, data)
        if event.event_type is EventType.AUV_ARRIVAL_SINK:
            assert isinstance(data, TravelLeg)
            self._arrive_at_sink(auv, data)
            return "arrived_sink"
        if event.event_type is EventType.UPLOAD_START:
            return self._start_upload(auv)
        if event.event_type is EventType.PACKET_UPLOAD_COMPLETE:
            assert isinstance(data, PacketCompletion)
            return self._complete_upload_packet(auv, data)
        if event.event_type is EventType.UPLOAD_COMPLETE:
            return self._finish_upload(auv)
        raise NotImplementedError(f"no simulator handler for {event.event_type.value}")

    def _arrive_at_ch(self, auv: AUV, leg: TravelLeg) -> CHArrival:
        assert auv.target_ch_id is not None
        ch = self.cluster_heads[auv.target_ch_id]
        if not in_communication_range(leg.destination, ch.position, ch.communication_radius_m):
            raise RuntimeError("visited position is outside CH communication range")
        auv.position = leg.destination
        report = self.create_ch_report(ch.ch_id, include_generation_counter=self.include_generation_counter)
        self._auv_reports[(auv.auv_id, ch.ch_id)] = report
        service_window = self.mission_duration_s - self.current_time_s
        if service_window <= 0:
            self._finish_visit(auv)
            return CHArrival(report, "deadline")
        # Reconstructing the deadline from a relative lease can round it down.
        if self.current_time_s + service_window < self.mission_duration_s:
            service_window = nextafter(service_window, inf)
        if not ch.try_acquire_service_lock(auv.auv_id, self.current_time_s, service_window):
            self._finish_visit(auv)
            return CHArrival(report, "service_conflict")
        auv.status = AUVStatus.COLLECTING
        started = self._start_next_packet(auv, ch)
        return CHArrival(report, "collection_started" if started else "nothing_to_collect")

    def _start_next_packet(self, auv: AUV, ch: ClusterHead) -> bool:
        packet = ch.reserve_next_packet(auv.auv_id, auv.available_storage_bytes)
        if packet is None:
            self._finish_visit(auv, ch)
            return False
        try:
            duration_s = packet.size_bytes / self.collection_rate_bytes_per_s
        except OverflowError:
            duration_s = inf
        completion_s = max(self.current_time_s + duration_s, nextafter(self.current_time_s, inf))
        self._schedule_auv_event(
            min(completion_s, self.mission_duration_s), EventType.COLLECTION_COMPLETE, auv.auv_id, packet
        )
        return True

    def _complete_packet(self, auv: AUV, packet: DataPacket) -> CollectionResult:
        ch = self.cluster_heads[packet.source_ch_id]
        if self.current_time_s >= self.mission_duration_s:
            self._finish_visit(auv, ch)
            return CollectionResult(packet, completed=False)
        if not auv.can_load_packets((packet,)):
            self._finish_visit(auv, ch)
            raise RuntimeError("reserved packet no longer fits in AUV cargo")
        auv.load_packets((ch.complete_reserved_packet(auv.auv_id, packet.packet_id),))
        self._start_next_packet(auv, ch)
        return CollectionResult(packet, completed=True)

    def _finish_visit(self, auv: AUV, ch: ClusterHead | None = None) -> None:
        if ch is not None:
            ch.release_service_lock(auv.auv_id)
        auv.target_ch_id = None
        auv.status = AUVStatus.IDLE
        if self.current_time_s < self.mission_duration_s:
            self._schedule_auv_event(self.current_time_s, EventType.AUV_DECISION_READY, auv.auv_id)

    def _arrive_at_sink(self, auv: AUV, leg: TravelLeg) -> None:
        sink = self.sink
        assert sink is not None
        auv.position = leg.destination
        cargo = auv.cargo
        if not cargo:
            self._finish_return(auv)
            return

        queue_was_empty = not sink.waiting_queue
        sink.enqueue_upload(auv.auv_id, cargo, self.current_time_s)
        auv.status = AUVStatus.WAITING_SINK
        if queue_was_empty and sink.active_upload is None:
            self._schedule_upload_start()

    def _schedule_upload_start(self) -> None:
        sink = self.sink
        assert sink is not None
        queue = sink.waiting_queue
        if queue and self.current_time_s < self.mission_duration_s:
            self._schedule_auv_event(self.current_time_s, EventType.UPLOAD_START, queue[0].auv_id)

    def _start_upload(self, auv: AUV) -> UploadJob:
        sink = self.sink
        assert sink is not None
        job = sink.start_next_upload(self.current_time_s)
        if job is None or job.request.auv_id != auv.auv_id:
            raise RuntimeError("scheduled upload start does not match the FCFS request")
        auv.status = AUVStatus.UPLOADING
        self._schedule_upload_packet(auv, job.packet_completions[0])
        return job

    def _schedule_upload_packet(self, auv: AUV, completion: PacketCompletion) -> None:
        if completion.completion_time_s <= self.mission_duration_s:
            self._schedule_auv_event(
                completion.completion_time_s, EventType.PACKET_UPLOAD_COMPLETE, auv.auv_id, completion
            )

    def _complete_upload_packet(self, auv: AUV, completion: PacketCompletion) -> DeliveryRecord:
        sink = self.sink
        assert sink is not None
        job = sink.active_upload
        if job is None or job.request.auv_id != auv.auv_id:
            raise RuntimeError("packet completion does not match the active upload")
        if auv.get_packet(completion.packet_id) != completion.packet:
            raise RuntimeError("packet completion requires the original packet in the AUV cargo")
        record = sink.record_packet_delivery(completion.packet_id, self.current_time_s, self.mission_duration_s)
        auv.remove_packets((record.packet_id,))
        next_completion = sink.next_packet_completion
        if next_completion is None:
            self._schedule_auv_event(self.current_time_s, EventType.UPLOAD_COMPLETE, auv.auv_id)
        else:
            self._schedule_upload_packet(auv, next_completion)
        return record

    def _finish_upload(self, auv: AUV) -> UploadJob:
        sink = self.sink
        assert sink is not None
        job = sink.active_upload
        if job is None or job.request.auv_id != auv.auv_id:
            raise RuntimeError("upload completion does not match the active upload")
        sink.finish_active_upload(self.current_time_s, start_next=False)
        self._schedule_upload_start()
        self._finish_return(auv)
        return job

    def _finish_return(self, auv: AUV) -> None:
        auv.status = AUVStatus.IDLE
        if self.current_time_s < self.mission_duration_s:
            self._schedule_auv_event(self.current_time_s, EventType.AUV_DECISION_READY, auv.auv_id)
