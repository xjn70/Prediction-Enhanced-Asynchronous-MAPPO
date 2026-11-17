"""Read-only observations containing public geometry and one AUV's history."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..data_generation import PacketParameters
from ..entities import AUVStatus, CHReport, Position3D
from ..navigation import TravelLeg, plan_visit
from .actions import Action
from .simulator import Simulator


@dataclass(frozen=True)
class CargoPacket:
    source_ch_id: int
    generation_time_s: float
    size_bytes: int
    base_value: float
    decay_rate_per_s: float


@dataclass(frozen=True)
class CHObservation:
    ch_id: int
    position: Position3D
    communication_radius_m: float
    packet_parameters: PacketParameters | None
    report: CHReport | None  # None denotes UNKNOWN.
    revisit_after_s: float = 0.0


@dataclass(frozen=True)
class MacroActionResult:
    action: Action
    outcome: str
    finished_at_s: float
    collected_packets: int = 0
    collected_bytes: int = 0


@dataclass(frozen=True)
class LocalObservation:
    auv_id: int
    current_time_s: float
    mission_deadline_s: float
    position: Position3D
    status: AUVStatus
    decision_ready: bool
    speed_mps: float
    storage_used_bytes: int
    storage_capacity_bytes: int
    cargo: tuple[CargoPacket, ...]
    last_action_result: MacroActionResult | None
    chs: tuple[CHObservation, ...]
    sink_position: Position3D
    collection_rate_bytes_per_s: float
    sink_upload_rate_bytes_per_s: float
    volume_weight: float
    legal_actions: tuple[Action, ...] = ()

    @property
    def mission_remaining_time_s(self) -> float:
        return self.mission_deadline_s - self.current_time_s

    @property
    def storage_remaining_bytes(self) -> int:
        return self.storage_capacity_bytes - self.storage_used_bytes

    def visit_leg(self, ch: CHObservation) -> TravelLeg:
        """Use the same public communication-sphere geometry as the simulator."""
        return plan_visit(self.position, ch.position, ch.communication_radius_m, self.speed_mps)


def build_local_observation(
    simulator: Simulator,
    auv_id: int,
    *,
    last_action_result: MacroActionResult | None = None,
    cooldowns: Mapping[int, float] | None = None,
) -> LocalObservation:
    """Copy static metadata and the last reports actually received by this AUV."""
    auv = simulator.auvs[auv_id]
    sink = simulator.sink
    if sink is None:
        raise ValueError("a fixed Sink is required for baseline observations")
    cooldowns = {} if cooldowns is None else cooldowns
    chs = tuple(
        CHObservation(
            ch_id=ch.ch_id,
            position=ch.position,
            communication_radius_m=ch.communication_radius_m,
            packet_parameters=simulator.generation_manager.public_packet_parameters(ch.ch_id),
            report=simulator.get_auv_ch_report(auv_id, ch.ch_id),
            revisit_after_s=cooldowns.get(ch.ch_id, 0.0),
        )
        for _, ch in sorted(simulator.cluster_heads.items())
    )
    cargo = tuple(
        CargoPacket(packet.source_ch_id, packet.generation_time_s, packet.size_bytes,
                    packet.base_value, packet.decay_rate_per_s)
        for packet in auv.cargo
    )
    return LocalObservation(
        auv_id=auv_id,
        current_time_s=simulator.current_time_s,
        mission_deadline_s=simulator.mission_duration_s,
        position=auv.position,
        status=auv.status,
        decision_ready=auv_id in simulator.decision_ready,
        speed_mps=auv.speed_mps,
        storage_used_bytes=auv.storage_used_bytes,
        storage_capacity_bytes=auv.storage_capacity_bytes,
        cargo=cargo,
        last_action_result=last_action_result,
        chs=chs,
        sink_position=sink.position,
        collection_rate_bytes_per_s=simulator.collection_rate_bytes_per_s,
        sink_upload_rate_bytes_per_s=sink.upload_rate_bytes_per_s,
        volume_weight=sink.volume_weight,
    )
