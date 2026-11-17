"""Shared navigation feasibility and return rule based on local data."""

from dataclasses import dataclass

from ..entities import AUVStatus
from ..entities.common import _nonnegative_float, _positive_float
from ..navigation import plan_return
from .actions import Action, ReturnAction, VisitAction
from .observation import LocalObservation


@dataclass(frozen=True)
class ReturnRuleConfig:
    storage_threshold: float
    return_reserve_s: float
    revisit_cooldown_s: float

    def __post_init__(self) -> None:
        storage_threshold = _positive_float(self.storage_threshold, "storage_threshold")
        if storage_threshold > 1:
            raise ValueError("storage_threshold must be at most 1")
        object.__setattr__(self, "storage_threshold", storage_threshold)
        object.__setattr__(self, "return_reserve_s", _nonnegative_float(self.return_reserve_s, "return_reserve_s"))
        object.__setattr__(self, "revisit_cooldown_s", _positive_float(self.revisit_cooldown_s, "revisit_cooldown_s"))


@dataclass(frozen=True)
class ActionOptions:
    actions: tuple[Action, ...]
    reason: str
    wake_at_s: float | None = None


def evaluate_actions(observation: LocalObservation, config: ReturnRuleConfig) -> ActionOptions:
    """Select feasible visits or a compulsory return using public rates and geometry."""
    obs = observation
    remaining_s = obs.mission_remaining_time_s
    if not obs.decision_ready or obs.status is not AUVStatus.IDLE or remaining_s <= 0:
        return ActionOptions((), "not_ready")
    upload_rate = obs.sink_upload_rate_bytes_per_s
    return_s = plan_return(obs.position, obs.sink_position, obs.speed_mps).travel_time_s
    required_s = return_s + obs.storage_used_bytes / upload_rate
    can_return = bool(obs.cargo) and return_s + obs.cargo[0].size_bytes / upload_rate <= remaining_s
    if can_return:
        if obs.storage_used_bytes >= config.storage_threshold * obs.storage_capacity_bytes:
            return ActionOptions((ReturnAction(),), "cargo_threshold")
        if required_s + config.return_reserve_s >= remaining_s:
            return ActionOptions((ReturnAction(),), "deadline_reserve")

    visits = []
    retry_at = []
    for ch in obs.chs:
        # Unregistered CHs have no public packet profile. One byte is then a lower bound.
        size = ch.packet_parameters.size_bytes if ch.packet_parameters is not None else 1
        if size > obs.storage_remaining_bytes:
            continue
        leg = obs.visit_leg(ch)
        back_s = plan_return(leg.destination, obs.sink_position, obs.speed_mps).travel_time_s
        min_trip_s = (leg.travel_time_s + size / obs.collection_rate_bytes_per_s + back_s
                      + (obs.storage_used_bytes + size) / upload_rate)
        available_at_s = max(obs.current_time_s, ch.revisit_after_s)
        if min_trip_s > obs.mission_deadline_s - available_at_s:
            continue
        if available_at_s > obs.current_time_s:
            retry_at.append(available_at_s)
        else:
            visits.append(VisitAction(ch.ch_id))
    if visits:
        return ActionOptions(tuple(visits), "feasible_visits")
    if can_return:
        return ActionOptions((ReturnAction(),), "no_feasible_visit")
    if retry_at:
        return ActionOptions((), "visit_cooldown", min(retry_at))
    return ActionOptions((), "no_feasible_action")
