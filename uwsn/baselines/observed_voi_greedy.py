"""Value-per-second ranking using only timestamped local CH reports."""

from math import fsum

from ..entities.common import _positive_float
from ..entities.data_packet import packet_voi_at_delivery
from ..environment.actions import Action, ReturnAction, VisitAction
from ..environment.observation import CHObservation, LocalObservation
from ..navigation import plan_return
from .nearest_neighbor import feasible_sites, nearest_visit


class ObservedVoIGreedy:
    def __init__(self, report_age_scale_s: float = 120.0) -> None:
        self.report_age_scale_s = _positive_float(report_age_scale_s, "report_age_scale_s")

    def choose_action(self, observation: LocalObservation) -> Action:
        actions = observation.legal_actions
        if actions and isinstance(actions[0], ReturnAction):
            return actions[0]
        sites = feasible_sites(observation)
        ranked = [(self.score(observation, ch), ch.ch_id) for ch in sites]
        if ranked:
            score, ch_id = max(ranked, key=lambda pair: (pair[0], -pair[1]))
            if score > 0:
                return VisitAction(ch_id)
        unknown = [ch for ch in sites if ch.report is None]
        return nearest_visit(observation, unknown or sites)

    def score(self, observation: LocalObservation, ch: CHObservation) -> float:
        """Estimate the value of a feasible FIFO prefix of a historical snapshot."""
        report, parameters = ch.report, ch.packet_parameters
        if report is None or parameters is None:
            return 0.0
        max_packets = min(
            len(report.packet_timestamps), observation.storage_remaining_bytes // parameters.size_bytes
        )
        leg = observation.visit_leg(ch)
        back_s = plan_return(leg.destination, observation.sink_position, observation.speed_mps).travel_time_s
        collect_s = parameters.size_bytes / observation.collection_rate_bytes_per_s
        upload_s = parameters.size_bytes / observation.sink_upload_rate_bytes_per_s
        cargo_upload_s = observation.storage_used_bytes / observation.sink_upload_rate_bytes_per_s
        arrival_without_collection_s = observation.current_time_s + leg.travel_time_s + back_s

        count = 0
        for candidate in range(1, max_packets + 1):
            last_delivery_s = arrival_without_collection_s + candidate * collect_s + cargo_upload_s + candidate * upload_s
            if last_delivery_s > observation.mission_deadline_s:
                break
            count = candidate
        if not count:
            return 0.0

        first_upload_s = arrival_without_collection_s + count * collect_s + cargo_upload_s
        value = fsum(
            packet_voi_at_delivery(
                timestamp, parameters.base_value, parameters.decay_rate_per_s,
                first_upload_s + index * upload_s, observation.mission_deadline_s,
                volume_weight=observation.volume_weight,
            )
            for index, timestamp in enumerate(report.packet_timestamps[:count], start=1)
        )
        age_s = observation.current_time_s - report.report_time_s
        history_weight = 1.0 / (1.0 + age_s / self.report_age_scale_s)
        duration_s = leg.travel_time_s + count * collect_s + back_s + cargo_upload_s + count * upload_s
        return history_weight * value / max(duration_s, 1e-9)
