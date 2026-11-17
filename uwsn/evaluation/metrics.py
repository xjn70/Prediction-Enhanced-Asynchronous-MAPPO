"""Metrics from completed simulator events and the actual Sink ledger."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, fields
from math import fsum
from statistics import mean, stdev
from typing import TYPE_CHECKING

from ..entities import UploadJob
from ..environment.actions import VisitAction
from ..environment.simulator import Simulator

if TYPE_CHECKING:
    from .baseline_runner import DecisionRecord


@dataclass(frozen=True)
class EpisodeMetrics:
    total_delivered_voi: float
    delivered_packets: int
    delivered_bytes: int
    generated_packets: int
    dropped_packets: int
    undelivered_packets: int
    buffered_packets: int
    carried_packets: int
    average_delivery_delay_s: float | None
    average_sink_wait_s: float | None
    sink_busy_time_s: float
    sink_observed_utilization: float
    sink_unstarted_upload_requests: int
    total_auv_travel_distance_m: float
    successful_ch_visits: int
    rejected_ch_visits: int
    empty_ch_visits: int
    per_ch_visit_counts: tuple[tuple[int, int], ...]


def calculate_metrics(
    simulator: Simulator,
    decisions: Iterable["DecisionRecord"],
    jobs: Iterable[UploadJob],
    *,
    generated_packets: int,
    dropped_packets: int,
    initial_packets: int = 0,
) -> EpisodeMetrics:
    """Summarize a finished episode. Visit counts include all actual CH arrivals."""
    sink = simulator.sink
    if sink is None:
        raise ValueError("a fixed Sink is required for delivery metrics")
    deliveries = sink.delivered_records
    jobs = tuple(jobs)
    deadline = simulator.mission_duration_s
    buffered = sum(len(ch.buffer) for ch in simulator.cluster_heads.values())
    carried = sum(len(auv.cargo) for auv in simulator.auvs.values())
    undelivered = initial_packets + generated_packets - len(deliveries)
    if undelivered != buffered + carried + dropped_packets:
        raise RuntimeError("packet counts do not balance across CHs, AUVs, drops and the Sink")

    visit_counts: Counter[int] = Counter()
    successful = rejected = empty = 0
    distances = []
    for record in decisions:
        if record.arrived_at_s is not None:
            distances.append(record.travel.distance_m)
        elif record.travel.travel_time_s > 0:
            elapsed = deadline - record.decision_time_s
            distances.append(record.travel.distance_m * min(1.0, elapsed / record.travel.travel_time_s))
        if isinstance(record.action, VisitAction) and record.arrived_at_s is not None:
            visit_counts[record.action.ch_id] += 1
            successful += int(record.collected_packets > 0)
            rejected += int(record.outcome == "service_conflict")
            empty += int(record.outcome == "nothing_to_collect")

    busy_s = fsum(min(job.finish_time_s, deadline) - job.start_time_s for job in jobs)
    return EpisodeMetrics(
        total_delivered_voi=sink.total_voi,
        delivered_packets=len(deliveries),
        delivered_bytes=sum(record.size_bytes for record in deliveries),
        generated_packets=generated_packets,
        dropped_packets=dropped_packets,
        undelivered_packets=undelivered,
        buffered_packets=buffered,
        carried_packets=carried,
        average_delivery_delay_s=(
            mean(record.delivery_time_s - record.generation_time_s for record in deliveries) if deliveries else None
        ),
        average_sink_wait_s=(
            mean(job.start_time_s - job.request.arrival_time_s for job in jobs) if jobs else None
        ),
        sink_busy_time_s=busy_s,
        sink_observed_utilization=busy_s / deadline,
        sink_unstarted_upload_requests=sum(item.start_time_s is None for item in sink.interrupted_uploads),
        total_auv_travel_distance_m=fsum(distances),
        successful_ch_visits=successful,
        rejected_ch_visits=rejected,
        empty_ch_visits=empty,
        per_ch_visit_counts=tuple((ch_id, visit_counts[ch_id]) for ch_id in sorted(simulator.cluster_heads)),
    )


def metric_columns(metrics: EpisodeMetrics) -> dict[str, int | float | None]:
    """Flatten metrics into common CSV columns, including each CH."""
    row = {
        field.name: getattr(metrics, field.name)
        for field in fields(metrics)
        if field.name != "per_ch_visit_counts"
    }
    row.update((f"ch_{ch_id}_visits", count) for ch_id, count in metrics.per_ch_visit_counts)
    return row


@dataclass(frozen=True)
class SampleSummary:
    mean: float | None
    std: float | None
    sample_count: int


def summarize_metrics(episodes: Iterable[EpisodeMetrics]) -> dict[str, SampleSummary]:
    """Aggregate by metric. Use sample standard deviation and omit missing observations."""
    columns: dict[str, list[float]] = {}
    for episode in episodes:
        for key, value in metric_columns(episode).items():
            column = columns.setdefault(key, [])
            if value is not None:
                column.append(value)
    return {
        key: SampleSummary(
            mean(values) if values else None,
            stdev(values) if len(values) > 1 else (0.0 if values else None),
            len(values),
        )
        for key, values in columns.items()
    }
