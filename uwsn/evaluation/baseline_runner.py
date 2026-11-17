"""Run local-observation policies on the existing asynchronous simulator."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from math import inf, nextafter
from typing import Protocol

from ..baselines import CyclicPatrol, NearestNeighbor, ObservedVoIGreedy
from ..data_generation import GenerationResult
from ..entities import DeliveryRecord, UploadJob
from ..environment.action_rules import ReturnRuleConfig, evaluate_actions
from ..environment.actions import Action, ReturnAction, VisitAction
from ..environment.events import EventType
from ..environment.observation import LocalObservation, MacroActionResult, build_local_observation
from ..environment.simulator import CHArrival, CollectionResult, EventLogEntry, Simulator
from ..navigation import TravelLeg
from .metrics import EpisodeMetrics, calculate_metrics


class Policy(Protocol):
    def choose_action(self, observation: LocalObservation) -> Action: ...


PolicyFactory = Callable[[int], Policy]
ScenarioFactory = Callable[[int], Simulator]

BASELINE_POLICIES: tuple[tuple[str, PolicyFactory], ...] = (
    ("cyclic_patrol", CyclicPatrol),
    ("nearest_neighbor", lambda _: NearestNeighbor()),
    ("observed_voi_greedy", lambda _: ObservedVoIGreedy()),
)


@dataclass
class DecisionRecord:
    auv_id: int
    decision_time_s: float
    action: Action
    reason: str
    report_time_s: float | None
    travel: TravelLeg
    arrived_at_s: float | None = None
    arrival_report_time_s: float | None = None
    outcome: str = "in_progress"
    finished_at_s: float | None = None
    collected_packets: int = 0
    collected_bytes: int = 0
    upload_started_at_s: float | None = None
    upload_finished_at_s: float | None = None
    uploaded_packets: int = 0


@dataclass(frozen=True)
class NoActionRecord:
    auv_id: int
    decision_time_s: float
    reason: str
    wake_at_s: float | None
    rejected_action: Action | None = None


@dataclass(frozen=True)
class EpisodeResult:
    metrics: EpisodeMetrics
    decisions: tuple[DecisionRecord, ...]
    inactive_decisions: tuple[NoActionRecord, ...]
    generation_trace: tuple[tuple[int, float, int], ...]
    deliveries: tuple[DeliveryRecord, ...]


@dataclass(frozen=True)
class BaselineResult:
    policy: str
    seed: int
    episode: EpisodeResult


class _EpisodeRun:
    def __init__(self, simulator: Simulator, policy_factory: PolicyFactory, config: ReturnRuleConfig) -> None:
        if simulator.sink is None:
            raise ValueError("baseline episodes require a fixed Sink")
        self.sim = simulator
        self.config = config
        self.policies = {auv_id: policy_factory(auv_id) for auv_id in simulator.auvs}
        self.cooldowns: dict[int, dict[int, float]] = {auv_id: {} for auv_id in simulator.auvs}
        self.last_results: dict[int, MacroActionResult] = {}
        self.active: dict[int, DecisionRecord] = {}
        self.decisions: list[DecisionRecord] = []
        self.inactive: list[NoActionRecord] = []
        self.generated: list[tuple[int, float, int]] = []
        self.dropped = 0
        self.jobs: list[UploadJob] = []

    def complete(self, record: DecisionRecord, time_s: float, outcome: str) -> None:
        record.outcome = outcome
        record.finished_at_s = time_s
        self.active.pop(record.auv_id)
        self.last_results[record.auv_id] = MacroActionResult(
            record.action, outcome, time_s, record.collected_packets, record.collected_bytes
        )
        if isinstance(record.action, VisitAction):
            self.cooldowns[record.auv_id][record.action.ch_id] = max(
                time_s + self.config.revisit_cooldown_s, nextafter(time_s, inf)
            )

    def record_event(self, entry: EventLogEntry) -> None:
        result = entry.result
        if isinstance(result, GenerationResult):
            packet = result.packet
            self.generated.append((packet.source_ch_id, packet.generation_time_s, packet.size_bytes))
            self.dropped += len(result.buffer_result.dropped_packets)
            return
        if entry.entity_id is None or (record := self.active.get(entry.entity_id)) is None:
            return

        now_s = entry.time_s
        if isinstance(result, CHArrival):
            record.arrived_at_s = now_s
            record.arrival_report_time_s = result.report.report_time_s
            if result.outcome != "collection_started":
                self.complete(record, now_s, result.outcome)
        elif isinstance(result, CollectionResult):
            if result.completed:
                record.collected_packets += 1
                record.collected_bytes += result.packet.size_bytes
            else:
                self.complete(record, now_s, "mission_cutoff")
        elif entry.event_type is EventType.AUV_ARRIVAL_SINK:
            record.arrived_at_s = now_s
            if not self.sim.auvs[record.auv_id].cargo:
                self.complete(record, now_s, "returned_empty")
        elif entry.event_type is EventType.UPLOAD_START and isinstance(result, UploadJob):
            record.upload_started_at_s = now_s
            self.jobs.append(result)
        elif isinstance(result, DeliveryRecord):
            record.uploaded_packets += 1
        elif entry.event_type is EventType.UPLOAD_COMPLETE and isinstance(result, UploadJob):
            record.upload_finished_at_s = now_s
            self.complete(record, now_s, "uploaded")
        elif result == "decision_ready" and isinstance(record.action, VisitAction):
            self.complete(record, now_s, "collected")

    def defer(
        self, auv_id: int, reason: str, wake_at_s: float | None, rejected_action: Action | None = None
    ) -> None:
        now_s = self.sim.current_time_s
        deadline = self.sim.mission_duration_s
        if wake_at_s is not None:
            wake_at_s = max(wake_at_s, nextafter(now_s, inf))
            if wake_at_s >= deadline:
                wake_at_s = None
        self.inactive.append(NoActionRecord(auv_id, now_s, reason, wake_at_s, rejected_action))
        # At the deadline the runner ignores readiness. Scheduling clears the current flag.
        self.sim.schedule_event(deadline if wake_at_s is None else wake_at_s, EventType.WAIT_COMPLETE, auv_id)

    def decide(self, auv_id: int) -> None:
        observation = build_local_observation(
            self.sim, auv_id, last_action_result=self.last_results.get(auv_id), cooldowns=self.cooldowns[auv_id]
        )
        options = evaluate_actions(observation, self.config)
        if not options.actions:
            self.defer(auv_id, options.reason, options.wake_at_s)
            return
        action = self.policies[auv_id].choose_action(replace(observation, legal_actions=options.actions))
        if not isinstance(action, (VisitAction, ReturnAction)):
            raise TypeError(f"AUV {auv_id} policy must return VisitAction or ReturnAction")
        if action not in options.actions:
            self.defer(auv_id, "illegal_action", None, action)
            return

        report_time_s = None
        if isinstance(action, VisitAction):
            ch = next(ch for ch in observation.chs if ch.ch_id == action.ch_id)
            if ch.report is not None:
                report_time_s = ch.report.report_time_s
            travel = self.sim.visit(auv_id, action.ch_id)
        else:
            travel = self.sim.return_to_sink(auv_id)
        record = DecisionRecord(auv_id, observation.current_time_s, action, options.reason, report_time_s, travel)
        self.active[auv_id] = record
        self.decisions.append(record)

    def run(self) -> EpisodeResult:
        sim = self.sim
        initial_packets = sum(len(ch.buffer) for ch in sim.cluster_heads.values())
        initial_packets += sum(len(auv.cargo) for auv in sim.auvs.values())
        for auv_id in sorted(sim.auvs):
            sim.schedule_event(0.0, EventType.AUV_DECISION_READY, auv_id)

        while (entry := sim.step_event()) is not None:
            self.record_event(entry)
            if entry.result != "decision_ready" or entry.time_s >= sim.mission_duration_s:
                continue
            # Process only events at the current time before breaking decision ties by ID.
            for simultaneous in sim.run_until(entry.time_s):
                self.record_event(simultaneous)
            for auv_id in sorted(sim.decision_ready):
                self.decide(auv_id)

        for record in tuple(self.active.values()):
            self.complete(record, sim.mission_duration_s, "mission_cutoff")
        metrics = calculate_metrics(
            sim, self.decisions, self.jobs,
            generated_packets=len(self.generated), dropped_packets=self.dropped, initial_packets=initial_packets,
        )
        assert sim.sink is not None
        return EpisodeResult(
            metrics, tuple(self.decisions), tuple(self.inactive), tuple(self.generated), sim.sink.delivered_records
        )


def run_episode(simulator: Simulator, policy_factory: PolicyFactory, config: ReturnRuleConfig) -> EpisodeResult:
    """Reset the scenario, then run fresh per-AUV policies through the mission end."""
    simulator.reset()
    return _EpisodeRun(simulator, policy_factory, config).run()


def run_comparison(
    scenario_factory: ScenarioFactory, seeds: Iterable[int], config: ReturnRuleConfig
) -> tuple[BaselineResult, ...]:
    """Run the three policies on separate copies of each seeded scenario."""
    results = []
    for seed in seeds:
        simulator = scenario_factory(seed)
        reference_trace = None
        for name, factory in BASELINE_POLICIES:
            episode = run_episode(simulator, factory, config)
            if reference_trace is None:
                reference_trace = episode.generation_trace
            elif episode.generation_trace != reference_trace:
                raise RuntimeError(f"policies did not share a CH generation trace for seed {seed}")
            results.append(BaselineResult(name, seed, episode))
    return tuple(results)
