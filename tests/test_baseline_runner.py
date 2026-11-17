"""Asynchronous episode, deadline, accounting, and paired-benchmark regressions."""

from dataclasses import replace
from math import fsum

import pytest

from uwsn.baselines import CyclicPatrol, NearestNeighbor
from uwsn.data_generation import GenerationManager, PacketParameters, RateSchedule
from uwsn.entities import AUV, ClusterHead, DataPacket, Sink
from uwsn.environment import EventType, Simulator, make_scenario
from uwsn.environment.action_rules import ReturnRuleConfig
from uwsn.environment.actions import ReturnAction, VisitAction
from uwsn.environment.simulator import MAX_DECISIONS_PER_INSTANT
from uwsn.evaluation import run_comparison, run_episode, summarize_metrics


RULE = ReturnRuleConfig(0.75, 3, 5)


def test_empty_local_visit_remains_legal_later_and_never_repeats_at_same_time():
    sim = Simulator(
        16, cluster_heads=[ClusterHead(0, (0, 0, 0), 10, 1)], auvs=[AUV(0, (0, 0, 0), 1, 10)],
        sink=Sink((0, 0, 0), 1), collection_rate_bytes_per_s=1,
    )
    result = run_episode(sim, lambda _: NearestNeighbor(), RULE)
    assert [decision.decision_time_s for decision in result.decisions] == [0, 5, 10]
    assert all(decision.action == VisitAction(0) for decision in result.decisions)
    assert all(decision.outcome == "nothing_to_collect" for decision in result.decisions)
    assert [entry.wake_at_s for entry in result.inactive_decisions] == [5, 10, None]
    assert sim.current_time_s == 16 and result.metrics.empty_ch_visits == 3


def test_lock_conflicts_cool_down_and_allow_other_auv_to_finish_collection():
    horizon = 60
    head = ClusterHead(0, (0, 0, 0), 10, 1)
    head.generate_packet(DataPacket(0, 0, 0, 10, 1, 0))
    manager = GenerationManager(horizon)
    manager.register_ch(head, RateSchedule.fixed(0, horizon), PacketParameters(10, 1, 0), seed=7)
    sim = Simulator(
        horizon, generation_manager=manager, auvs=[AUV(i, (0, 0, 0), 1, 10) for i in (0, 1)],
        sink=Sink((0, 0, 0), 5), collection_rate_bytes_per_s=1,
    )
    result = run_episode(sim, lambda _: NearestNeighbor(), ReturnRuleConfig(0.75, 1, 2))
    conflicts = [d for d in result.decisions if d.outcome == "service_conflict"]
    assert [d.decision_time_s for d in conflicts] == [0, 2, 4, 6, 8]
    assert all(d.auv_id == 1 and d.arrival_report_time_s == d.decision_time_s for d in conflicts)
    assert result.metrics.rejected_ch_visits == len(conflicts)
    winner = result.decisions[0]
    assert winner.auv_id == 0 and winner.collected_bytes == 10 and winner.finished_at_s == 10
    assert result.metrics.delivered_packets == 1


def test_no_feasible_action_parks_until_mission_end_without_calling_policy():
    class MustNotChoose:
        def choose_action(self, observation):
            pytest.fail("a policy must not run with no feasible action")

    ch = ClusterHead(0, (300, 0, 0), 10, 1)
    sim = Simulator(20, cluster_heads=[ch], auvs=[AUV(0, (0, 0, 0), 1, 10)], sink=Sink((0, 0, 0), 1))
    result = run_episode(sim, lambda _: MustNotChoose(), RULE)
    assert result.decisions == ()
    parked, = result.inactive_decisions
    assert parked.reason == "no_feasible_action" and parked.wake_at_s is None
    assert sim.current_time_s == 20 and result.metrics.average_delivery_delay_s is None


def test_illegal_action_is_recorded_without_dispatch_or_retry():
    class AlwaysReturn:
        def choose_action(self, observation):
            return ReturnAction()

    ch = ClusterHead(0, (1, 0, 0), 10, 1)
    sim = Simulator(20, cluster_heads=[ch], auvs=[AUV(0, (0, 0, 0), 1, 10)], sink=Sink((0, 0, 0), 1))
    result = run_episode(sim, lambda _: AlwaysReturn(), RULE)
    assert result.decisions == ()
    rejected, = result.inactive_decisions
    assert rejected.reason == "illegal_action" and rejected.rejected_action == ReturnAction()
    assert sim.current_time_s == 20


def test_policies_use_local_observations_at_independent_async_times():
    chs = [ClusterHead(0, (0, 0, 0), 10, 1), ClusterHead(1, (40, 0, 0), 10, 1)]
    chs[0].generate_packet(DataPacket(0, 0, 0, 1, 1, 0))
    chs[1].generate_packet(DataPacket(1, 1, 0, 1, 1, 0))
    sim = Simulator(
        120, cluster_heads=chs, auvs=[AUV(i, (0, 0, 0), 1, 1) for i in (0, 1)],
        sink=Sink((0, 0, 0), 1), collection_rate_bytes_per_s=1,
    )
    observations = []

    class RecordingPatrol(CyclicPatrol):
        def choose_action(self, observation):
            observations.append(observation)
            return super().choose_action(observation)

    episode = run_episode(sim, RecordingPatrol, RULE)
    assert [(r.auv_id, r.decision_time_s, r.action) for r in episode.decisions[:3]] == [
        (0, 0, VisitAction(0)), (1, 0, VisitAction(1)), (0, 1, ReturnAction()),
    ]
    assert observations[2].last_action_result.outcome == "collected"
    assert observations[2].last_action_result.collected_bytes == 1
    assert all(obs.decision_ready for obs in observations)
    for auv_id in (0, 1):
        decisions = [decision for decision in episode.decisions if decision.auv_id == auv_id]
        assert all(first.finished_at_s <= later.decision_time_s for first, later in zip(decisions, decisions[1:]))


def test_same_ch_regenerates_and_supplies_delivered_packets_on_multiple_rounds():
    ch = ClusterHead(0, (3, 0, 0), 10, 1)
    manager = GenerationManager(100)
    manager.register_ch(ch, RateSchedule.fixed(2, 100), PacketParameters(1, 1, 0.01), seed=5)
    sim = Simulator(
        100, generation_manager=manager, auvs=[AUV(0, (0, 0, 0), 1, 4)],
        sink=Sink((0, 0, 0), 4), collection_rate_bytes_per_s=4, record_log=False,
    )
    episode = run_episode(sim, CyclicPatrol, RULE)
    visits = [decision for decision in episode.decisions if decision.outcome == "collected"]
    first, second = visits[:2]
    assert first.action == second.action == VisitAction(0)
    assert first.finished_at_s < second.decision_time_s
    assert second.report_time_s == first.arrival_report_time_s
    assert any(record.generation_time_s < first.finished_at_s for record in episode.deliveries)
    assert any(record.generation_time_s > first.finished_at_s for record in episode.deliveries)
    assert len(visits) >= 5 and episode.metrics.delivered_packets > first.collected_packets
    assert sim.event_log == []


def test_cutoff_metrics_use_actual_packet_ledger_and_include_partial_channel_occupancy():
    first = AUV(0, (0, 0, 0), 1, 6)
    first.load_packets(tuple(DataPacket(index, 0, 0, 3, 1, 0.1) for index in (1, 2)))
    second = AUV(1, (0, 0, 0), 1, 1)
    second.load_packets((DataPacket(3, 0, 0, 1, 1, 0.1),))
    sim = Simulator(5, auvs=[first, second], sink=Sink((0, 0, 0), 1))
    result = run_episode(sim, lambda _: NearestNeighbor(), RULE)
    metrics = result.metrics
    assert result.deliveries == sim.sink.delivered_records
    assert metrics.total_delivered_voi == fsum(record.delivered_voi for record in result.deliveries)
    assert metrics.delivered_packets == 1 and metrics.delivered_bytes == 3
    assert metrics.undelivered_packets == metrics.carried_packets == 2
    assert metrics.average_delivery_delay_s == 3 and metrics.average_sink_wait_s == 0
    assert metrics.sink_busy_time_s == 5 and metrics.sink_observed_utilization == 1
    assert metrics.sink_unstarted_upload_requests == 1
    assert all(record.outcome == "mission_cutoff" for record in result.decisions)
    assert result.decisions[0].uploaded_packets == 1 and result.decisions[0].upload_finished_at_s is None


def test_paired_seeds_match_ch_generation_and_reproduce_all_policy_results():
    def scenario(seed):
        return make_scenario("S3", seed, horizon_s=180, record_log=False)

    results = run_comparison(scenario, (41, 42), RULE)
    assert results == run_comparison(scenario, (41, 42), RULE)
    traces = {
        seed: {result.episode.generation_trace for result in results if result.seed == seed}
        for seed in (41, 42)
    }
    assert all(len(same_seed) == 1 for same_seed in traces.values())
    assert traces[41] != traces[42]
    assert all(len(packet) == 3 for trace in traces[41] for packet in trace)


def test_sample_metrics_statistics_cover_missing_delays_and_per_ch_counts():
    sim = Simulator(10, sink=Sink((0, 0, 0), 1))
    empty = run_episode(sim, CyclicPatrol, RULE).metrics
    samples = (
        replace(empty, total_delivered_voi=voi, average_sink_wait_s=wait, per_ch_visit_counts=((0, count),))
        for voi, wait, count in ((1, None, 0), (3, 1.0, 2), (5, 3.0, 4))
    )
    summaries = summarize_metrics(samples)
    assert summaries["total_delivered_voi"].mean == 3
    assert summaries["total_delivered_voi"].std == 2
    assert summaries["ch_0_visits"].mean == 2
    assert summaries["average_sink_wait_s"].sample_count == 2
    assert summaries["average_sink_wait_s"].std == pytest.approx(2**0.5)
    assert summaries["average_delivery_delay_s"].mean is None
    assert summaries["average_delivery_delay_s"].sample_count == 0
    assert summarize_metrics([empty])["total_delivered_voi"].std == 0


def test_simulator_rejects_direct_same_time_return_loop_outside_the_baseline_runner():
    sim = Simulator(20, auvs=[AUV(0, (0, 0, 0), 1, 10)], sink=Sink((0, 0, 0), 1))
    sim.schedule_event(0, EventType.AUV_DECISION_READY, 0)
    sim.run_until(0)
    for _ in range(MAX_DECISIONS_PER_INSTANT):
        sim.return_to_sink(0)
        sim.run_until(0)
    with pytest.raises(RuntimeError, match=r"AUV 0 .*t=0 s without time advancing"):
        sim.return_to_sink(0)
    assert sim.decision_ready == {0}
    sim.run_until(1)
    sim.return_to_sink(0)
    sim.run_until(1)
    assert sim.decision_ready == {0}
