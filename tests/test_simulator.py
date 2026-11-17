"""Mixed generation, time alignment, and asynchronous AUV tests."""

from random import Random
from unittest.mock import patch

import pytest

from uwsn.data_generation import (
    GenerationManager,
    GenerationMode,
    GenerationResult,
    MMPPConfig,
    MMPPStats,
    PacketParameters,
    RateSchedule,
    RateSegment,
)
from uwsn.entities import AUV, ClusterHead, DataPacket, OverflowPolicy, Sink
from uwsn.environment import EventType, Simulator


PARAMETERS = PacketParameters(10, 1.0, 0.0)
CH_EVENT = EventType.CH_DATA_GENERATION
WAIT = EventType.WAIT_COMPLETE
READY = EventType.AUV_DECISION_READY
END = EventType.MISSION_END


def head(ch_id: int) -> ClusterHead:
    return ClusterHead(ch_id, (0, 0, -10), 400, 30, overflow_policy=OverflowPolicy.DROP_OLDEST)


def vehicles() -> list[AUV]:
    return [AUV(index, (0, 0, 0), 1.0, 500) for index in range(3)]


def mixed_simulator(order=(0, 1, 2), other_seed=13) -> Simulator:
    manager = GenerationManager(120, first_packet_id=50)
    schedule = RateSchedule((RateSegment(0, 30, 0), RateSegment(30, 120, 0.6)))
    for ch_id in order:
        if ch_id == 0:
            manager.register_ch(head(ch_id), RateSchedule.fixed(0.7, 120), PARAMETERS, seed=44)
        elif ch_id == 1:
            manager.register_ch(head(ch_id), schedule, PARAMETERS, seed=other_seed)
        else:
            config = MMPPConfig(0.1, 1.2, 0.2, 0.4, GenerationMode.STATIONARY, random_seed=61)
            manager.register_mmpp_ch(head(ch_id), config, PARAMETERS)
    sim = Simulator(120, generation_manager=manager, auvs=vehicles(), sink=Sink((0, 0, 0), 1.0))
    for index, time_s in enumerate((100, 120, 70)):
        sim.schedule_event(time_s, WAIT, index)
    return sim


def packets(sim: Simulator, ch_id: int | None = None) -> list[DataPacket]:
    return [
        entry.result.packet for entry in sim.event_log
        if isinstance(entry.result, GenerationResult) and (ch_id is None or entry.entity_id == ch_id)
    ]


def test_mixed_generation_updates_collocated_entities_and_global_packet_ids():
    sim = mixed_simulator()
    assert sim.current_time_s == 0
    assert all(ch.total_generated_packets == 0 for ch in sim.cluster_heads.values())
    assert sim.generation_manager.get_mode_stats(2) == MMPPStats()
    sim.run()

    all_packets = packets(sim)
    assert [packet.packet_id for packet in all_packets] == list(range(50, 50 + len(all_packets)))
    assert [entry.time_s for entry in sim.event_log] == sorted(entry.time_s for entry in sim.event_log)
    assert all(packet.generation_time_s < 120 for packet in all_packets)
    assert all(packet.generation_time_s >= 30 for packet in packets(sim, 1))
    for ch_id, ch in sim.cluster_heads.items():
        generated = packets(sim, ch_id)
        assert generated and ch.total_generated_packets == len(generated)
        assert ch.buffer == tuple(generated[-40:])
        assert ch.buffer_size_bytes == min(len(generated), 40) * 10
    mode = sim.generation_manager.get_mode_stats(2)
    assert mode.packets_generated_in_normal + mode.packets_generated_in_burst == len(packets(sim, 2))
    assert mode.normal_mode_duration_s + mode.burst_mode_duration_s == pytest.approx(120)
    assert sim.sink is not None and not sim.sink.delivered_records


def test_split_runs_reset_and_one_shot_have_identical_results():
    one_shot, split = mixed_simulator(), mixed_simulator()
    expected_log = one_shot.run()
    records = split.run_until(0) + split.run_until(50) + split.run_until(70) + split.run_until(120)
    assert records == expected_log
    assert split.generation_manager.get_mode_stats(2) == one_shot.generation_manager.get_mode_stats(2)
    assert split.cluster_heads == one_shot.cluster_heads
    assert split.decision_ready == one_shot.decision_ready == {0, 1, 2}
    assert split.run() == () and split.step_event() is None

    split.reset()
    assert split.current_time_s == 0 and split.event_log == [] and split.decision_ready == set()
    assert all(not ch.buffer for ch in split.cluster_heads.values())
    for index, time_s in enumerate((100, 120, 70)):
        split.schedule_event(time_s, WAIT, index)
    assert split.run() == expected_log


def test_ch_streams_are_independent_of_registration_order_and_other_ch_seed():
    baseline, reversed_order = mixed_simulator(), mixed_simulator((2, 1, 0))
    baseline.run()
    assert reversed_order.run() == tuple(baseline.event_log)

    changed = mixed_simulator((2, 1, 0), other_seed=997)
    changed.run()
    for ch_id in (0, 2):
        assert [p.generation_time_s for p in packets(changed, ch_id)] == [
            p.generation_time_s for p in packets(baseline, ch_id)
        ]
    assert changed.generation_manager.get_mode_stats(2) == baseline.generation_manager.get_mode_stats(2)
    assert [p.generation_time_s for p in packets(changed, 1)] != [p.generation_time_s for p in packets(baseline, 1)]


def test_ch_report_is_independent_of_global_packet_id_gaps():
    baseline, changed = mixed_simulator(), mixed_simulator(other_seed=997)
    baseline.run_until(100)
    changed.run_until(100)

    left_ids = tuple(p.packet_id for p in baseline.cluster_heads[0].buffer)
    right_ids = tuple(p.packet_id for p in changed.cluster_heads[0].buffer)
    assert left_ids != right_ids
    report = baseline.create_ch_report(0)
    assert report == changed.create_ch_report(0)
    assert report.packet_timestamps == tuple(p.generation_time_s for p in baseline.cluster_heads[0].buffer)
    assert report.total_generated_packets is None


def test_ch_report_tracks_clock_and_rejects_backdating_after_dropped_packet():
    manager = GenerationManager(100)
    ch = ClusterHead(0, (0, 0, -10), 10, 30, overflow_policy=OverflowPolicy.DROP_NEWEST)
    manager.register_ch(ch, RateSchedule.fixed(1, 100), PARAMETERS, seed=10)
    # The only Poisson CH draws one intensity gap per arrival and one at the cutoff.
    with patch.object(Random, "expovariate", side_effect=[25, 25, 25, 25]):
        sim = Simulator(100, generation_manager=manager)
        assert sim.create_ch_report(0).packet_timestamps == ()
        sim.run_until(50)
        snapshot = sim.create_ch_report(0, include_generation_counter=True)
        sim.run_until(100)
        final = sim.create_ch_report(0, include_generation_counter=True)

    assert (snapshot.report_time_s, snapshot.total_generated_packets) == (50, 2)
    assert (final.report_time_s, final.total_generated_packets) == (100, 3)
    assert snapshot.packet_timestamps == final.packet_timestamps == (25,)
    with pytest.raises(ValueError, match="latest packet generation"):
        sim.cluster_heads[0].create_report(50, include_generation_counter=True)
    with pytest.raises(ValueError, match="backwards"):
        sim.run_until(50)


def test_start_zero_rate_gap_same_time_order_and_exact_deadline():
    manager = GenerationManager(10)
    manager.register_ch(head(2), RateSchedule.fixed(1, 10), PARAMETERS, seed=1)
    schedule = RateSchedule((RateSegment(0, 1, 1), RateSegment(1, 2, 0), RateSegment(2, 10, 1)))
    manager.register_ch(head(1), schedule, PARAMETERS, seed=2)
    manager.register_mmpp_ch(head(3), MMPPConfig(1, 0, 0, 0, random_seed=0), PARAMETERS)

    # The segmented CH consumes one unit of intensity before the zero-rate gap.
    # First arrivals tie at 2, and the next arrivals hit the deadline exactly.
    # Initialization draws for CHs 2, 1, 3 in registration order. At t=2 it draws
    # for CHs 1, 2, 3 in event order. MMPP CH 3 draws only its arrival clock.
    with patch.object(Random, "expovariate", side_effect=[2, 1, 2, 8, 8, 8]):
        extra_auv = AUV(3, (0, 0, 0), 1.0, 500)
        sim = Simulator(10, generation_manager=manager, auvs=[*vehicles(), extra_auv])
        sim.schedule_event(0, READY, 0)
        sim.schedule_event(2, WAIT, 1)
        sim.schedule_event(10, WAIT, 2)
        sim.schedule_event(11, WAIT, 3)

        assert [(entry.time_s, entry.event_type) for entry in sim.run_until(0)] == [(0, READY)]
        assert sim.decision_ready == {0}
        tied = sim.run_until(2)
        assert [(entry.time_s, entry.event_type, entry.entity_id) for entry in tied] == [
            (2, CH_EVENT, 1), (2, CH_EVENT, 2), (2, CH_EVENT, 3), (2, WAIT, 1),
        ]
        assert [(entry.time_s, entry.event_type) for entry in sim.run_until(25)] == [(10, WAIT), (10, END)]

    assert sim.current_time_s == 10 and sim.event_queue.is_empty()
    assert len(packets(sim)) == 3
    assert sim.step_event() is None


def test_mmpp_silent_switches_follow_clock_even_with_no_pending_packet():
    manager = GenerationManager(10)
    manager.register_mmpp_ch(head(0), MMPPConfig(0, 0, 1, 2, random_seed=0), PARAMETERS)
    # There are switches at 2 and 5, with a would-be switch exactly at the deadline.
    # MMPP draws only the switch clock in each mode because both arrival rates are zero.
    with patch.object(Random, "expovariate", side_effect=[2, 3, 5]) as draw:
        sim = Simulator(10, generation_manager=manager, auvs=vehicles())
        sim.schedule_event(5, WAIT, 0)
        diagnostics = sim.generation_manager.get_mode_stats
        assert diagnostics(0) == MMPPStats(accounted_until_s=0)
        assert sim.run_until(1) == ()
        assert diagnostics(0) == MMPPStats(1, 0, 0, 0, 0, 0, 1)
        sim.run_until(2)
        assert diagnostics(0) == MMPPStats(2, 0, 1, 0, 0, 0, 2)
        sim.run_until(4)
        assert diagnostics(0) == MMPPStats(2, 2, 1, 0, 0, 0, 4)
        assert sim.step_event().time_s == 5
        assert diagnostics(0) == MMPPStats(2, 3, 1, 1, 0, 0, 5)
        sim.run_until(8)
        assert diagnostics(0) == MMPPStats(5, 3, 1, 1, 0, 0, 8)
        sim.run()
        assert diagnostics(0) == MMPPStats(7, 3, 1, 1, 0, 0, 10)
        assert sim.run() == () and diagnostics(0).normal_mode_duration_s == 7
        assert draw.call_count == 3
    assert not packets(sim)


def test_mmpp_pending_packet_and_final_tail_are_counted_only_at_the_correct_time():
    manager = GenerationManager(10)
    manager.register_mmpp_ch(head(0), MMPPConfig(0, 2, 1, 1, random_seed=0), PARAMETERS)
    # Switch at 1, generate in BURST at 3, switch at 5, then stay in NORMAL.
    # Draws are NORMAL switch; BURST arrival and switch twice; NORMAL switch.
    with patch.object(Random, "expovariate", side_effect=[1, 2, 4, 3, 2, 6]) as draw:
        sim = Simulator(10, generation_manager=manager)
        diagnostics = sim.generation_manager.get_mode_stats
        assert sim.run_until(2) == ()
        assert diagnostics(0) == MMPPStats(1, 1, 1, 0, 0, 0, 2)
        entry = sim.step_event()
        assert entry is not None and entry.time_s == 3
        assert diagnostics(0) == MMPPStats(1, 2, 1, 0, 0, 1, 3)
        assert sim.run_until(6) == ()
        assert diagnostics(0) == MMPPStats(2, 4, 1, 1, 0, 1, 6)
        sim.run()
        assert diagnostics(0) == MMPPStats(6, 4, 1, 1, 0, 1, 10)
        assert draw.call_count == 6
    assert len(packets(sim)) == 1
    report = sim.create_ch_report(0)
    assert not hasattr(report, "mode") and report.total_generated_packets is None


def test_auv_waits_complete_independently_without_clearing_other_plans():
    sim = Simulator(200, auvs=vehicles())
    sim.schedule_event(100, WAIT, 0)
    sim.schedule_event(160, WAIT, 1)
    sim.schedule_event(70, WAIT, 2)
    assert sim.run_until(69) == () and sim.decision_ready == set()

    assert sim.step_event().entity_id == 2
    assert sim.current_time_s == 70 and sim.decision_ready == {2}
    sim.schedule_event(115, WAIT, 2)
    assert sim.decision_ready == set()
    assert sim.step_event().entity_id == 0
    assert sim.current_time_s == 100 and sim.decision_ready == {0}
    assert sim.step_event().entity_id == 2
    assert sim.current_time_s == 115 and sim.decision_ready == {0, 2}
    assert sim.step_event().entity_id == 1
    assert sim.current_time_s == 160 and sim.decision_ready == {0, 1, 2}


def test_invalid_scheduling_cannot_move_clock_or_clear_ready_auv():
    sim = Simulator(10, auvs=vehicles())
    sim.schedule_event(1, READY, 0)
    sim.run_until(2)
    with pytest.raises(ValueError, match="past"):
        sim.schedule_event(1, WAIT, 0)
    with pytest.raises(ValueError, match="backwards"):
        sim.run_until(1)
    with pytest.raises(ValueError, match="unknown AUV"):
        sim.schedule_event(3, WAIT, 5)
    assert sim.current_time_s == 2 and sim.decision_ready == {0}


def test_superseded_wait_is_ignored_and_latest_plan_wins():
    sim = Simulator(100, auvs=vehicles())
    sim.schedule_event(10, WAIT, 0)
    sim.schedule_event(40, WAIT, 0)
    entries = sim.run_until(20)
    assert [(e.time_s, e.result) for e in entries] == [(10, "stale_ignored")]
    assert sim.decision_ready == frozenset()
    with pytest.raises(ValueError, match="past"):
        sim.schedule_event(19, READY, 0)
    sim.run_until(50)
    assert sim.decision_ready == {0}


def test_true_rate_follows_schedule_and_hidden_mode_without_consuming_randomness():
    manager = GenerationManager(10)
    segments = RateSchedule((RateSegment(0, 4, 0.5), RateSegment(6, 10, 2.0)))
    manager.register_ch(head(0), segments, PARAMETERS, seed=1)
    manager.register_mmpp_ch(head(1), MMPPConfig(0, 2, 1, 1, random_seed=2), PARAMETERS)
    # Poisson CH 0 consumes the first intensity budget, which spans the mission.
    # MMPP CH 1 draws NORMAL switch=1, then the two BURST clocks (2, 4), then
    # (3, 2) after the arrival at t=3, and a NORMAL switch=6 after t=5.
    with patch.object(Random, "expovariate", side_effect=[50, 1, 2, 4, 3, 2, 6]) as draw:
        sim = Simulator(10, generation_manager=manager)
        rates = []
        for clock in (0, 1, 2, 4.5, 5, 6, 8):
            sim.run_until(clock)
            before = draw.call_count
            rates.append(tuple(sim.generation_manager.true_rate_at(ch_id) for ch_id in (0, 1)))
            assert sim.generation_manager.true_rate_at(1, clock) == rates[-1][1]
            assert draw.call_count == before
    assert rates == [(0.5, 0), (0.5, 2), (0.5, 2), (0.0, 2), (0.0, 0), (2.0, 0), (2.0, 0)]


def test_disable_event_log_preserves_step_and_run_results():
    def build(record_log):
        sim = Simulator(10, auvs=vehicles(), record_log=record_log)
        sim.schedule_event(1, WAIT, 0)
        sim.schedule_event(2, WAIT, 1)
        return sim

    logged, unlogged = build(True), build(False)
    assert unlogged.step_event() == logged.step_event()
    assert unlogged.run_until(3) == logged.run_until(3)
    assert unlogged.run() == logged.run()
    assert unlogged.event_log == [] and len(logged.event_log) == 3
    assert unlogged.decision_ready == logged.decision_ready == {0, 1}
