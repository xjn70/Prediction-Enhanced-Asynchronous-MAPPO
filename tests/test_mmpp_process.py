"""Continuous-time MMPP sampling, diagnostics, and mixed CH generation tests."""

from dataclasses import asdict, fields, replace
from statistics import mean, variance
from unittest.mock import patch

import pytest

from uwsn.data_generation import (
    GenerationManager,
    GenerationMode,
    MMPPConfig,
    MMPPEvent,
    MMPPProcess,
    MMPPStats,
    PacketParameters,
    RateSchedule,
)
from uwsn.entities import ClusterHead, DataPacket, OverflowPolicy


def drain(process: MMPPProcess) -> list[MMPPEvent]:
    events = []
    while (event := process.next_event()) is not None:
        events.append(event)
    return events


def make_ch(ch_id: int, capacity_bytes: int = 10_000) -> ClusterHead:
    return ClusterHead(ch_id, (0.0, 0.0, -10.0), capacity_bytes, 20.0, 0.0)


def test_seed_reproduces_packet_times_modes_and_diagnostics():
    config = MMPPConfig(0.4, 4.0, 0.12, 0.28, GenerationMode.STATIONARY, 21)
    first, second = MMPPProcess(config, 100), MMPPProcess(config, 100)
    events = drain(first)

    assert events == drain(second)
    assert first.get_stats() == second.get_stats()
    assert events != drain(MMPPProcess(replace(config, random_seed=22), 100))


def test_competing_clocks_keep_mode_on_arrival_and_switch_in_both_directions():
    process = MMPPProcess(MMPPConfig(0.4, 6.0, 0.1, 0.25), 6)
    # NORMAL switches at 2, BURST emits at 2.5, switches at 3.25, NORMAL emits at 3.75.
    waits = [4, 2, 0.5, 2, 4, 0.75, 0.5, 8, 99, 99]
    with patch.object(process._rng, "expovariate", side_effect=waits) as draw:
        assert drain(process) == [
            MMPPEvent(2.5, GenerationMode.BURST), MMPPEvent(3.75, GenerationMode.NORMAL),
        ]
        assert [call.args[0] for call in draw.call_args_list] == [
            0.4, 0.1, 6.0, 0.25, 6.0, 0.25, 0.4, 0.1, 0.4, 0.1,
        ]

    assert process.get_stats() == MMPPStats(4.75, 1.25, 1, 1, 1, 1, 6.0)


def test_peek_caches_waits_and_defers_statistics_including_final_tail():
    process = MMPPProcess(MMPPConfig(0, 2, 1, 1), 10)
    with patch.object(process._rng, "expovariate", side_effect=[2, 1, 6, 10, 10]) as draw:
        initial_stats = process.get_stats()
        event = process.peek_next_event()
        rng_state = process._rng.getstate()
        assert event == MMPPEvent(3, GenerationMode.BURST)
        assert process.peek_next_event() is event
        assert draw.call_count == 3
        assert process._rng.getstate() == rng_state
        assert process.get_stats() == initial_stats
        assert process.current_mode is GenerationMode.NORMAL

        assert process.next_event() is event
        assert draw.call_count == 3
        assert process.get_stats() == MMPPStats(2, 1, 1, 0, 0, 1, 3)
        assert process.peek_next_event() is None
        assert process.peek_next_event() is None
        assert draw.call_count == 5
        assert process.get_stats().accounted_until_s == 3
        assert process.next_event() is None
        assert process.next_time_s() is None
        assert draw.call_count == 5
    assert process.get_stats() == MMPPStats(2, 8, 1, 0, 0, 1, 10)


@pytest.mark.parametrize("mode,q01,q10", [
    (GenerationMode.NORMAL, 0, 3), (GenerationMode.BURST, 3, 0),
    (GenerationMode.NORMAL, 0, 0), (GenerationMode.BURST, 0, 0),
])
def test_explicit_initial_modes_with_zero_outgoing_rate(mode, q01, q10):
    process = MMPPProcess(MMPPConfig(1, 2, q01, q10, mode, 8), 20)
    assert process.current_mode is mode

    events = drain(process)
    stats = process.get_stats()
    assert events and all(event.mode is mode for event in events)
    assert stats.normal_to_burst_transitions == stats.burst_to_normal_transitions == 0
    expected_durations = (20, 0) if mode is GenerationMode.NORMAL else (0, 20)
    assert (stats.normal_mode_duration_s, stats.burst_mode_duration_s) == expected_durations


def test_stationary_initialization_probability_across_independent_seeds():
    config = MMPPConfig(1, 4, 0.12, 0.28, GenerationMode.STATIONARY)
    assert config.stationary_probabilities() == pytest.approx((0.7, 0.3))
    initial_modes = [
        MMPPProcess(replace(config, random_seed=seed), 1).current_mode for seed in range(1000)
    ]
    assert 0.65 < initial_modes.count(GenerationMode.NORMAL) / len(initial_modes) < 0.75
    assert 0.25 < initial_modes.count(GenerationMode.BURST) / len(initial_modes) < 0.35


@pytest.mark.parametrize("q01,q10,mode", [
    (0, 0.2, GenerationMode.NORMAL), (0.2, 0, GenerationMode.BURST),
])
def test_stationary_initialization_selects_the_absorbing_mode(q01, q10, mode):
    config = MMPPConfig(0, 0, q01, q10, GenerationMode.STATIONARY)
    assert MMPPProcess(config, 1).current_mode is mode


def test_stationary_initialization_requires_a_positive_transition_rate():
    with pytest.raises(ValueError, match="specify NORMAL or BURST"):
        MMPPConfig(1, 4, 0, 0, GenerationMode.STATIONARY)


def test_zero_arrival_rates_still_account_for_all_mode_switches():
    process = MMPPProcess(MMPPConfig(0, 0, 2, 3), 8, start_time_s=3)
    with patch.object(process._rng, "expovariate", side_effect=[1, 2, 2]) as draw:
        assert process.next_event() is None
        assert [call.args[0] for call in draw.call_args_list] == [2, 3, 2]
    # The would-be switch at the right endpoint is excluded.
    assert process.get_stats() == MMPPStats(3, 2, 1, 1, 0, 0, 8)


@pytest.mark.parametrize("mode", [GenerationMode.NORMAL, GenerationMode.BURST])
def test_zero_rate_mode_can_switch_to_an_emitting_mode(mode):
    normal_is_silent = mode is GenerationMode.NORMAL
    config = MMPPConfig(0, 2, 1, 0, mode) if normal_is_silent else MMPPConfig(2, 0, 0, 1, mode)
    process = MMPPProcess(config, 10)
    with patch.object(process._rng, "expovariate", side_effect=[1, 0.5, 20]):
        events = drain(process)
    emitting_mode = GenerationMode.BURST if normal_is_silent else GenerationMode.NORMAL
    assert events == [MMPPEvent(1.5, emitting_mode)]


def test_packet_times_are_strictly_increasing_and_exclude_deadline():
    process = MMPPProcess(MMPPConfig(2, 0, 0, 0), end_time_s=10, start_time_s=4)
    with patch.object(process._rng, "expovariate", side_effect=[1, 2, 3]):
        assert process.next_time_s() == 5
        assert process.next_time_s() == 7
        assert process.next_time_s() is None
    assert process.get_stats() == MMPPStats(6, 0, 0, 0, 2, 0, 10)


@pytest.mark.parametrize("name,value", [
    ("lambda_normal_pps", -1), ("lambda_burst_pps", float("inf")),
    ("q_normal_to_burst_per_s", float("nan")), ("q_burst_to_normal_per_s", -1),
    ("initial_mode", "invalid"), ("random_seed", -1),
])
def test_invalid_config_is_rejected(name, value):
    with pytest.raises(ValueError):
        replace(MMPPConfig(1, 2, 0.1, 0.2), **{name: value})


def test_stationary_long_run_occupancy_arrival_rate_and_transition_trend():
    duration_s = 400
    runs = 64
    config = MMPPConfig(0.4, 4.0, 0.12, 0.28, GenerationMode.STATIONARY)
    summaries = []
    for multiplier in (1, 2):
        stats_by_run = []
        for seed in range(runs):
            process = MMPPProcess(replace(
                config,
                q_normal_to_burst_per_s=0.12 * multiplier,
                q_burst_to_normal_per_s=0.28 * multiplier,
                random_seed=1000 * multiplier + seed,
            ), duration_s)
            events = drain(process)
            stats = process.get_stats()
            assert all(a.time_s < b.time_s for a, b in zip(events, events[1:]))
            assert stats.normal_mode_duration_s + stats.burst_mode_duration_s == pytest.approx(duration_s)
            assert stats.packets_generated_in_normal + stats.packets_generated_in_burst == len(events)
            assert abs(stats.normal_to_burst_transitions - stats.burst_to_normal_transitions) <= 1
            stats_by_run.append(stats)

        normal_fraction = mean(s.normal_mode_duration_s / duration_s for s in stats_by_run)
        burst_fraction = mean(s.burst_mode_duration_s / duration_s for s in stats_by_run)
        counts = [s.packets_generated_in_normal + s.packets_generated_in_burst for s in stats_by_run]
        rate = mean(counts) / duration_s
        n_to_b = mean(s.normal_to_burst_transitions for s in stats_by_run)
        b_to_n = mean(s.burst_to_normal_transitions for s in stats_by_run)
        assert normal_fraction == pytest.approx(0.7, abs=0.025)
        assert burst_fraction == pytest.approx(0.3, abs=0.025)
        assert rate == pytest.approx(1.48, abs=0.12)
        assert n_to_b == pytest.approx(33.6 * multiplier, abs=4)
        assert b_to_n == pytest.approx(33.6 * multiplier, abs=4)
        summaries.append(n_to_b + b_to_n)
        print(
            f"q x{multiplier}: normal={normal_fraction:.6f} burst={burst_fraction:.6f} "
            f"rate={rate:.6f} mean_count={mean(counts):.4f} sample_variance={variance(counts):.2f} "
            f"N_to_B={n_to_b:.4f} B_to_N={b_to_n:.4f}"
        )
    assert summaries[1] > 1.7 * summaries[0]


def test_mixed_manager_uses_global_ids_and_keeps_hidden_modes_out_of_entities():
    heads = [make_ch(0), make_ch(1)]
    manager = GenerationManager(40, start_time_s=5, first_packet_id=10)
    parameters = PacketParameters(4, 1, 0)
    manager.register_ch(heads[0], RateSchedule.fixed(1.1, 40, 5), parameters, seed=0)
    manager.register_mmpp_ch(heads[1], MMPPConfig(1, 4, 0.4, 0.3, random_seed=10), parameters)
    manager.initialize()
    assert manager.get_mode_stats(1) == MMPPStats(accounted_until_s=5)
    packets_by_ch = {0: [], 1: []}
    mode_counts = {GenerationMode.NORMAL: 0, GenerationMode.BURST: 0}
    ids = []
    last_time = 5.0
    while (event := manager.peek_next_event()) is not None:
        assert manager.peek_next_event() is event
        assert last_time <= event.time_s < 40
        last_time = event.time_s
        result = manager.process_event(event)
        packet = result.packet
        assert packet.generation_time_s == event.time_s
        assert packet.source_ch_id == event.ch_id
        assert result.buffer_result.inserted
        packets_by_ch[event.ch_id].append(packet)
        ids.append(packet.packet_id)
        if event.ch_id == 0:
            assert event.mode is None
        else:
            assert event.mode in mode_counts
            mode_counts[event.mode] += 1

    assert ids == list(range(10, 10 + len(ids)))
    for ch in heads:
        assert ch.buffer and ch.buffer == tuple(packets_by_ch[ch.ch_id])
        assert manager.get_stats(ch.ch_id).generated_packets == len(ch.buffer)

    stats = manager.get_mode_stats(1)
    assert stats.accounted_until_s == 40
    assert stats.normal_mode_duration_s + stats.burst_mode_duration_s == pytest.approx(35)
    assert stats.packets_generated_in_normal == mode_counts[GenerationMode.NORMAL] > 0
    assert stats.packets_generated_in_burst == mode_counts[GenerationMode.BURST] > 0
    assert {field.name for field in fields(DataPacket)} == {
        "packet_id", "source_ch_id", "generation_time_s", "size_bytes", "base_value", "decay_rate_per_s",
    }
    report = asdict(heads[1].create_report(40))
    assert set(report) == {
        "ch_id", "report_time_s", "buffer_size_bytes", "packet_timestamps", "total_generated_packets",
    }
    assert report["total_generated_packets"] is None
    assert all(set(stamp) == {"packet_id", "generation_time_s"} for stamp in report["packet_timestamps"])
    with pytest.raises(ValueError, match="no MMPP mode statistics"):
        manager.get_mode_stats(0)


def test_mmpp_ch_random_stream_is_independent_of_other_chs_and_registration_order():
    def trace(order, other_seed):
        manager = GenerationManager(50)
        for ch_id in order:
            config = MMPPConfig(0.5, 3, 0.3, 0.2, random_seed=33 if ch_id == 0 else other_seed)
            manager.register_mmpp_ch(make_ch(ch_id), config, PacketParameters(4, 1, 0))
        manager.initialize()
        events = []
        while (event := manager.peek_next_event()) is not None:
            manager.process_event(event)
            if event.ch_id == 0:
                events.append((event.time_s, event.mode))
        return events, manager.get_mode_stats(0)

    assert trace([0], 0) == trace([0, 1], 12) == trace([1, 0], 28)


def test_tied_poisson_and_mmpp_events_use_ch_id_order():
    manager = GenerationManager(10)
    parameters = PacketParameters(4, 1, 0)
    manager.register_mmpp_ch(make_ch(2), MMPPConfig(1, 0, 0, 0, random_seed=18), parameters)
    manager.register_ch(make_ch(1), RateSchedule.fixed(1, 10), parameters, seed=18)
    manager.initialize()

    first = manager.peek_next_event()
    assert first is not None and first.ch_id == 1
    assert manager.process_event(first).packet.packet_id == 0
    second = manager.peek_next_event()
    assert second is not None and second.ch_id == 2 and second.time_s == first.time_s
    assert manager.process_event(second).packet.packet_id == 1


@pytest.mark.parametrize("capacity,oversize", [(0, True), (4, False)])
def test_mode_counts_include_oversized_and_overflow_rejections(capacity, oversize):
    manager = GenerationManager(10)
    ch = ClusterHead(0, (0, 0, -10), capacity, 20, 0, OverflowPolicy.DROP_NEWEST)
    manager.register_mmpp_ch(ch, MMPPConfig(2, 0, 0, 0, random_seed=1), PacketParameters(4, 1, 0))
    manager.initialize()
    while (event := manager.peek_next_event()) is not None:
        manager.process_event(event)
    stats = manager.get_stats(0)
    assert stats.generated_packets > 1
    assert manager.get_mode_stats(0).packets_generated_in_normal == stats.generated_packets
    assert stats.oversized_rejected_packets == (stats.generated_packets if oversize else 0)
    assert stats.overflow_dropped_packets == (0 if oversize else stats.generated_packets - 1)
    assert stats.buffer_size_bytes == capacity
