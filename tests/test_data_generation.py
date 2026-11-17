"""Deterministic, statistical and entity-integration tests for CH generation."""

from math import sqrt
from random import Random
from statistics import mean, variance
from unittest.mock import patch

import pytest

from uwsn.data_generation import (
    GenerationEvent,
    GenerationManager,
    PacketParameters,
    PoissonProcess,
    RateSchedule,
    RateSegment,
)
from uwsn.entities import ClusterHead, DataPacket, OverflowPolicy


def sample_times(schedule, seed=42, end_time_s=100.0, start_time_s=0.0):
    process = PoissonProcess(schedule, end_time_s, seed, start_time_s)
    times = []
    while (time_s := process.next_time_s()) is not None:
        times.append(time_s)
    assert process.next_time_s() is None
    return times


def make_ch(ch_id=0, capacity=10, policy=OverflowPolicy.DROP_OLDEST):
    return ClusterHead(ch_id, (0.0, 0.0, -10.0), capacity, 20.0, overflow_policy=policy)


def run_manager(manager):
    packets = []
    while (event := manager.peek_next_event()) is not None:
        assert manager.peek_next_event() == event
        result = manager.process_event(event)
        assert result.packet.generation_time_s == event.time_s
        assert result.packet.source_ch_id == event.ch_id
        packets.append(result.packet)
    return packets


def test_schedule_rates_boundaries_gaps_and_integrated_intensity():
    schedule = RateSchedule((RateSegment(2, 4, 2), RateSegment(6, 10, 3)))

    assert [schedule.rate_at(t) for t in (0, 2, 3, 4, 5, 6, 9, 10)] == [0, 2, 2, 0, 0, 3, 3, 0]
    assert schedule.integrated_intensity(0, 12) == 16
    assert schedule.integrated_intensity(3, 7) == 5
    assert schedule.integrated_intensity(7, 7) == 0
    assert RateSchedule.fixed(0, 10).integrated_intensity(0, 10) == 0


@pytest.mark.parametrize("start,end,rate", [
    (-1, 2, 1), (2, 2, 1), (3, 2, 1), (0, 2, -1),
    (0, float("inf"), 1), (0, 2, float("nan")), (False, 2, 1), (0, 2, "1"),
])
def test_invalid_segments_raise_value_error(start, end, rate):
    with pytest.raises(ValueError):
        RateSegment(start, end, rate)


@pytest.mark.parametrize("segments", [
    (), ("invalid",), None,
    (RateSegment(0, 5, 1), RateSegment(4, 6, 1)),
    (RateSegment(5, 6, 1), RateSegment(0, 4, 1)),
])
def test_invalid_schedule_order_overlap_and_contents(segments):
    with pytest.raises(ValueError):
        RateSchedule(segments)


def test_schedule_defensively_copies_input_segments():
    segments = [RateSegment(0, 10, 1)]
    schedule = RateSchedule(segments)
    segments.clear()

    assert schedule.rate_at(5) == 1


def test_fixed_rate_times_increase_and_match_exponential_intervals():
    seed = 34
    times = sample_times(RateSchedule.fixed(2.0, 100.0), seed)
    rng = Random(seed)
    expected_time = 0.0
    for actual_time in times:
        expected_time += rng.expovariate(1.0) / 2.0
        assert actual_time == expected_time
    assert len(times) > 100
    assert all(left < right for left, right in zip([0.0] + times, times))


def test_same_seed_produces_identical_sequence_and_different_seed_changes_it():
    schedule = RateSchedule((RateSegment(0, 20, 1), RateSegment(30, 100, 2)))
    times = sample_times(schedule, 7)

    assert times == sample_times(schedule, 7)
    assert times != sample_times(schedule, 8)


def test_residual_intensity_crosses_rate_changes_and_zero_segments_without_redraw():
    schedule = RateSchedule((
        RateSegment(0, 1, 1), RateSegment(1, 3, 0), RateSegment(5, 10, 2),
    ))
    process = PoissonProcess(schedule, 10, seed=1)
    # One intensity budget spans all three segments.
    with patch.object(process._rng, "expovariate", return_value=2.5) as draw:
        assert process.next_time_s() == 5.75  # 5 + (2.5 - 1) / 2
        draw.assert_called_once_with(1.0)


def test_splitting_same_rate_into_many_segments_preserves_event_sequence():
    fixed = RateSchedule.fixed(0.5, 100)
    split = RateSchedule(tuple(RateSegment(i, i + 1, 0.5) for i in range(100)))

    assert sample_times(split, 77) == pytest.approx(sample_times(fixed, 77), abs=1e-11)


def test_fixed_rate_long_run_counts_match_poisson_mean_and_variance():
    schedule = RateSchedule.fixed(2.0, 100.0)
    counts = [len(sample_times(schedule, seed)) for seed in range(80)]
    expected_per_run = 200.0
    expected_total = len(counts) * expected_per_run

    assert abs(sum(counts) - expected_total) < 6 * sqrt(expected_total)
    assert 0.55 * expected_per_run < variance(counts) < 1.6 * expected_per_run
    print(f"Fixed-rate statistics: runs=80, mean={mean(counts):.3f}, variance={variance(counts):.3f}, expected=200")


def test_piecewise_rates_have_expected_counts_and_trend():
    schedule = RateSchedule((RateSegment(0, 4000, 0.5), RateSegment(4000, 8000, 2)))
    times = sample_times(schedule, seed=2026, end_time_s=8000)
    early = sum(t < 4000 for t in times)
    late = len(times) - early

    assert abs(early - 2000) < 6 * sqrt(2000)
    assert abs(late - 8000) < 6 * sqrt(8000)
    assert late > early
    print(f"Piecewise statistics: early={early} (expected 2000), late={late} (expected 8000)")


def test_zero_rates_and_gaps_produce_no_events():
    schedule = RateSchedule((
        RateSegment(0, 10, 0), RateSegment(10, 30, 2),
        RateSegment(30, 50, 0), RateSegment(70, 100, 2),
    ))
    times = sample_times(schedule)

    assert times
    assert all(10 <= t < 30 or 70 <= t < 100 for t in times)
    assert sample_times(RateSchedule.fixed(0, 100)) == []


def test_mission_window_clips_schedule_at_both_ends():
    times = sample_times(RateSchedule.fixed(2, 100), end_time_s=30, start_time_s=20)

    assert times
    assert all(20 < t < 30 for t in times)
    assert sample_times(RateSchedule.fixed(1, 10), start_time_s=20) == []


def test_exact_intensity_boundary_uses_next_positive_segment():
    schedule = RateSchedule((RateSegment(0, 1, 1), RateSegment(1, 2, 0), RateSegment(2, 3, 1)))
    process = PoissonProcess(schedule, 3, seed=1)
    # An identical budget of one is drawn for the first arrival and again for the cutoff.
    with patch.object(process._rng, "expovariate", return_value=1.0):
        assert process.next_time_s() == 2.0
        assert process.next_time_s() is None  # Exact mission end is excluded.


def test_roundoff_cannot_repeat_a_generation_time():
    process = PoissonProcess(RateSchedule.fixed(1e20, 2), 2, seed=1, start_time_s=1)

    first = process.next_time_s()
    second = process.next_time_s()
    assert first is not None and second is not None
    assert 1 < first < second < 2


@pytest.mark.parametrize("kwargs", [
    {"end_time_s": 0}, {"start_time_s": 100}, {"seed": -1}, {"seed": True},
    {"end_time_s": float("inf")}, {"schedule": None},
])
def test_invalid_process_parameters(kwargs):
    parameters = {"schedule": RateSchedule.fixed(1, 100), "end_time_s": 100, "seed": 1}
    with pytest.raises(ValueError):
        PoissonProcess(**(parameters | kwargs))


def test_ch_streams_are_independent_of_registration_order_and_other_seeds():
    def collect(order, other_seed):
        manager = GenerationManager(100)
        for ch_id in order:
            seed = 55 if ch_id == 0 else other_seed
            manager.register_ch(make_ch(ch_id), RateSchedule.fixed(2, 100), PacketParameters(4, 3, 0.1), seed=seed)
        manager.initialize()
        return [p.generation_time_s for p in run_manager(manager) if p.source_ch_id == 0]

    assert collect([0], 1) == collect([0, 1], 2) == collect([1, 0], 99)
    assert collect([0], 1) == sample_times(RateSchedule.fixed(2, 100), seed=55)


def test_manager_packet_ids_timestamps_parameters_and_ch_buffers():
    manager = GenerationManager(20, first_packet_id=100)
    heads = [make_ch(ch_id, capacity=100_000) for ch_id in range(3)]
    for ch in heads:
        manager.register_ch(
            ch, RateSchedule.fixed(ch.ch_id + 1, 20),
            PacketParameters(ch.ch_id + 4, ch.ch_id + 10, 0.1), seed=ch.ch_id + 10,
        )
    manager.initialize()
    packets = run_manager(manager)

    assert [p.packet_id for p in packets] == list(range(100, 100 + len(packets)))
    assert all(a.generation_time_s <= b.generation_time_s for a, b in zip(packets, packets[1:]))
    for ch in heads:
        expected = tuple(p for p in packets if p.source_ch_id == ch.ch_id)
        assert ch.buffer == expected
        assert all(p.size_bytes == ch.ch_id + 4 and p.base_value == ch.ch_id + 10 for p in expected)
        stats = manager.get_stats(ch.ch_id)
        assert stats.generated_packets == stats.inserted_packets == len(expected)
        assert stats.generated_bytes == ch.total_generated_bytes == sum(p.size_bytes for p in expected)
        assert stats.buffer_size_bytes == ch.buffer_size_bytes
        assert stats.overflow_dropped_packets == stats.oversized_rejected_packets == 0


@pytest.mark.parametrize("policy", list(OverflowPolicy))
def test_manager_overflow_statistics_count_evictions_and_rejections(policy):
    manager = GenerationManager(20)
    ch = make_ch(capacity=10, policy=policy)
    manager.register_ch(ch, RateSchedule.fixed(2, 20), PacketParameters(4, 1, 0), seed=5)
    manager.initialize()
    packets = run_manager(manager)
    stats = manager.get_stats(0)

    assert len(packets) > 2
    assert stats.generated_packets == len(packets) == ch.total_generated_packets
    assert stats.generated_bytes == 4 * len(packets)
    assert stats.buffer_size_bytes == 8
    assert stats.overflow_dropped_packets == len(packets) - 2
    assert stats.oversized_rejected_packets == 0
    if policy is OverflowPolicy.DROP_OLDEST:
        assert stats.inserted_packets == len(packets)
        assert ch.buffer == tuple(packets[-2:])
    else:
        assert stats.inserted_packets == 2
        assert ch.buffer == tuple(packets[:2])


def test_oversized_rejections_preserve_existing_buffer_and_use_counter_differences():
    ch = make_ch(capacity=10)
    existing = DataPacket(50, 0, 0, 6, 1, 0)
    ch.generate_packet(existing)
    manager = GenerationManager(20)
    manager.register_ch(ch, RateSchedule.fixed(1, 20), PacketParameters(11, 1, 0), seed=3)
    manager.initialize()
    packets = run_manager(manager)
    stats = manager.get_stats(0)

    assert packets[0].packet_id == 51
    assert stats.generated_packets == stats.oversized_rejected_packets == len(packets)
    assert ch.total_generated_packets == len(packets) + 1
    assert stats.generated_bytes == len(packets) * 11
    assert stats.inserted_packets == stats.overflow_dropped_packets == 0
    assert ch.buffer == (existing,)
    assert stats.buffer_size_bytes == 6


def test_manager_statistics_follow_live_buffer_after_collection():
    manager = GenerationManager(10)
    ch = make_ch(capacity=1000)
    manager.register_ch(ch, RateSchedule.fixed(1, 10), PacketParameters(4, 1, 0), seed=3)
    manager.initialize()
    packets = run_manager(manager)
    before = manager.get_stats(0)
    ch.remove_packets([packets[0].packet_id])
    after = manager.get_stats(0)

    assert after.buffer_size_bytes == before.buffer_size_bytes - 4
    assert after.generated_packets == before.generated_packets
    assert after.inserted_packets == before.inserted_packets


def test_manager_equal_times_are_ordered_by_ch_id_and_queries_are_idempotent():
    manager = GenerationManager(10)
    for ch_id in (1, 0):
        manager.register_ch(make_ch(ch_id), RateSchedule.fixed(1, 10), PacketParameters(4, 1, 0), seed=1)
    manager.initialize()
    first = manager.peek_next_event()
    assert first is not None and first.ch_id == 0
    manager.process_event(first)
    second = manager.peek_next_event()
    assert second is not None and second.ch_id == 1 and second.time_s == first.time_s


def test_invalid_and_stale_events_do_not_mutate_manager_or_ch():
    manager = GenerationManager(10)
    ch = make_ch()
    manager.register_ch(ch, RateSchedule.fixed(1, 10), PacketParameters(4, 1, 0), seed=2)
    manager.initialize()
    event = manager.peek_next_event()
    assert event is not None
    with pytest.raises(ValueError, match="next pending"):
        manager.process_event(GenerationEvent(0, event.time_s + 1))
    assert manager.peek_next_event() == event
    assert ch.total_generated_packets == 0
    manager.process_event(event)
    with pytest.raises(ValueError, match="next pending"):
        manager.process_event(event)
    assert ch.total_generated_packets == 1


def test_manager_lifecycle_and_duplicate_registration_errors():
    manager = GenerationManager(10)
    ch = make_ch()
    schedule = RateSchedule.fixed(1, 10)
    parameters = PacketParameters(4, 1, 0)
    with pytest.raises(RuntimeError, match="initialize"):
        manager.peek_next_event()
    with pytest.raises(RuntimeError, match="initialize"):
        manager.get_stats(0)
    manager.register_ch(ch, schedule, parameters, seed=1)
    with pytest.raises(ValueError, match="already registered"):
        manager.register_ch(ch, schedule, parameters, seed=2)
    manager.initialize()
    with pytest.raises(RuntimeError, match="already initialized"):
        manager.initialize()
    with pytest.raises(RuntimeError, match="before initializing"):
        manager.register_ch(make_ch(1), schedule, parameters, seed=2)
    with pytest.raises(KeyError, match="not registered"):
        manager.get_stats(99)


def test_duplicate_existing_ids_are_rejected_before_initialization():
    manager = GenerationManager(10)
    for ch_id in (0, 1):
        ch = make_ch(ch_id)
        ch.generate_packet(DataPacket(5, ch_id, 0, 4, 1, 0))
        manager.register_ch(ch, RateSchedule.fixed(1, 10), PacketParameters(4, 1, 0), seed=ch_id)
    with pytest.raises(ValueError, match="duplicate global packet IDs"):
        manager.initialize()


def test_zero_rate_and_empty_managers_exhaust_without_generating_packets():
    empty = GenerationManager(10)
    empty.initialize()
    assert empty.peek_next_event() is None
    zero = GenerationManager(10)
    ch = make_ch()
    zero.register_ch(ch, RateSchedule.fixed(0, 10), PacketParameters(4, 1, 0), seed=1)
    zero.initialize()
    assert run_manager(zero) == []
    assert zero.get_stats(0).generated_packets == 0


@pytest.mark.parametrize("size,value,decay", [(0, 1, 0), (4, -1, 0), (4, 1, -1), (4, float("nan"), 0)])
def test_packet_parameters_validate_before_generation(size, value, decay):
    with pytest.raises(ValueError):
        PacketParameters(size, value, decay)


def test_manager_validates_mission_and_id_configuration():
    with pytest.raises(ValueError, match="mission_deadline_s"):
        GenerationManager(0)
    with pytest.raises(ValueError, match="first_packet_id"):
        GenerationManager(10, first_packet_id=-1)
