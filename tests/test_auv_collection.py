"""Integration tests for arrival, report access, and packet conservation."""

from dataclasses import fields
from math import inf, nextafter
from random import Random
from unittest.mock import patch

import pytest

from uwsn.data_generation import GenerationManager, GenerationResult, PacketParameters, RateSchedule
from uwsn.entities import AUV, AUVStatus, ClusterHead, DataPacket, OverflowPolicy, Sink
from uwsn.environment import CHArrival, CollectionResult, EventType, Simulator


def head(ch_id=0, position=(0, 0, 0), capacity=100, policy=OverflowPolicy.DROP_OLDEST):
    return ClusterHead(ch_id, position, capacity, 1, overflow_policy=policy)


def add_packet(ch, packet_id, size_bytes):
    packet = DataPacket(packet_id, ch.ch_id, 0, size_bytes, 1.0, 0.01)
    assert ch.generate_packet(packet).inserted
    return packet


def initialize_auvs(sim):
    for auv_id in sim.auvs:
        sim.schedule_event(0, EventType.AUV_DECISION_READY, auv_id)
    sim.run_until(0)


def assert_packet_conservation(sim, initial_packets):
    records = [entry.result for entry in sim.event_log if isinstance(entry.result, GenerationResult)]
    supplied = [*initial_packets, *(record.packet for record in records)]
    dropped = [packet for record in records for packet in record.buffer_result.dropped_packets]
    reserved = [ch.reserved_packet for ch in sim.cluster_heads.values() if ch.reserved_packet is not None]
    reserved_ids = {packet.packet_id for packet in reserved}
    buffered = [packet for ch in sim.cluster_heads.values() for packet in ch.buffer
                if packet.packet_id not in reserved_ids]
    carried = [packet for auv in sim.auvs.values() for packet in auv.cargo]
    all_accounted = buffered + reserved + carried + dropped
    assert sorted(p.packet_id for p in all_accounted) == sorted(p.packet_id for p in supplied)
    assert len({p.packet_id for p in supplied}) == len(supplied)
    assert sum(p.size_bytes for p in all_accounted) == sum(p.size_bytes for p in supplied)


def test_arrival_gates_local_report_and_generation_continues_during_travel_and_collection():
    ch = head(position=(10, 0, 0))
    manager = GenerationManager(20)
    manager.register_ch(ch, RateSchedule.fixed(1, 20), PacketParameters(4, 1.0, 0.01), seed=1)
    # CH 0 alone draws intensity gaps of 2, 3, 4, 1, 2 and a post-cutoff gap.
    with patch.object(Random, "expovariate", side_effect=[2, 3, 4, 1, 2, 100]):
        sim = Simulator(20, generation_manager=manager, auvs=[AUV(0, (0, 0, 0), 1, 100)],
                        collection_rate_bytes_per_s=4, include_generation_counter=True)
        sim.schedule_event(3, EventType.AUV_DECISION_READY, 0)
        initialize_auvs(sim)
        travel = sim.visit(0, 0)
        assert travel.distance_m == travel.travel_time_s == 9
        sim.run_until(8)
        assert sim.auvs[0].position == (0, 0, 0) and sim.auvs[0].status is AUVStatus.NAVIGATING
        assert sim.cluster_heads[0].total_generated_packets == 2
        assert sim.get_auv_ch_report(0, 0) is None
        assert sim.decision_ready == set()
        with pytest.raises(RuntimeError, match="not decision-ready"):
            sim.visit(0, 0)
        with pytest.raises(RuntimeError, match="busy AUV"):
            sim.schedule_event(8, EventType.AUV_DECISION_READY, 0)

        sim.run_until(9)
        assert sim.auvs[0].position == (9, 0, 0)
        report = sim.get_auv_ch_report(0, 0)
        assert report is not None
        assert {field.name for field in fields(report)} == {
            "ch_id", "report_time_s", "buffer_size_bytes", "packet_timestamps", "total_generated_packets",
        }
        assert report.report_time_s == 9 and report.total_generated_packets == 3
        assert report.packet_timestamps == (2.0, 5.0, 9.0) and report.buffer_size_bytes == 12
        assert sim.auvs[0].cargo == () and sim.cluster_heads[0].reserved_packet is not None
        assert_packet_conservation(sim, ())

        entries = sim.run_until(10)
        assert [e.event_type for e in entries] == [EventType.CH_DATA_GENERATION, EventType.COLLECTION_COMPLETE]
        assert len(sim.auvs[0].cargo) == 1 and sim.cluster_heads[0].buffer_size_bytes == 12
        sim.run_until(14)
        assert tuple(p.generation_time_s for p in sim.auvs[0].cargo) == (2, 5, 9, 10, 12)
        assert sim.cluster_heads[0].buffer == () and sim.decision_ready == {0}
        assert sim.get_auv_ch_report(0, 0) == report
        assert_packet_conservation(sim, ())


def test_collects_whole_packets_in_fifo_order_and_stops_at_first_packet_that_cannot_fit():
    ch = head()
    packets = [add_packet(ch, index, size) for index, size in enumerate((3, 5, 2))]
    sim = Simulator(20, cluster_heads=[ch], auvs=[AUV(0, (0, 0, 0), 1, 7)], collection_rate_bytes_per_s=2)
    initialize_auvs(sim)
    assert sim.visit(0, 0).travel_time_s == 0
    sim.run_until(1.49)
    assert sim.auvs[0].cargo == ()
    assert sim.cluster_heads[0].buffer == tuple(packets)
    assert sim.cluster_heads[0].reserved_packet == packets[0]
    sim.run_until(1.5)
    assert sim.auvs[0].cargo == (packets[0],)
    assert sim.cluster_heads[0].buffer == tuple(packets[1:])
    assert sim.cluster_heads[0].service_lock_owner is None and sim.decision_ready == {0}
    assert_packet_conservation(sim, packets)


def test_empty_or_full_arrival_releases_the_lock_without_transferring():
    empty, stocked = head(0), head(1)
    packet = add_packet(stocked, 1, 4)
    sim = Simulator(20, cluster_heads=[empty, stocked], auvs=[AUV(0, (0, 0, 0), 1, 0)])
    initialize_auvs(sim)
    for target in (0, 1):
        sim.visit(0, target)
        arrived, ready = sim.run_until(0)
        assert isinstance(arrived.result, CHArrival) and arrived.result.outcome == "nothing_to_collect"
        assert ready.event_type is EventType.AUV_DECISION_READY
        assert sim.cluster_heads[target].service_lock_owner is None
    assert_packet_conservation(sim, (packet,))


def test_same_time_arrivals_use_enqueue_order_and_other_ch_completes_independently():
    first, second = head(0, (5, 0, 0)), head(1, (0, 7, 0))
    long, short = add_packet(first, 0, 4), add_packet(second, 1, 1)
    auvs = [AUV(index, (0, 0, 0), 1, 10) for index in range(3)]
    sim = Simulator(20, cluster_heads=[first, second], auvs=auvs, collection_rate_bytes_per_s=1)
    initialize_auvs(sim)
    sim.visit(1, 0)
    sim.visit(0, 0)
    sim.visit(2, 1)

    arrivals = [entry for entry in sim.run_until(4) if isinstance(entry.result, CHArrival)]
    assert [entry.entity_id for entry in arrivals] == [1, 0]
    assert [entry.result.outcome for entry in arrivals] == ["collection_started", "service_conflict"]
    assert sim.cluster_heads[0].service_lock_owner == 1
    assert sim.cluster_heads[0].buffer == (long,) and sim.auvs[0].cargo == ()
    assert sim.decision_ready == {0} and sim.auvs[2].status is AUVStatus.NAVIGATING
    sim.run_until(6)
    assert sim.cluster_heads[0].service_lock_owner == 1 and sim.cluster_heads[1].service_lock_owner == 2
    assert_packet_conservation(sim, (long, short))
    sim.run_until(7)
    assert sim.auvs[2].cargo == (short,) and sim.decision_ready == {0, 2}
    assert sim.auvs[1].status is AUVStatus.COLLECTING
    sim.run_until(8)
    assert sim.auvs[1].cargo == (long,) and sim.decision_ready == {0, 1, 2}
    assert all(ch.service_lock_owner is None for ch in sim.cluster_heads.values())
    assert_packet_conservation(sim, (long, short))


def test_lease_roundoff_does_not_allow_another_auv_to_steal_an_in_flight_packet():
    deadline = nextafter(1.0, inf)
    start = 2 ** -53
    ch = head()
    packet = add_packet(ch, 0, 1)
    auvs = [AUV(0, (0, 0, 0), 1, 10), AUV(1, (2, 0, 0), 1, 10)]
    sim = Simulator(deadline, cluster_heads=[ch], auvs=auvs, collection_rate_bytes_per_s=1)
    initialize_auvs(sim)
    sim.visit(1, 0)
    sim.run_until(start)
    sim.visit(0, 0)
    sim.run()

    arrivals = [entry.result for entry in sim.event_log if isinstance(entry.result, CHArrival)]
    assert [arrival.outcome for arrival in arrivals] == ["collection_started", "service_conflict"]
    assert sim.auvs[0].cargo == (packet,) and sim.auvs[1].cargo == ()


def test_starting_a_macro_action_clears_only_that_auvs_readiness():
    auvs = [AUV(auv_id, (0, 0, 0), 1, 10) for auv_id in (7, 1)]
    sim = Simulator(20, cluster_heads=[head(position=(5, 0, 0))], auvs=auvs)
    initialize_auvs(sim)
    assert sim.decision_ready == {1, 7}
    first, second = sorted(sim.decision_ready)
    sim.visit(first, 0)
    assert sim.decision_ready == {second}
    assert sim.auvs[first].status is AUVStatus.NAVIGATING
    assert sim.auvs[second].status is AUVStatus.IDLE


@pytest.mark.parametrize("deadline", [3, 4])
@pytest.mark.parametrize("policy,new_size,expected_ids", [
    (OverflowPolicy.DROP_OLDEST, 4, (10, 13)),
    (OverflowPolicy.DROP_OLDEST, 8, (10, 11)),
    (OverflowPolicy.DROP_NEWEST, 4, (10, 11)),
])
def test_cutoff_cancels_reserved_packet_without_losing_it_to_overflow(deadline, policy, new_size, expected_ids):
    ch = head(capacity=8, policy=policy)
    initial = [add_packet(ch, index, 4) for index in (10, 11)]
    manager = GenerationManager(deadline)
    manager.register_ch(ch, RateSchedule.fixed(1, deadline), PacketParameters(new_size, 1, 0), seed=1)
    # Poisson CH 0 draws two one-second gaps, then one that skips the deadline.
    with patch.object(Random, "expovariate", side_effect=[1, 1, 100]):
        sim = Simulator(deadline, generation_manager=manager, auvs=[AUV(0, (0, 0, 0), 1, 12)],
                        collection_rate_bytes_per_s=1)
        initialize_auvs(sim)
        sim.visit(0, 0)
        sim.run_until(2)
        live_ch = sim.cluster_heads[0]
        assert live_ch.reserved_packet == initial[0] and live_ch.buffer_size_bytes == 8
        assert sim.auvs[0].cargo == ()
        assert_packet_conservation(sim, initial)
        sim.run()

    assert tuple(packet.packet_id for packet in live_ch.buffer) == expected_ids
    assert live_ch.reserved_packet is None and live_ch.service_lock_owner is None
    assert sim.auvs[0].cargo == () and sim.decision_ready == set()
    result = next(entry for entry in sim.event_log if entry.event_type is EventType.COLLECTION_COMPLETE)
    assert result.time_s == deadline and result.result == CollectionResult(initial[0], completed=False)
    assert_packet_conservation(sim, initial)


def test_completed_packet_stays_in_cargo_when_the_next_packet_is_interrupted():
    ch = head()
    first, second = add_packet(ch, 10, 3), add_packet(ch, 11, 5)
    sim = Simulator(8, cluster_heads=[ch], auvs=[AUV(0, (0, 0, 0), 1, 10)], collection_rate_bytes_per_s=1)
    initialize_auvs(sim)
    sim.visit(0, 0)
    sim.run()
    assert sim.auvs[0].cargo == (first,) and sim.cluster_heads[0].buffer == (second,)
    assert_packet_conservation(sim, (first, second))


def test_return_updates_position_and_leaves_cargo_and_sink_ledger_intact():
    auv = AUV(0, (0, 0, 0), 2, 10)
    packet = DataPacket(50, 0, 0, 4, 1, 0.01)
    auv.load_packets((packet,))
    sink = Sink((3, 4, -12), 20)
    sim = Simulator(10, auvs=[auv], sink=sink)
    initialize_auvs(sim)
    leg = sim.return_to_sink(0)
    assert leg.distance_m == 13 and leg.travel_time_s == 6.5
    sim.run_until(6)
    assert sim.auvs[0].position == (0, 0, 0) and sim.auvs[0].status is AUVStatus.RETURNING
    arrived = sim.step_event()
    assert arrived is not None
    assert arrived.time_s == 6.5 and arrived.event_type is EventType.AUV_ARRIVAL_SINK
    assert sim.auvs[0].position == sink.position and sim.auvs[0].status is AUVStatus.WAITING_SINK
    assert sim.auvs[0].cargo == (packet,) and sim.decision_ready == set()
    assert sim.sink is not None and sim.sink.active_upload is None and sim.sink.delivered_records == ()
    assert len(sim.sink.waiting_queue) == 1 and sim.sink.waiting_queue[0].packets == (packet,)
    assert sim.sink.total_voi == 0


def test_arrival_at_cutoff_can_record_report_but_cannot_start_a_transfer():
    ch = head(position=(11, 0, 0))
    packet = add_packet(ch, 0, 1)
    sim = Simulator(10, cluster_heads=[ch], auvs=[AUV(0, (0, 0, 0), 1, 10)])
    initialize_auvs(sim)
    sim.visit(0, 0)
    arrived, ended = sim.run()
    assert isinstance(arrived.result, CHArrival) and arrived.result.outcome == "deadline"
    assert ended.event_type is EventType.MISSION_END and sim.decision_ready == set()
    assert sim.cluster_heads[0].service_lock_owner is None and sim.cluster_heads[0].buffer == (packet,)


def test_generation_uses_ids_above_initial_cargo_and_unregistered_ch_buffers():
    generated_head, preloaded_head = head(0), head(1)
    add_packet(preloaded_head, 100, 1)
    auv = AUV(0, (0, 0, 0), 1, 10)
    auv.load_packets((DataPacket(120, 1, 0, 1, 1, 0),))
    manager = GenerationManager(5)
    manager.register_ch(generated_head, RateSchedule.fixed(1, 5), PacketParameters(1, 1, 0), seed=1)
    # Only registered CH 0 draws an arrival at t=1 and a post-cutoff gap.
    with patch.object(Random, "expovariate", side_effect=[1, 10]):
        sim = Simulator(5, generation_manager=manager, cluster_heads=[generated_head, preloaded_head], auvs=[auv])
        sim.run()
    assert sim.cluster_heads[0].buffer[0].packet_id == 121
