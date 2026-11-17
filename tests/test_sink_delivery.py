"""Exercise FCFS service, cargo ownership, and VoI through simulator events."""

from math import exp

import pytest

from uwsn.data_generation import GenerationManager, GenerationResult, PacketParameters, RateSchedule
from uwsn.entities import AUV, AUVStatus, ClusterHead, DataPacket, Sink
from uwsn.environment import EventType, Simulator


def loaded_auv(auv_id, distance_m, packet_sizes):
    auv = AUV(auv_id, (distance_m, 0, 0), 1, sum(packet_sizes))
    auv.load_packets(
        DataPacket(100 * auv_id + index, 7, 0, size, 10, 0.1)
        for index, size in enumerate(packet_sizes)
    )
    return auv


def ready_at_zero(sim):
    for auv_id in sim.auvs:
        sim.schedule_event(0, EventType.AUV_DECISION_READY, auv_id)
    sim.run_until(0)


def timeline(entries, event_type):
    return [(entry.entity_id, entry.time_s) for entry in entries if entry.event_type is event_type]


def test_single_return_delivers_cargo_in_order_and_only_finishes_after_last_packet():
    auv = loaded_auv(0, 3, [100, 300, 200])
    sim = Simulator(12, auvs=[auv], sink=Sink((0, 0, 0), 100))
    ready_at_zero(sim)
    sim.return_to_sink(0)
    sim.run_until(3.5)
    assert timeline(sim.event_log, EventType.UPLOAD_START) == [(0, 3)]
    assert sim.sink.active_upload.request.arrival_time_s == 3
    assert sim.sink.waiting_queue == ()
    assert sim.auvs[0].status is AUVStatus.UPLOADING
    assert sim.auvs[0].cargo == auv.cargo
    assert sim.sink.delivered_records == () and not sim.decision_ready

    sim.run_until(4)
    record, = sim.sink.delivered_records
    assert (record.packet_id, record.source_ch_id, record.auv_id, record.generation_time_s,
            record.delivery_time_s, record.size_bytes) == (0, 7, 0, 0, 4, 100)
    assert sim.auvs[0].cargo == auv.cargo[1:]
    assert sim.auvs[0].storage_used_bytes == 500 and not sim.decision_ready

    sim.run_until(7)
    assert sim.auvs[0].cargo == auv.cargo[2:] and not sim.decision_ready
    sim.run_until(9)
    assert timeline(sim.event_log, EventType.PACKET_UPLOAD_COMPLETE) == [(0, 4), (0, 7), (0, 9)]
    assert timeline(sim.event_log, EventType.UPLOAD_COMPLETE) == [(0, 9)]
    assert sim.auvs[0].cargo == () and sim.sink.active_upload is None
    assert sim.auvs[0].status is AUVStatus.IDLE and sim.decision_ready == {0}
    assert sim.sink.delivered_packet_ids == {0, 1, 2}
    with pytest.raises(ValueError, match="already been delivered"):
        sim.sink.enqueue_upload(0, auv.cargo[:1], 9)


def test_fcfs_uses_arrival_then_event_order_and_hands_off_at_same_time():
    auvs = [loaded_auv(0, 4, [300]), loaded_auv(1, 1, [300]), loaded_auv(2, 1, [300])]
    sim = Simulator(11, auvs=auvs, sink=Sink((0, 0, 0), 100))
    ready_at_zero(sim)
    for auv_id in (0, 2, 1):
        sim.return_to_sink(auv_id)
    sim.run_until(1)
    assert sim.sink.active_upload.request.auv_id == 2
    assert [request.auv_id for request in sim.sink.waiting_queue] == [1]
    assert [auv.status for auv in sim.auvs.values()] == [
        AUVStatus.RETURNING, AUVStatus.WAITING_SINK, AUVStatus.UPLOADING,
    ]
    with pytest.raises(RuntimeError, match="busy AUV"):
        sim.schedule_event(2, EventType.AUV_DECISION_READY, 1)

    sim.run_until(4)
    assert sim.sink.active_upload.request.auv_id == 1
    assert [request.auv_id for request in sim.sink.waiting_queue] == [0]
    assert sim.decision_ready == {2}
    sim.run()
    assert timeline(sim.event_log, EventType.AUV_ARRIVAL_SINK) == [(2, 1), (1, 1), (0, 4)]
    assert timeline(sim.event_log, EventType.UPLOAD_START) == [(2, 1), (1, 4), (0, 7)]
    assert timeline(sim.event_log, EventType.UPLOAD_COMPLETE) == [(2, 4), (1, 7), (0, 10)]
    assert [record.auv_id for record in sim.sink.delivered_records] == [2, 1, 0]


def test_sink_allows_only_one_pending_request_per_auv_even_with_distinct_packets():
    sim_auv = loaded_auv(0, 0, [10, 10])
    first, second = sim_auv.cargo
    sink = Sink((0, 0, 0), 10)
    sink.enqueue_upload(0, (first,), 0)
    with pytest.raises(ValueError, match="already has a queued or active upload"):
        sink.enqueue_upload(0, (second,), 0)
    sink.start_next_upload(0)
    sink.record_packet_delivery(first.packet_id, 1, 10)
    with pytest.raises(ValueError, match="already has a queued or active upload"):
        sink.enqueue_upload(0, (second,), 1)
    sink.finish_active_upload(1)
    assert sink.enqueue_upload(0, (second,), 1).packets == (second,)


def test_empty_return_keeps_channel_free_and_auv_ready():
    auv = loaded_auv(0, 2, [])
    sim = Simulator(5, auvs=[auv], sink=Sink((0, 0, 0), 100))
    ready_at_zero(sim)
    sim.return_to_sink(0)
    arrival, ready = sim.run_until(2)
    assert arrival.event_type is EventType.AUV_ARRIVAL_SINK
    assert ready.event_type is EventType.AUV_DECISION_READY
    assert sim.auvs[0].position == sim.sink.position and sim.decision_ready == {0}
    assert sim.sink.waiting_queue == () and sim.sink.active_upload is None
    assert sim.sink.delivered_records == ()


@pytest.mark.parametrize("volume_weight", [0, 0.3, 1])
def test_voi_at_delivery_uses_configured_weight_and_actual_packet_completion(volume_weight):
    auv = loaded_auv(0, 1, [100, 100])
    sim = Simulator(5, auvs=[auv], sink=Sink((0, 0, 0), 100, volume_weight=volume_weight))
    ready_at_zero(sim)
    sim.return_to_sink(0)
    sim.run_until(1)
    assert sim.sink.total_voi == 0
    sim.run()
    first, second = sim.sink.delivered_records
    assert first.delivered_voi == pytest.approx(10 * (volume_weight + (1 - volume_weight) * exp(-0.2)))
    assert second.delivered_voi == pytest.approx(10 * (volume_weight + (1 - volume_weight) * exp(-0.3)))
    assert second.delivered_voi <= first.delivered_voi
    assert sim.sink.total_voi == pytest.approx(first.delivered_voi + second.delivered_voi)


def test_deadline_includes_exact_packet_and_retains_active_and_queued_cargo():
    vehicles = [
        loaded_auv(0, 0, [100, 200, 400]), loaded_auv(1, 1, [100]), loaded_auv(2, 3, [100]),
    ]
    sim = Simulator(3, auvs=vehicles, sink=Sink((0, 0, 0), 100))
    ready_at_zero(sim)
    for auv_id in sim.auvs:
        sim.return_to_sink(auv_id)
    sim.run_until(2)
    assert sim.sink.delivered_packet_ids == {0}
    assert sim.auvs[0].cargo == vehicles[0].cargo[1:]
    ended = sim.run()
    assert [(entry.time_s, entry.event_type) for entry in ended] == [
        (3, EventType.AUV_ARRIVAL_SINK), (3, EventType.PACKET_UPLOAD_COMPLETE), (3, EventType.MISSION_END),
    ]
    assert sim.sink.delivered_packet_ids == {0, 1}
    assert sim.sink.delivered_records[-1].delivery_time_s == 3
    assert sim.sink.delivered_records[-1].delivered_voi > 0
    assert timeline(sim.event_log, EventType.UPLOAD_START) == [(0, 0)]
    assert not timeline(sim.event_log, EventType.UPLOAD_COMPLETE)
    assert [(item.auv_id, item.arrival_time_s, item.start_time_s) for item in sim.sink.interrupted_uploads] == [
        (0, 0, 0), (1, 1, None), (2, 3, None),
    ]
    assert [item.packets for item in sim.sink.interrupted_uploads] == [
        vehicles[0].cargo[2:], vehicles[1].cargo, vehicles[2].cargo,
    ]
    assert [auv.cargo for auv in sim.auvs.values()] == [
        vehicles[0].cargo[2:], vehicles[1].cargo, vehicles[2].cargo,
    ]
    assert all(auv.status is AUVStatus.WAITING_SINK for auv in sim.auvs.values())
    assert sim.sink.active_upload is None and sim.sink.waiting_queue == ()
    assert sim.event_queue.is_empty() and sim.step_event() is None and not sim.decision_ready


def test_exact_deadline_job_finishes_without_starting_the_next_request():
    sim = Simulator(3, auvs=[loaded_auv(0, 0, [300]), loaded_auv(1, 1, [100])],
                    sink=Sink((0, 0, 0), 100))
    ready_at_zero(sim)
    sim.return_to_sink(0)
    sim.return_to_sink(1)
    sim.run()
    assert [entry.event_type for entry in sim.event_log if entry.time_s == 3] == [
        EventType.PACKET_UPLOAD_COMPLETE, EventType.UPLOAD_COMPLETE, EventType.MISSION_END,
    ]
    assert sim.auvs[0].status is AUVStatus.IDLE and sim.auvs[0].cargo == ()
    assert not sim.decision_ready
    interrupted, = sim.sink.interrupted_uploads
    assert interrupted.auv_id == 1 and interrupted.start_time_s is None


def seeded_collection_and_delivery(seed, cuts=()):
    manager = GenerationManager(12)
    ch = ClusterHead(7, (0, 0, 0), 6, 1)
    manager.register_ch(ch, RateSchedule.fixed(4, 12), PacketParameters(2, 1, 0.1), seed=seed)
    started = loaded_auv(0, 1, [4, 4])
    collector = AUV(1, (0, 0, 0), 1, 4)
    sim = Simulator(
        12, generation_manager=manager, auvs=[started, collector],
        sink=Sink((0, 0, 0), 2), collection_rate_bytes_per_s=2,
    )
    ready_at_zero(sim)
    sim.return_to_sink(0)
    sim.run_until(2)
    sim.visit(1, 7)
    sim.run_until(4.5)
    assert sim.auvs[0].status is AUVStatus.UPLOADING
    assert sim.auvs[1].cargo and sim.decision_ready == {1}
    assert sim.get_auv_ch_report(0, 7) is None
    assert sim.get_auv_ch_report(1, 7).report_time_s == 2
    sim.return_to_sink(1)
    for cut in cuts:
        sim.run_until(cut)
    sim.run()
    return sim, started.cargo


def test_generation_collection_and_delivery_conserve_packets_and_bytes():
    sim, initial = seeded_collection_and_delivery(81)
    generated = [entry.result for entry in sim.event_log if isinstance(entry.result, GenerationResult)]
    supplied = [*initial, *(item.packet for item in generated)]
    buffered = [packet for ch in sim.cluster_heads.values() for packet in ch.buffer]
    carried = [packet for auv in sim.auvs.values() for packet in auv.cargo]
    delivered = [item.packet for item in sim.sink.delivered_records]
    dropped = [packet for item in generated for packet in item.buffer_result.dropped_packets]
    accounted = buffered + carried + delivered + dropped
    assert dropped and buffered
    assert {item.auv_id for item in sim.sink.delivered_records} == {0, 1}
    assert any(1 < item.packet.generation_time_s < 5 for item in generated)
    assert sorted(supplied, key=lambda p: p.packet_id) == sorted(accounted, key=lambda p: p.packet_id)
    assert len({p.packet_id for p in accounted}) == len(accounted)
    assert sum(p.size_bytes for p in supplied) == sum(p.size_bytes for p in accounted)


def test_seeded_delivery_logs_are_reproducible_and_split_invariant():
    direct, _ = seeded_collection_and_delivery(81)
    split, _ = seeded_collection_and_delivery(81, cuts=(4.5, 5, 5.5, 6, 8, 12))
    replay, _ = seeded_collection_and_delivery(81)
    another, _ = seeded_collection_and_delivery(82)
    assert split.event_log == direct.event_log == replay.event_log
    assert split.sink.delivered_records == direct.sink.delivered_records == replay.sink.delivered_records
    assert split.sink.total_voi == direct.sink.total_voi
    assert another.sink.delivered_records != direct.sink.delivered_records
