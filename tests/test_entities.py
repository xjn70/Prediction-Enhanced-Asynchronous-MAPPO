"""Behavior and consistency tests for the core UWSN entities."""

from dataclasses import FrozenInstanceError
from math import exp

import pytest

from uwsn.entities import (
    AUV,
    AUVStatus,
    BufferInsertResult,
    ClusterHead,
    DataPacket,
    OverflowPolicy,
    Sink,
)


def packet(
    packet_id: int, size_bytes: int = 4, generation_time_s: float = 0.0
) -> DataPacket:
    return DataPacket(packet_id, 0, generation_time_s, size_bytes, 10.0, 0.1)


def cluster_head(overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST) -> ClusterHead:
    return ClusterHead(0, (1.0, 2.0, -3.0), 10, 25.0, overflow_policy=overflow_policy)


def test_packet_value_decays_from_generation_until_delivery():
    sample = packet(1, generation_time_s=2.0)

    assert sample.voi_at_delivery(2.0, 8.0) == 10.0
    assert sample.voi_at_delivery(8.0, 8.0) == pytest.approx(10.0 * (0.3 + 0.7 * exp(-0.6)))
    assert sample.voi_at_delivery(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_packet_value_is_zero_after_deadline():
    assert packet(1).voi_at_delivery(10.1, 10.0) == 0.0


@pytest.mark.parametrize("volume_weight,expected", [
    (0.0, 10 * exp(-0.1)),
    (0.3, 10 * (0.3 + 0.7 * exp(-0.1))),
    (1.0, 10.0),
])
def test_sink_delivery_records_apply_configured_voi_weight(volume_weight, expected):
    sink = Sink((0, 0, 0), 4, volume_weight=volume_weight)
    sink.enqueue_upload(1, [packet(1), packet(2)], 0)
    sink.start_next_upload(0)

    on_time = sink.record_packet_delivery(1, 1, 1)
    late = sink.record_packet_delivery(2, 2, 1)
    assert on_time.voi == pytest.approx(expected)
    assert late.voi == 0.0
    assert sink.total_voi == pytest.approx(expected)


@pytest.mark.parametrize("invalid_weight", [-0.1, 1.1, float("nan")])
def test_value_weight_must_be_a_probability(invalid_weight):
    with pytest.raises(ValueError, match="volume_weight"):
        Sink((0, 0, 0), 4, volume_weight=invalid_weight)
    with pytest.raises(ValueError, match="volume_weight"):
        packet(1).voi_at_delivery(1, 1, volume_weight=invalid_weight)


def test_packet_is_immutable_and_rejects_delivery_before_generation():
    sample = packet(1, generation_time_s=4.0)

    with pytest.raises(FrozenInstanceError):
        sample.generation_time_s = 1.0
    with pytest.raises(ValueError, match="generation_time_s"):
        sample.voi_at_delivery(3.0, 2.0)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("packet_id", -1),
        ("source_ch_id", True),
        ("generation_time_s", float("nan")),
        ("size_bytes", 0),
        ("base_value", -1),
        ("decay_rate_per_s", float("inf")),
    ],
)
def test_packet_validates_fields(field, invalid):
    fields = vars(packet(1)) | {field: invalid}

    with pytest.raises(ValueError, match=field):
        DataPacket(**fields)


def test_ch_buffers_and_counts_packets():
    ch = cluster_head()
    first, second = packet(1, 6), packet(2, 4)

    assert ch.generate_packet(first) == BufferInsertResult(True)
    assert ch.generate_packet(second) == BufferInsertResult(True)
    assert ch.buffer == (first, second)
    assert ch.buffer_size_bytes == 10
    assert ch.total_generated_packets == 2
    assert ch.total_generated_bytes == 10


def test_ch_drop_oldest_evicts_only_the_required_fifo_prefix():
    ch = cluster_head()
    first, second, third, fourth = (packet(1, 2), packet(2, 3), packet(3, 4), packet(4, 6))
    for sample in (first, second, third):
        ch.generate_packet(sample)

    assert ch.generate_packet(fourth) == BufferInsertResult(True, (first, second))
    assert ch.buffer == (third, fourth)
    assert ch.buffer_size_bytes == 10
    assert ch.total_generated_packets == 4
    assert ch.total_generated_bytes == 15


def test_ch_drop_newest_counts_rejected_packets():
    ch = cluster_head(OverflowPolicy.DROP_NEWEST)
    first, second = packet(1, 6), packet(2, 5)
    ch.generate_packet(first)

    assert ch.generate_packet(second) == BufferInsertResult(False, (second,))
    assert ch.buffer == (first,)
    assert ch.buffer_size_bytes == 6
    assert ch.total_generated_packets == 2
    assert ch.total_generated_bytes == 11


@pytest.mark.parametrize("policy", list(OverflowPolicy))
def test_ch_oversize_packet_preserves_existing_buffer(policy):
    ch = cluster_head(policy)
    first, oversize = packet(1, 8), packet(2, 11)
    ch.generate_packet(first)

    assert ch.generate_packet(oversize) == BufferInsertResult(False, (oversize,))
    assert ch.buffer == (first,)
    assert ch.buffer_size_bytes == 8
    assert ch.total_generated_packets == 2
    assert ch.total_generated_bytes == 19


def test_ch_peeks_whole_fifo_prefix_and_removes_atomically():
    ch = cluster_head()
    first, second = packet(1, 6), packet(2, 4)
    ch.generate_packet(first)
    ch.generate_packet(second)

    assert ch.peek_packets(4) == ()
    assert ch.peek_packets(9) == (first,)
    assert ch.peek_packets(10) == (first, second)
    with pytest.raises(KeyError, match="99"):
        ch.remove_packets([1, 99])
    assert ch.buffer == (first, second)
    assert ch.remove_packets([1]) == (first,)
    assert ch.buffer_size_bytes == 4
    assert ch.total_generated_packets == 2


def test_ch_rejects_wrong_source_and_duplicate_buffered_id():
    ch = cluster_head()
    sample = packet(1)
    ch.generate_packet(sample)

    with pytest.raises(ValueError, match="source_ch_id"):
        ch.generate_packet(DataPacket(2, 1, 0.0, 4, 1.0, 0.0))
    with pytest.raises(ValueError, match="already buffered"):
        ch.generate_packet(sample)
    assert ch.total_generated_packets == 1


def test_ch_report_snapshots_buffer_and_only_includes_counter_on_request():
    ch = cluster_head()
    ch.generate_packet(packet(1, 7, 2.0))
    ch.generate_packet(packet(2, 3, 3.0))
    ch.generate_packet(packet(3, 11, 4.0))
    report = ch.create_report(5.0)

    assert report.ch_id == ch.ch_id
    assert report.report_time_s == 5.0
    assert report.buffer_size_bytes == 10
    assert report.packet_timestamps == (2.0, 3.0)
    assert report.total_generated_packets is None
    assert not hasattr(report, "generation_rate_pps")
    assert not hasattr(ch, "generation_rate_pps")
    assert ch.create_report(5.0, include_generation_counter=True).total_generated_packets == 3

    ch.remove_packets([1])
    assert report.buffer_size_bytes == 10


def test_ch_service_lock_is_exclusive_until_expiry_and_only_owner_can_release():
    ch = cluster_head()
    assert ch.try_acquire_service_lock(1, 2.0, 4.0)
    assert not ch.try_acquire_service_lock(2, 5.0, 4.0)
    assert not ch.release_service_lock(2)
    assert ch.service_lock_owner == 1
    assert ch.service_lock_expiry_s == 6.0

    assert ch.try_acquire_service_lock(2, 6.0, 4.0)
    assert not ch.release_service_lock(1)
    assert ch.try_acquire_service_lock(2, 7.0, 4.0)
    assert ch.service_lock_expiry_s == 11.0
    assert ch.release_service_lock(2)
    assert ch.service_lock_owner is None
    assert ch.service_lock_expiry_s is None


def test_auv_capacity_and_atomic_batch_loading():
    auv = AUV(0, (0.0, 0.0, 0.0), 1.0, 10)
    first, second = packet(1, 6), packet(2, 5)

    assert auv.status is AUVStatus.IDLE
    assert not auv.can_load_packets([first, second])
    with pytest.raises(ValueError, match="storage"):
        auv.load_packets([first, second])
    assert auv.cargo == ()
    auv.load_packets([first])
    assert auv.storage_used_bytes == 6
    assert auv.available_storage_bytes == 4
    assert not auv.can_load_packets([second])
    assert auv.can_load_packets([packet(3, 4)])


def test_auv_rejects_duplicates_and_removes_selected_packets_atomically():
    auv = AUV(1, (0.0, 0.0, 0.0), 1.0, 10)
    first, second = packet(1), packet(2)

    assert not auv.can_load_packets([first, first])
    auv.load_packets(iter((first, second)))
    assert auv.cargo == (first, second)
    assert not auv.can_load_packets([first])
    with pytest.raises(ValueError, match="unique"):
        auv.load_packets([first])
    with pytest.raises(KeyError, match="99"):
        auv.remove_packets([1, 99])
    assert auv.cargo == (first, second)
    assert auv.remove_packets([2, 1]) == (second, first)
    assert auv.storage_used_bytes == 0
    assert auv.available_storage_bytes == 10


def test_sink_orders_requests_by_arrival_then_enqueue_order():
    sink = Sink((0.0, 0.0, 0.0), 2.0)
    later = sink.enqueue_upload(1, [packet(1)], 5.0)
    first = sink.enqueue_upload(2, [packet(2)], 3.0)
    tied = sink.enqueue_upload(3, [packet(3)], 3.0)

    assert sink.waiting_queue == (first, tied, later)
    assert sink.start_next_upload(2.0) is None
    job = sink.start_next_upload(3.0)
    assert job.request == first
    assert sink.start_next_upload(4.0) is None
    sink.record_packet_delivery(2, job.finish_time_s, 20.0)
    job = sink.finish_active_upload(job.finish_time_s)
    assert job.request == tied
    sink.record_packet_delivery(3, job.finish_time_s, 20.0)
    assert sink.finish_active_upload(job.finish_time_s).request == later


def test_sink_assigns_individual_completion_times_in_packet_order():
    sink = Sink((0.0, 0.0, 0.0), 10.0)
    first, second, third = packet(3, 5), packet(1, 15), packet(2, 10)
    sink.enqueue_upload(1, [first, second, third], 1.0)
    job = sink.start_next_upload(2.0)

    assert job.start_time_s == 2.0
    assert [item.packet for item in job.packet_completions] == [first, second, third]
    assert [item.completion_time_s for item in job.packet_completions] == [2.5, 4.0, 5.0]
    assert job.finish_time_s == 5.0


def test_sink_records_only_valid_completions_and_preserves_auv_cargo():
    sink = Sink((0.0, 0.0, 0.0), 4.0)
    auv = AUV(1, sink.position, 1.0, 10)
    first, second = packet(1), packet(2)
    auv.load_packets([first, second])
    sink.enqueue_upload(auv.auv_id, auv.cargo, 1.0)
    job = sink.start_next_upload(1.0)
    assert sink.total_voi == 0.0

    with pytest.raises(KeyError, match="not in the active"):
        sink.record_packet_delivery(99, 2.0, 10.0)
    with pytest.raises(RuntimeError, match="upload order"):
        sink.record_packet_delivery(2, 3.0, 10.0)
    with pytest.raises(ValueError, match="scheduled time"):
        sink.record_packet_delivery(1, 1.5, 10.0)
    assert sink.delivered_records == ()

    record = sink.record_packet_delivery(1, 2.0, 10.0)
    assert record.packet is first
    assert record.auv_id == auv.auv_id
    assert record.delivery_time_s == 2.0
    assert sink.delivered_packet_ids == {1}
    assert sink.total_voi == first.voi_at_delivery(2.0, 10.0)
    assert auv.cargo == (first, second)
    assert auv.remove_packets([record.packet_id]) == (first,)
    assert sink.active_upload is job


def test_sink_rejects_duplicate_packets_in_queue_service_and_delivery():
    sink = Sink((0.0, 0.0, 0.0), 4.0)
    sample = packet(1)

    with pytest.raises(ValueError, match="unique"):
        sink.enqueue_upload(1, [sample, sample], 0.0)
    sink.enqueue_upload(1, [sample], 0.0)
    with pytest.raises(ValueError, match="queued or in service"):
        sink.enqueue_upload(2, [sample], 0.0)
    job = sink.start_next_upload(0.0)
    with pytest.raises(ValueError, match="queued or in service"):
        sink.enqueue_upload(2, [sample], 0.0)
    sink.record_packet_delivery(1, job.finish_time_s, 10.0)

    with pytest.raises(RuntimeError, match="already been delivered"):
        sink.record_packet_delivery(1, job.finish_time_s, 10.0)
    with pytest.raises(KeyError, match="not in the active"):
        sink.record_packet_delivery(99, job.finish_time_s, 10.0)
    assert sink.finish_active_upload(job.finish_time_s) is None
    with pytest.raises(ValueError, match="already been delivered"):
        sink.enqueue_upload(2, [sample], 2.0)
    assert len(sink.delivered_records) == 1


def test_sink_requires_scheduled_finish_time_and_all_packet_records():
    sink = Sink((0.0, 0.0, 0.0), 4.0)
    sink.enqueue_upload(1, [packet(1)], 0.0)
    sink.enqueue_upload(2, [packet(2)], 4.0)
    job = sink.start_next_upload(0.0)

    with pytest.raises(RuntimeError, match="scheduled finish"):
        sink.finish_active_upload(0.5)
    with pytest.raises(RuntimeError, match="undelivered"):
        sink.finish_active_upload(job.finish_time_s)
    assert sink.active_upload is job
    sink.record_packet_delivery(1, job.finish_time_s, 10.0)
    assert sink.finish_active_upload(job.finish_time_s) is None
    assert sink.start_next_upload(3.0) is None
    assert sink.start_next_upload(4.0).request.auv_id == 2


def test_sink_counts_packets_before_and_at_deadline_with_upload_still_active():
    sink = Sink((0.0, 0.0, 0.0), 4.0)
    first, second, third = packet(1), packet(2), packet(3)
    sink.enqueue_upload(1, [first, second, third], 1.0)
    job = sink.start_next_upload(1.0)
    deadline_s = 3.0

    first_record = sink.record_packet_delivery(1, 2.0, deadline_s)
    second_record = sink.record_packet_delivery(2, 3.0, deadline_s)
    assert first_record.voi > 0
    assert second_record.voi > 0
    assert sink.delivered_packet_ids == {1, 2}
    assert sink.active_upload is job
    assert job.finish_time_s > deadline_s
    expected_voi = first_record.voi + second_record.voi
    assert sink.total_voi == expected_voi

    late_record = sink.record_packet_delivery(3, job.finish_time_s, deadline_s)
    assert late_record.voi == 0.0
    assert sink.total_voi == expected_voi


def test_sink_checks_deadline_using_scheduled_time_even_if_input_was_rounded():
    sink = Sink((0.0, 0.0, 0.0), 10.0)
    sink.enqueue_upload(1, [packet(1, 2)], 0.1)
    job = sink.start_next_upload(0.1)

    record = sink.record_packet_delivery(1, 0.3, 0.3)

    assert record.delivery_time_s == job.finish_time_s
    assert record.voi == 0.0


def test_entity_configuration_and_upload_time_are_validated():
    with pytest.raises(ValueError, match="position"):
        Sink((0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="upload_rate_bytes_per_s"):
        Sink((0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="speed_mps"):
        AUV(1, (0.0, 0.0, 0.0), -1.0, 10)
    with pytest.raises(ValueError, match="buffer_capacity_bytes"):
        ClusterHead(0, (0.0, 0.0, 0.0), -1, 25.0)

    sink = Sink((0.0, 0.0, 0.0), 4.0)
    with pytest.raises(ValueError, match="at least one packet"):
        sink.enqueue_upload(1, [], 0.0)
    with pytest.raises(ValueError, match="arrival_time_s"):
        sink.enqueue_upload(1, [packet(1, generation_time_s=2.0)], 1.0)
    assert sink.waiting_queue == ()
