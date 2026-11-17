"""Demonstrate FCFS waiting, individual packet deliveries, and the cutoff."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_auv_collection_demo import make_collection_scenario
from uwsn.entities import DeliveryRecord, UploadJob
from uwsn.environment import EventType


def main() -> None:
    sim, targets = make_collection_scenario(upload_rate_bytes_per_s=100)
    sink = sim.sink
    assert sink is not None
    returned: set[int] = set()
    print(f"Mission: {sim.mission_duration_s:g} s, Sink: {sink.upload_rate_bytes_per_s:g} B/s")
    for auv_id in sorted(sim.auvs):
        sim.schedule_event(0, EventType.AUV_DECISION_READY, auv_id)

    while (entry := sim.step_event()) is not None:
        kind, auv_id = entry.event_type, entry.entity_id
        if auv_id is None or kind is EventType.CH_DATA_GENERATION:
            continue
        prefix = f"t={entry.time_s:6.2f} s  AUV {auv_id}"
        if kind is EventType.AUV_DECISION_READY and auv_id in sim.decision_ready:
            if targets[auv_id]:
                sim.visit(auv_id, targets[auv_id].popleft())
            elif auv_id not in returned:
                sim.return_to_sink(auv_id)
                returned.add(auv_id)
        elif kind is EventType.AUV_ARRIVAL_SINK:
            print(f"{prefix}  arrived, cargo={len(sim.auvs[auv_id].cargo)} packets, "
                  f"requests awaiting start={len(sink.waiting_queue)}")
        elif kind is EventType.UPLOAD_START:
            job = entry.result
            assert isinstance(job, UploadJob)
            wait_s = job.start_time_s - job.request.arrival_time_s
            print(f"{prefix}  upload started, queue wait={wait_s:.2f} s, "
                  f"scheduled finish={job.finish_time_s:.2f} s, queue length={len(sink.waiting_queue)}")
        elif isinstance(entry.result, DeliveryRecord):
            record = entry.result
            print(f"{prefix}  packet {record.packet_id} delivered ({record.size_bytes} B), "
                  f"VoI={record.delivered_voi:.6f}, remaining cargo={len(sim.auvs[auv_id].cargo)}")
        elif kind is EventType.UPLOAD_COMPLETE:
            print(f"{prefix}  upload finished, remaining cargo={len(sim.auvs[auv_id].cargo)}")

    for request in sink.interrupted_uploads:
        state = "queued" if request.start_time_s is None else "uploading"
        print(f"Cutoff: AUV {request.auv_id} was {state}; {len(request.packets)} undelivered packets retained")
    for auv_id, auv in sorted(sim.auvs.items()):
        print(f"AUV {auv_id} final cargo: {len(auv.cargo)} packets, {auv.storage_used_bytes} B")

    generated = sum(ch.total_generated_packets for ch in sim.cluster_heads.values())
    cached = sum(len(ch.buffer) for ch in sim.cluster_heads.values())
    carried = sum(len(auv.cargo) for auv in sim.auvs.values())
    delivered = len(sink.delivered_records)
    stats = [sim.generation_manager.get_stats(ch_id) for ch_id in sim.cluster_heads]
    dropped = sum(ch.overflow_dropped_packets + ch.oversized_rejected_packets for ch in stats)
    assert generated == cached + carried + delivered + dropped
    print(f"Sink: delivered={delivered}, cumulative VoI={sink.total_voi:.6f}, "
          f"final queue length={len(sink.waiting_queue)}")
    print(f"Undelivered packets: {generated - delivered} total, {carried} still in AUV cargo")
    print(f"Packet accounting: {generated} generated = {cached} cached + {carried} carried "
          f"+ {delivered} delivered + {dropped} dropped")


if __name__ == "__main__":
    main()
