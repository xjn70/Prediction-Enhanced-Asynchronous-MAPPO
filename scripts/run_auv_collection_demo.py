"""Run three scripted AUVs against five continuously generating CHs."""

from collections import deque

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.data_generation import GenerationManager, PacketParameters, RateSchedule
from uwsn.entities import AUV, ClusterHead, Sink
from uwsn.environment import CHArrival, CollectionResult, EventType, Simulator


def main() -> None:
    deadline_s = 75.0
    manager = GenerationManager(deadline_s)
    positions = ((12, 0, -5), (0, 24, -5), (25, 8, -7), (0, -15, -8), (-12, 10, -4))
    rates = (0.70, 0.40, 0.55, 0.20, 0.30)
    parameters = PacketParameters(500, 1.0, 0.01)
    for index, (position, rate) in enumerate(zip(positions, rates)):
        ch = ClusterHead(index, position, buffer_capacity_bytes=6000, communication_radius_m=3)
        manager.register_ch(ch, RateSchedule.fixed(rate, deadline_s), parameters, seed=70 + index)
    auvs = (
        AUV(0, (0, 0, 0), 1, storage_capacity_bytes=2000),
        AUV(1, (0, 0, 0), 1, storage_capacity_bytes=1500),
        AUV(2, (0, 0, -5), 1.5, storage_capacity_bytes=2500),
    )
    sim = Simulator(deadline_s, generation_manager=manager, auvs=auvs,
                    sink=Sink((0, 0, 0), 1000), collection_rate_bytes_per_s=250)
    targets = {0: deque([0]), 1: deque([0, 2]), 2: deque([1])}
    collection_starts: dict[int, tuple[int, int, int]] = {}
    for auv in auvs:
        print(f"AUV {auv.auv_id} initial position={auv.position}, targets={list(targets[auv.auv_id])}")
        sim.schedule_event(0, EventType.AUV_DECISION_READY, auv.auv_id)

    while (entry := sim.step_event()) is not None:
        auv_id = entry.entity_id
        if auv_id is None or entry.event_type is EventType.CH_DATA_GENERATION:
            continue
        auv = sim.auvs[auv_id]
        prefix = f"t={entry.time_s:6.2f} s AUV {auv_id}"
        if isinstance(entry.result, CHArrival):
            report = entry.result.report
            print(f"{prefix} arrived at CH {report.ch_id} at {auv.position}, report t={report.report_time_s:.2f} s, "
                  f"cache={report.buffer_size_bytes} B, outcome={entry.result.outcome}")
            if entry.result.outcome == "collection_started":
                collection_starts[auv_id] = (report.ch_id, len(auv.cargo), auv.storage_used_bytes)
                print(f"{prefix} collection started")
        elif isinstance(entry.result, CollectionResult):
            if entry.result.completed:
                print(f"{prefix} completed {entry.result.packet.size_bytes} B, cargo={auv.storage_used_bytes} B")
            else:
                print(f"{prefix} collection interrupted, unfinished packet retained at CH")
        elif entry.event_type is EventType.AUV_DECISION_READY and auv_id in sim.decision_ready:
            pending = collection_starts.pop(auv_id, None)
            if pending is not None:
                ch_id, old_count, old_bytes = pending
                print(f"{prefix} collection ended at CH {ch_id}: {len(auv.cargo) - old_count} packets, "
                      f"{auv.storage_used_bytes - old_bytes} B, CH cache={sim.cluster_heads[ch_id].buffer_size_bytes} B")
            if targets[auv_id]:
                ch_id = targets[auv_id].popleft()
                leg = sim.visit(auv_id, ch_id)
                print(f"{prefix} Visit({ch_id}): {leg.distance_m:.2f} m, ETA={entry.time_s + leg.travel_time_s:.2f} s")
            else:
                leg = sim.return_to_sink(auv_id)
                print(f"{prefix} Return: {leg.distance_m:.2f} m, ETA={entry.time_s + leg.travel_time_s:.2f} s")
        elif entry.event_type is EventType.AUV_ARRIVAL_SINK:
            print(f"{prefix} arrived at Sink, {auv.status.name}, cargo={len(auv.cargo)} packets "
                  f"({auv.storage_used_bytes} B)")

    for ch_id, ch in sim.cluster_heads.items():
        stats = sim.generation_manager.get_stats(ch_id)
        print(f"CH {ch_id} final: generated={stats.generated_packets}, cache={len(ch.buffer)} packets "
              f"({ch.buffer_size_bytes} B), dropped={stats.overflow_dropped_packets + stats.oversized_rejected_packets}")
    generated = sum(ch.total_generated_packets for ch in sim.cluster_heads.values())
    cached = sum(len(ch.buffer) for ch in sim.cluster_heads.values())
    carried = sum(len(auv.cargo) for auv in sim.auvs.values())
    dropped = sum(sim.generation_manager.get_stats(ch_id).overflow_dropped_packets
                  + sim.generation_manager.get_stats(ch_id).oversized_rejected_packets for ch_id in sim.cluster_heads)
    print(f"Packet accounting: {generated} generated = {cached} cached + {carried} carried + {dropped} dropped")


if __name__ == "__main__":
    main()
