"""Show a 120-second mixed-generation run and asynchronous AUV events."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.data_generation import (
    GenerationManager, GenerationMode, GenerationResult, MMPPConfig, PacketParameters, RateSchedule, RateSegment,
)
from uwsn.entities import AUV, ClusterHead, Sink
from uwsn.environment import EventType, Simulator


def main() -> None:
    deadline_s = 120.0
    manager = GenerationManager(deadline_s)
    parameters = PacketParameters(1000, 10.0, 0.01)
    heads = [ClusterHead(ch_id, (100.0 * ch_id, 0, -20), 20_000, 30) for ch_id in range(3)]
    manager.register_ch(heads[0], RateSchedule.fixed(0.35, deadline_s), parameters, seed=401)
    manager.register_ch(heads[1], RateSchedule((
        RateSegment(0, 40, 0.2), RateSegment(40, 60, 0), RateSegment(60, deadline_s, 0.7),
    )), parameters, seed=402)
    config = MMPPConfig(0.05, 0.85, 0.08, 0.12, GenerationMode.STATIONARY, random_seed=403)
    manager.register_mmpp_ch(heads[2], config, parameters)
    auvs = [AUV(auv_id, (0, 0, 0), 1.0, 50_000) for auv_id in range(3)]
    sim = Simulator(deadline_s, generation_manager=manager, auvs=auvs, sink=Sink((0, 0, 0), 2000))
    for auv_id, time_s in enumerate((100, 115, 70)):
        sim.schedule_event(time_s, EventType.WAIT_COMPLETE, auv_id)

    print("Mission [0, 120) s | CH 0 fixed Poisson | CH 1 segmented Poisson | CH 2 MMPP")
    sim.run()
    ready_times: list[tuple[int | None, float]] = []
    for entry in sim.event_log:
        if isinstance(entry.result, GenerationResult):
            result = f"packet_id={entry.result.packet.packet_id}"
        else:
            result = entry.result
            if result == "decision_ready":
                ready_times.append((entry.entity_id, entry.time_s))
        entity = "-" if entry.entity_id is None else str(entry.entity_id)
        print(f"{entry.time_s:9.4f} s  {entry.event_type.name:<19}  entity={entity}  {result}")

    for ch_id, ch in sim.cluster_heads.items():
        stats = sim.generation_manager.get_stats(ch_id)
        print(f"CH {ch_id}: generated={stats.generated_packets}, final buffer={len(ch.buffer)} packets "
              f"({stats.buffer_size_bytes} bytes), overflow={stats.overflow_dropped_packets}")
    mode = sim.generation_manager.get_mode_stats(2)
    print(f"CH 2 MMPP diagnostics: NORMAL={mode.normal_mode_duration_s:.4f} s, "
          f"BURST={mode.burst_mode_duration_s:.4f} s, accounted through {mode.accounted_until_s:g} s")
    print("Decision-ready events: " + ", ".join(f"AUV {idx} at {time:g} s" for idx, time in ready_times))
    print(f"Total processed events: {len(sim.event_log)}")


if __name__ == "__main__":
    main()
