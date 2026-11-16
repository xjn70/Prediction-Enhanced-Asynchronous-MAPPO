"""Run with python -m scripts.run_generation_demo or python scripts/run_generation_demo.py."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.data_generation import GenerationManager, PacketParameters, RateSchedule, RateSegment
from uwsn.entities import ClusterHead, OverflowPolicy


def main() -> None:
    """Generate packets for three CHs during a reproducible 120-second mission."""
    deadline_s = 120.0
    manager = GenerationManager(deadline_s)
    schedules = (
        RateSchedule.fixed(0.5, deadline_s),
        RateSchedule.fixed(1.5, deadline_s),
        RateSchedule((
            RateSegment(0.0, 40.0, 0.2),
            RateSegment(40.0, 60.0, 0.0),
            RateSegment(60.0, deadline_s, 2.0),
        )),
    )
    policies = (OverflowPolicy.DROP_OLDEST, OverflowPolicy.DROP_NEWEST, OverflowPolicy.DROP_OLDEST)
    for ch_id, (schedule, policy) in enumerate(zip(schedules, policies)):
        ch = ClusterHead(
            ch_id=ch_id,
            position=(ch_id * 100.0, 0.0, -20.0),
            buffer_capacity_bytes=20_000,
            communication_radius_m=30.0,
            overflow_policy=policy,
        )
        manager.register_ch(ch, schedule, PacketParameters(1_000, 10.0, 0.01), seed=100 + ch_id)

    manager.initialize()
    while (event := manager.peek_next_event()) is not None:
        manager.process_event(event)

    print("Mission [0, 120) s; packet size 1000 bytes; CH capacity 20000 bytes")
    print("CH 0: 0.5 pps; DROP_OLDEST; seed 100")
    print("CH 1: 1.5 pps; DROP_NEWEST; seed 101")
    print("CH 2: 0.2 / 0 / 2.0 pps over [0,40) / [40,60) / [60,120); seed 102")
    print("Inserted counts all successful insertions, including packets later evicted.")
    print("CH  Expected  Generated  Inserted  Overflow  Oversize  GeneratedBytes  BufferBytes")
    for ch_id, schedule in enumerate(schedules):
        stats = manager.get_stats(ch_id)
        expected = schedule.integrated_intensity(0.0, deadline_s)
        print(
            f"{ch_id:2d}  {expected:8.1f}  {stats.generated_packets:9d}"
            f"  {stats.inserted_packets:8d}  {stats.overflow_dropped_packets:8d}"
            f"  {stats.oversized_rejected_packets:8d}  {stats.generated_bytes:14d}"
            f"  {stats.buffer_size_bytes:11d}"
        )


if __name__ == "__main__":
    main()
