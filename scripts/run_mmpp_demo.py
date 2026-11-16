"""Show simulation-only MMPP diagnostics for three independently seeded CHs."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.data_generation import GenerationManager, GenerationMode, MMPPConfig, PacketParameters
from uwsn.entities import ClusterHead, OverflowPolicy


def main() -> None:
    """Drain a finite mission and report generation and final buffer statistics."""
    deadline_s = 120.0
    manager = GenerationManager(deadline_s)
    configs = (
        MMPPConfig(0.2, 2.0, 0.08, 0.32, GenerationMode.STATIONARY, random_seed=301),
        MMPPConfig(0.5, 3.5, 0.16, 0.24, GenerationMode.STATIONARY, random_seed=302),
        MMPPConfig(0.0, 5.0, 0.12, 0.18, GenerationMode.STATIONARY, random_seed=303),
    )
    capacities = (20_000, 30_000, 40_000)
    policies = (OverflowPolicy.DROP_OLDEST, OverflowPolicy.DROP_NEWEST, OverflowPolicy.DROP_OLDEST)
    heads = []
    for ch_id, (config, capacity, policy) in enumerate(zip(configs, capacities, policies)):
        ch = ClusterHead(ch_id, (100.0 * ch_id, 0.0, -20.0), capacity, 30.0, config.lambda_normal_pps, policy)
        manager.register_mmpp_ch(ch, config, PacketParameters(1000, 10.0, 0.01))
        heads.append(ch)

    manager.initialize()
    while (event := manager.peek_next_event()) is not None:
        manager.process_event(event)

    print("Simulation-only MMPP diagnostics | Mission [0, 120) s | packet size 1000 bytes")
    for ch, config in zip(heads, configs):
        mode = manager.get_mode_stats(ch.ch_id)
        stats = manager.get_stats(ch.ch_id)
        pi_normal, pi_burst = config.stationary_probabilities()
        expected_rate = pi_normal * config.lambda_normal_pps + pi_burst * config.lambda_burst_pps
        print(
            f"CH {ch.ch_id} | seed={config.random_seed} | {ch.overflow_policy.name} | "
            f"lambda={config.lambda_normal_pps:g}/{config.lambda_burst_pps:g} pps | "
            f"q={config.q_normal_to_burst_per_s:g}/{config.q_burst_to_normal_per_s:g} /s"
        )
        print(
            f"  Switches N->B={mode.normal_to_burst_transitions} B->N={mode.burst_to_normal_transitions} | "
            f"duration N={mode.normal_mode_duration_s:.2f} s B={mode.burst_mode_duration_s:.2f} s"
        )
        print(
            f"  Generated N={mode.packets_generated_in_normal} B={mode.packets_generated_in_burst} "
            f"total={stats.generated_packets} | stationary expected rate={expected_rate:.2f} pps"
        )
        print(
            f"  Inserted={stats.inserted_packets} overflow={stats.overflow_dropped_packets} "
            f"oversize={stats.oversized_rejected_packets} | "
            f"final buffer={len(ch.buffer)} packets ({stats.buffer_size_bytes}/{ch.buffer_capacity_bytes} bytes)"
        )


if __name__ == "__main__":
    main()
