"""Reproducible S1–S3 debug scenarios from research plan v1.1 §3.2."""

from math import log, nextafter
from random import Random

from ..data_generation import (
    GenerationManager,
    GenerationMode,
    MMPPConfig,
    PacketParameters,
    RateSchedule,
    RateSegment,
)
from ..entities import AUV, ClusterHead, OverflowPolicy, Sink
from ..entities.common import _require_int
from ..utils.seeding import derive_seed
from .simulator import Simulator


def _piecewise_schedule(rng: Random, horizon_s: float) -> RateSchedule:
    segments = []
    start_s = 0.0
    while start_s < horizon_s:
        rate_pps = rng.gammavariate(2, 0.1)
        duration_s = horizon_s / 20 + rng.expovariate(1.0) * (horizon_s / 3)
        end_s = min(horizon_s, max(start_s + duration_s, nextafter(start_s, horizon_s)))
        segments.append(RateSegment(start_s, end_s, rate_pps))
        start_s = end_s
    return RateSchedule(tuple(segments))


def make_scenario(
    kind: str,
    master_seed: int,
    *,
    n_ch: int = 5,
    n_auv: int = 3,
    horizon_s: float = 600.0,
    record_log: bool = True,
    include_generation_counter: bool = True,
) -> Simulator:
    """Build one S1, S2 or S3 episode with independent rates and arrival streams.

    Build again with a different master seed for a new episode. Resetting the
    returned simulator replays its original scenario and generation streams.
    """
    if kind not in ("S1", "S2", "S3"):
        raise ValueError(f"unknown scenario kind {kind!r}")
    _require_int(n_ch, "n_ch", minimum=1)
    _require_int(n_auv, "n_auv", minimum=1)
    manager = GenerationManager(horizon_s)
    horizon_s = manager.mission_deadline_s
    seeds = [derive_seed(master_seed, "gen", ch_id) for ch_id in range(n_ch)]
    if len(set(seeds)) != n_ch:
        raise ValueError("derived CH generation seeds collided; use a different master_seed")

    # Under the default VoI weights, an age of 300 s leaves half the initial value.
    parameters = PacketParameters(1000, 1.0, log(3.5) / 300)
    for ch_id, seed in enumerate(seeds):
        rng = Random(derive_seed(master_seed, "rate", ch_id))
        ch = ClusterHead(
            ch_id, (50.0 * ch_id, 0.0, -20.0), 20_000, 30.0,
            overflow_policy=OverflowPolicy.DROP_NEWEST,
        )
        if kind == "S2":
            manager.register_ch(ch, _piecewise_schedule(rng, horizon_s), parameters, seed=seed)
        else:
            rate_pps = rng.gammavariate(2, 0.1)  # Gamma shape 2, rate 10.
            if kind == "S1":
                manager.register_ch(ch, RateSchedule.fixed(rate_pps, horizon_s), parameters, seed=seed)
            else:
                # A 4.5x burst rate and 15% stationary burst occupancy.
                config = MMPPConfig(
                    rate_pps, 4.5 * rate_pps, 0.006, 0.034,
                    GenerationMode.STATIONARY, random_seed=seed,
                )
                manager.register_mmpp_ch(ch, config, parameters)

    auvs = [AUV(auv_id, (0.0, 0.0, 0.0), 1.5, 100_000) for auv_id in range(n_auv)]
    return Simulator(
        horizon_s, generation_manager=manager, auvs=auvs, sink=Sink((0.0, 0.0, 0.0), 500.0),
        record_log=record_log, include_generation_counter=include_generation_counter,
    )
