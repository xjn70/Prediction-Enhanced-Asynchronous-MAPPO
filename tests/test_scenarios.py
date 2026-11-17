"""Scenario parameters and seed isolation at the episode boundary."""

from random import Random
from unittest.mock import patch

import pytest

from uwsn.data_generation import GenerationResult
from uwsn.entities import OverflowPolicy
from uwsn.environment import make_scenario
from uwsn.utils.seeding import derive_seed


def test_derive_seed_is_stable_and_collision_free_over_a_grid():
    assert derive_seed(1, "gen", 3) == derive_seed(1, "gen", 3)
    seeds = {derive_seed(m, role, i) for m in range(20) for role in ("gen", "init", "noise") for i in range(50)}
    assert len(seeds) == 20 * 3 * 50


@pytest.mark.parametrize("kind", ["S1", "S2", "S3"])
def test_scenarios_reproduce_episode_logs_from_the_master_seed(kind):
    original = make_scenario(kind, 41, horizon_s=120)
    first_log = original.run()
    assert first_log == make_scenario(kind, 41, horizon_s=120).run()
    assert first_log != make_scenario(kind, 42, horizon_s=120).run()
    assert all(ch.overflow_policy is OverflowPolicy.DROP_NEWEST for ch in original.cluster_heads.values())
    original.reset()
    assert original.run() == first_log


def test_equal_poisson_rates_still_use_distinct_ch_streams():
    with patch.object(Random, "gammavariate", return_value=0.2):
        sim = make_scenario("S1", 12, n_ch=2)
    assert sim.generation_manager.true_rate_at(0) == sim.generation_manager.true_rate_at(1) == 0.2
    times = {0: [], 1: []}
    for entry in sim.run():
        if isinstance(entry.result, GenerationResult):
            times[entry.result.packet.source_ch_id].append(entry.time_s)
    assert times[0] and times[1] and times[0] != times[1]


def test_s2_observes_random_rate_changes_per_ch():
    sim = make_scenario("S2", 41, horizon_s=600)
    manager = sim.generation_manager
    sequences = {tuple(manager.true_rate_at(ch_id, t) for t in range(600)) for ch_id in sim.cluster_heads}
    assert len(sequences) == len(sim.cluster_heads)
    assert all(rate > 0 for rates in sequences for rate in rates)
    assert any(len(set(rates)) > 2 for rates in sequences)
    assert all(manager.true_rate_at(ch_id, 600) == 0 for ch_id in sim.cluster_heads)


def test_s3_uses_two_rates_with_the_prescribed_burst_ratio():
    sim = make_scenario("S3", 41)
    rates_by_ch = {ch_id: set() for ch_id in sim.cluster_heads}
    for time_s in range(600):
        sim.run_until(time_s)
        for ch_id, rates in rates_by_ch.items():
            rates.add(sim.generation_manager.true_rate_at(ch_id))
    assert all(1 <= len(rates) <= 2 for rates in rates_by_ch.values())
    ratios = [max(rates) / min(rates) for rates in rates_by_ch.values() if len(rates) == 2]
    assert ratios and ratios == pytest.approx([4.5] * len(ratios))
