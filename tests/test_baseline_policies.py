"""Local observations, public action rules, and rule-based policy choices."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from uwsn.baselines import CyclicPatrol, NearestNeighbor, ObservedVoIGreedy
from uwsn.data_generation import GenerationManager, MMPPConfig, PacketParameters, RateSchedule
from uwsn.entities import AUV, CHReport, ClusterHead, DataPacket, Sink
from uwsn.environment import EventType, Simulator
from uwsn.environment.action_rules import ReturnRuleConfig, evaluate_actions
from uwsn.environment.actions import ReturnAction, VisitAction
from uwsn.environment.observation import build_local_observation


RULE = ReturnRuleConfig(storage_threshold=0.75, return_reserve_s=3, revisit_cooldown_s=5)
PROFILE = PacketParameters(1, 1, 0.1)


def local_map(positions=((0, 0, 0), (3, 0, 0), (7, 0, 0)), radii=(1, 1, 1), *, capacity=4):
    manager = GenerationManager(100)
    for index, (position, radius) in enumerate(zip(positions, radii)):
        ch = ClusterHead(index, position, 10, radius)
        manager.register_ch(ch, RateSchedule.fixed(0, 100), PROFILE, seed=index)
    sim = Simulator(
        100, generation_manager=manager, auvs=[AUV(0, (0, 0, 0), 1, capacity)],
        sink=Sink((0, 0, 0), 1), collection_rate_bytes_per_s=1,
    )
    sim.schedule_event(0, EventType.AUV_DECISION_READY, 0)
    sim.run_until(0)
    return sim


def ready_observation(sim):
    obs = build_local_observation(sim, 0)
    return replace(obs, legal_actions=evaluate_actions(obs, RULE).actions)


def test_cyclic_wraps_and_preserves_its_index_through_return():
    obs = ready_observation(local_map())
    patrol = CyclicPatrol(initial_offset=1)
    assert [patrol.choose_action(obs) for _ in range(3)] == [VisitAction(1), VisitAction(2), VisitAction(0)]
    assert patrol.choose_action(replace(obs, legal_actions=(ReturnAction(),))) == ReturnAction()
    assert patrol.choose_action(obs) == VisitAction(1)


def test_cyclic_skips_temporarily_masked_sites_and_keeps_independent_indices():
    obs = ready_observation(local_map())
    first, second = CyclicPatrol(0), CyclicPatrol(2)
    assert first.choose_action(replace(obs, legal_actions=(VisitAction(1), VisitAction(2)))) == VisitAction(1)
    assert second.choose_action(obs) == VisitAction(2)
    assert first.choose_action(obs) == VisitAction(2)
    assert second.choose_action(obs) == VisitAction(0)


def test_nearest_neighbor_uses_three_dimensional_distance_to_sphere():
    obs = ready_observation(local_map(((4, 0, 0), (5, 0, 12)), (1, 12)))
    assert [obs.visit_leg(ch).distance_m for ch in obs.chs] == [3, 1]
    assert NearestNeighbor().choose_action(obs) == VisitAction(1)


def test_nearest_neighbor_ties_by_id_and_respects_cooldown():
    sim = local_map(((0, 0, 1), (1, 0, 0)), (2, 2))
    obs = ready_observation(sim)
    assert NearestNeighbor().choose_action(obs) == VisitAction(0)
    obs = build_local_observation(sim, 0, cooldowns={0: 5})
    options = evaluate_actions(obs, RULE)
    assert NearestNeighbor().choose_action(replace(obs, legal_actions=options.actions)) == VisitAction(1)


def test_unknown_and_stale_reports_do_not_refresh_from_live_cache():
    sim = local_map()
    before = build_local_observation(sim, 0)
    sim.cluster_heads[0].generate_packet(DataPacket(10, 0, 0, 1, 1, 0.1))
    sim.cluster_heads[1].generate_packet(DataPacket(11, 1, 0, 1, 1, 0.1))
    assert before == build_local_observation(sim, 0)

    sim.visit(0, 0)
    sim.run_until(1)
    original = build_local_observation(sim, 0)
    sim.cluster_heads[0].generate_packet(DataPacket(20, 0, 2, 1, 1, 0.1))
    sim.run_until(2)
    latest = build_local_observation(sim, 0)
    assert latest.chs[0].report == original.chs[0].report == CHReport(0, 0, 1, (0,))
    assert latest.chs[1].report is None
    assert latest.cargo[0].generation_time_s == 0
    assert set(asdict(latest.cargo[0])) == {
        "source_ch_id", "generation_time_s", "size_bytes", "base_value", "decay_rate_per_s",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(latest, "storage_used_bytes", 0)


def test_only_the_visiting_auv_receives_a_ch_report():
    ch = ClusterHead(0, (0, 0, 0), 10, 1)
    vehicles = [AUV(auv_id, (0, 0, 0), 1, 10) for auv_id in (0, 1)]
    sim = Simulator(20, cluster_heads=[ch], auvs=vehicles, sink=Sink((0, 0, 0), 1))
    sim.schedule_event(0, EventType.AUV_DECISION_READY, 0)
    sim.run_until(0)
    sim.visit(0, 0)
    sim.run_until(0)
    assert build_local_observation(sim, 0).chs[0].report == CHReport(0, 0, 0, ())
    assert build_local_observation(sim, 1).chs[0].report is None


def test_different_hidden_mmpp_modes_have_identical_local_observations(monkeypatch):
    from uwsn.data_generation import GenerationMode

    def build(mode):
        manager = GenerationManager(20)
        ch = ClusterHead(0, (0, 0, 0), 10, 1)
        manager.register_mmpp_ch(ch, MMPPConfig(0, 4, 0, 0, mode, random_seed=5), PROFILE)
        return Simulator(20, generation_manager=manager, auvs=[AUV(0, (0, 0, 0), 1, 10)],
                         sink=Sink((0, 0, 0), 1))

    normal, burst = build(GenerationMode.NORMAL), build(GenerationMode.BURST)
    normal.run_until(2)
    burst.run_until(2)
    assert len(burst.cluster_heads[0].buffer) > len(normal.cluster_heads[0].buffer)

    def denied(*args):
        pytest.fail("an observation must never query Oracle rates")

    monkeypatch.setattr(GenerationManager, "true_rate_at", denied)
    assert build_local_observation(normal, 0) == build_local_observation(burst, 0)


def test_greedy_explores_nearest_unknown_when_reports_have_no_deliverable_value():
    obs = ready_observation(local_map())
    sites = (replace(obs.chs[0], report=CHReport(0, 0, 0, ())), *obs.chs[1:])
    obs = replace(obs, chs=sites)
    greedy = ObservedVoIGreedy()
    assert all(greedy.score(obs, ch) == 0 for ch in sites)
    assert greedy.choose_action(obs) == VisitAction(1)


def test_greedy_uses_received_history_even_if_a_nearer_unvisited_site_has_packets():
    sim = local_map(((0, 0, 0), (3, 0, 0)), (1, 1), capacity=8)
    for packet_id in range(2):
        sim.cluster_heads[1].generate_packet(DataPacket(packet_id, 1, 0, 1, 1, 0.1))
    sim.visit(0, 1)
    sim.run_until(4)
    obs = ready_observation(sim)
    assert obs.chs[1].report.packet_timestamps == (0, 0)
    assert sim.cluster_heads[1].buffer == ()
    sim.cluster_heads[0].generate_packet(DataPacket(10, 0, 4, 1, 100, 0))
    assert ready_observation(sim) == obs
    assert ObservedVoIGreedy().choose_action(obs) == VisitAction(1)


@pytest.mark.parametrize(("deadline", "expected_count"), [(13.5, 1), (14, 2), (100, 2)])
def test_greedy_scores_only_fifo_prefix_fitting_capacity_and_deadline(deadline, expected_count):
    initial = ready_observation(local_map(capacity=2))
    obs = replace(initial, current_time_s=10, mission_deadline_s=deadline)
    ch = replace(obs.chs[0], report=CHReport(0, 0, 3, (0.0, 0.5, 1.0)))
    expected = sum(
        DataPacket(index, 0, timestamp, 1, 1, 0.1).voi_at_delivery(10 + expected_count + index + 1, deadline)
        for index, timestamp in enumerate(ch.report.packet_timestamps[:expected_count])
    )
    expected /= 2 * expected_count * (1 + 10 / 120)
    assert ObservedVoIGreedy().score(obs, ch) == pytest.approx(expected)


def test_full_auv_must_return_and_empty_auv_at_sink_cannot_return():
    sim = local_map(capacity=1)
    obs = ready_observation(sim)
    assert obs.legal_actions and all(isinstance(action, VisitAction) for action in obs.legal_actions)
    sim.auvs[0].load_packets((DataPacket(1, 0, 0, 1, 1, 0.1),))
    obs = build_local_observation(sim, 0)
    options = evaluate_actions(obs, RULE)
    assert options.actions == (ReturnAction(),) and options.reason == "cargo_threshold"


def test_return_reserve_uses_only_own_cargo_and_public_terminal_rates():
    sim = local_map(capacity=20)
    sim.auvs[0].load_packets((DataPacket(1, 0, 0, 1, 1, 0.1),))
    obs = replace(build_local_observation(sim, 0), mission_deadline_s=4)
    assert evaluate_actions(obs, RULE).reason == "deadline_reserve"
    sim.visit(0, 1)
    busy = build_local_observation(sim, 0)
    assert evaluate_actions(busy, RULE).actions == ()
