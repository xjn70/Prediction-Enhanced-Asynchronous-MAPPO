"""Geometry and time calculations for the AUV navigation legs."""

import pytest

from uwsn.navigation import distance_m, in_communication_range, plan_return, plan_visit


def test_3d_geometry_uses_the_closest_boundary_point():
    start, center = (3, 4, -12), (0, 0, 0)
    assert distance_m(start, center) == 13
    leg = plan_visit(start, center, 2, 2)
    assert leg.destination == pytest.approx((6 / 13, 8 / 13, -24 / 13))
    assert distance_m(leg.destination, center) == pytest.approx(2)
    assert distance_m(start, leg.destination) == pytest.approx(11)
    assert leg.distance_m == 11 and leg.travel_time_s == 5.5


@pytest.mark.parametrize("position", [(0, 0, 0), (0, 0, 0.5), (0, 0, 1)])
def test_inside_or_on_the_sphere_starts_service_at_the_current_position(position):
    leg = plan_visit(position, (0, 0, 0), 1, 2)
    assert leg.destination == position and leg.distance_m == leg.travel_time_s == 0


def test_boundary_tolerance_accepts_roundoff_but_rejects_out_of_range_points():
    assert in_communication_range((1 + 1e-12, 0, 0), (0, 0, 0), 1)
    assert not in_communication_range((1 + 1e-6, 0, 0), (0, 0, 0), 1)


def test_return_targets_sink_center():
    leg = plan_return((0, 0, 0), (3, 4, -12), 2)
    assert leg.destination == (3, 4, -12)
    assert leg.distance_m == 13 and leg.travel_time_s == 6.5


def test_invalid_speed_and_coordinates_are_rejected():
    with pytest.raises(ValueError, match="speed_mps"):
        plan_visit((0, 0, 0), (3, 4, 0), 1, 0)
    with pytest.raises(ValueError, match="position"):
        distance_m((0, 0, 0), (float("inf"), 0, 0))
