"""Tests for milliampere_dp.vessel constants."""

import numpy as np

from milliampere_dp.vessel import (
    ACTUATOR_CONSTRAINTS,
    DEFAULT_INITIAL_ANGLES,
    HIGH_ACTION,
    LOW_ACTION,
    MAX_ANGULAR_SPEED,
    MAX_AZIMUTH_ANGLE,
    MAX_DISTANCE,
    MAX_HEADING_ANGLE,
    MAX_LINEAR_SPEED,
    MAX_THRUSTER_RPM,
    NED_EAST_OFFSET,
    NED_NORTH_OFFSET,
    OPPOSITE_QUADRANTS,
    VESSEL_BEAM,
    VESSEL_LENGTH,
)


def test_max_thruster_rpm():
    assert MAX_THRUSTER_RPM == 900


def test_max_azimuth_angle():
    assert MAX_AZIMUTH_ANGLE == 90


def test_max_distance():
    assert MAX_DISTANCE == 10.0


def test_physical_limits_positive():
    assert MAX_LINEAR_SPEED > 0
    assert MAX_ANGULAR_SPEED > 0
    assert MAX_HEADING_ANGLE > 0


def test_ned_offsets():
    assert NED_NORTH_OFFSET == 334.61
    assert NED_EAST_OFFSET == 990.24


def test_vessel_dimensions():
    assert VESSEL_LENGTH == 5.06
    assert VESSEL_BEAM == 2.86


def test_actuator_constraints_count():
    assert len(ACTUATOR_CONSTRAINTS) == 4


def test_opposite_quadrants_count():
    assert len(OPPOSITE_QUADRANTS) == 4


def test_opposite_quadrants_are_pi_apart():
    for (lo, hi), (opp_lo, opp_hi) in zip(
        ACTUATOR_CONSTRAINTS, OPPOSITE_QUADRANTS, strict=True
    ):
        mid = (lo + hi) / 2
        opp_mid = (opp_lo + opp_hi) / 2
        diff = abs(mid - opp_mid)
        assert abs(diff - np.pi) < 0.01


def test_action_space_bounds_shape():
    assert LOW_ACTION.shape == (8,)
    assert HIGH_ACTION.shape == (8,)


def test_low_action_le_high_action():
    assert np.all(LOW_ACTION <= HIGH_ACTION)


def test_default_initial_angles_shape():
    assert DEFAULT_INITIAL_ANGLES.shape == (4,)
