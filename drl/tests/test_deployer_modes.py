"""Tests for DeployMode enum (deployer class requires ROS, tested minimally)."""

from __future__ import annotations

from milliampere_drl.deployer import DeployMode


class TestDeployMode:
    def test_enum_values(self):
        assert DeployMode.DP == 0
        assert DeployMode.DP_TEST == 1
        assert DeployMode.NORTH_TEST == 2
        assert DeployMode.SPLINE_TEST == 3
        assert DeployMode.ACTION_SAMPLE == 4
        assert DeployMode.VF_SAMPLE == 5

    def test_all_modes_are_ints(self):
        for mode in DeployMode:
            assert isinstance(mode, int)

    def test_mode_count(self):
        assert len(DeployMode) == 6

    def test_mode_from_int(self):
        assert DeployMode(0) == DeployMode.DP
        assert DeployMode(3) == DeployMode.SPLINE_TEST
