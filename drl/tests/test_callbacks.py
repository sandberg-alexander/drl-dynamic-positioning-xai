"""Tests for SB3 training callbacks."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import pytest

from milliampere_drl.callbacks import SaveModelCallback


class TestSaveModelCallback:
    @pytest.fixture()
    def callback(self, tmp_path):
        cb = SaveModelCallback(save_interval=2, save_path=str(tmp_path), verbose=0)
        # Mock the SB3 model
        cb.model = MagicMock()
        cb.model.ep_info_buffer = deque([{"r": 10.0}, {"r": 20.0}], maxlen=100)
        cb.num_timesteps = 1000
        return cb

    def test_saves_on_interval(self, callback, tmp_path):
        # First rollout -- should NOT save (1 % 2 != 0)
        callback._on_rollout_end()
        callback.model.save.assert_not_called()

        # Second rollout -- SHOULD save (2 % 2 == 0)
        callback._on_rollout_end()
        callback.model.save.assert_called_once()

    def test_skips_when_not_interval(self, tmp_path):
        cb = SaveModelCallback(save_interval=3, save_path=str(tmp_path), verbose=0)
        cb.model = MagicMock()
        cb.model.ep_info_buffer = deque(maxlen=100)
        cb.num_timesteps = 1000

        cb._on_rollout_end()
        cb._on_rollout_end()
        cb.model.save.assert_not_called()

        cb._on_rollout_end()  # 3rd rollout -- should save
        cb.model.save.assert_called_once()

    def test_on_step_returns_true_by_default(self, callback):
        assert callback._on_step() is True

    def test_request_stop_halts_training(self, callback):
        assert callback._on_step() is True
        callback.request_stop()
        assert callback._on_step() is False
