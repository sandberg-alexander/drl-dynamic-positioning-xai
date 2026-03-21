"""Smoke tests for plotting entry points — verify they are importable and callable."""

from __future__ import annotations


class TestPlotEval:
    def test_importable(self):
        from milliampere_dp.plotting.plot_eval import plot_evaluation_rewards

        assert callable(plot_evaluation_rewards)

    def test_main_callable(self):
        from milliampere_dp.plotting.plot_eval import main

        assert callable(main)


class TestPlotMonitoredData:
    def test_importable(self):
        from milliampere_dp.plotting.plot_monitored_data import main

        assert callable(main)


class TestPlotRun:
    def test_importable(self):
        from milliampere_dp.plotting.plot_run import load_data, main

        assert callable(load_data)
        assert callable(main)


class TestPlotSpline:
    def test_importable(self):
        from milliampere_dp.plotting.plot_spline import main, plot_spline

        assert callable(plot_spline)
        assert callable(main)
