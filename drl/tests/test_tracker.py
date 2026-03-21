"""Tests for EvaluationTracker and append_monitor_data."""

from __future__ import annotations

import csv

from milliampere_drl.tracker import EvaluationTracker, append_monitor_data


class TestEvaluationTracker:
    def test_fresh_tracker_has_empty_progress(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        assert tracker.get_current_model() is None
        assert tracker.get_completed_episodes() == 0
        assert tracker.get_all_completed_models() == []

    def test_start_model_sets_current(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        tracker.start_model_evaluation("/path/to/model.zip")
        assert tracker.get_current_model() == "/path/to/model.zip"

    def test_complete_episode_increments(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        tracker.start_model_evaluation("/path/to/model.zip")
        tracker.complete_episode()
        assert tracker.get_completed_episodes() == 1
        tracker.complete_episode()
        assert tracker.get_completed_episodes() == 2

    def test_complete_model_moves_to_completed(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        tracker.start_model_evaluation("/path/to/model.zip")
        tracker.complete_model_evaluation()
        assert tracker.is_model_completed("/path/to/model.zip")
        assert tracker.get_current_model() is None
        assert tracker.get_completed_episodes() == 0

    def test_is_model_completed_false_for_unknown(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        assert not tracker.is_model_completed("/unknown/model.zip")

    def test_persistence_across_instances(self, tmp_path):
        t1 = EvaluationTracker(str(tmp_path))
        t1.start_model_evaluation("/path/to/model.zip")
        t1.complete_episode()
        t1.complete_episode()

        t2 = EvaluationTracker(str(tmp_path))
        assert t2.get_current_model() == "/path/to/model.zip"
        assert t2.get_completed_episodes() == 2

    def test_resume_same_model_does_not_reset(self, tmp_path):
        tracker = EvaluationTracker(str(tmp_path))
        tracker.start_model_evaluation("/path/to/model.zip")
        tracker.complete_episode()
        # Start same model again -- should resume, not reset
        tracker.start_model_evaluation("/path/to/model.zip")
        assert tracker.get_completed_episodes() == 1


class TestAppendMonitorData:
    def test_appends_rows_to_new_file(self, tmp_path):
        temp_csv = tmp_path / "temp" / "monitor.csv"
        temp_csv.parent.mkdir()
        temp_csv.write_text("r,l,t\n1.0,100,0.5\n2.0,200,1.0\n")

        main_csv = tmp_path / "main" / "monitor.csv"
        result = append_monitor_data(str(temp_csv), str(main_csv))
        assert result is True
        assert main_csv.exists()

        with open(main_csv) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows

    def test_header_only_written_once(self, tmp_path):
        main_csv = tmp_path / "main" / "monitor.csv"
        main_csv.parent.mkdir()
        main_csv.write_text("r,l,t\n0.5,50,0.1\n")

        temp_csv = tmp_path / "temp" / "monitor.csv"
        temp_csv.parent.mkdir()
        temp_csv.write_text("r,l,t\n1.0,100,0.5\n")

        append_monitor_data(str(temp_csv), str(main_csv))

        with open(main_csv) as f:
            content = f.read()
        # Only one header line
        assert content.count("r,l,t") == 1

    def test_missing_temp_file_returns_false(self, tmp_path):
        result = append_monitor_data("/nonexistent/file.csv", str(tmp_path / "out.csv"))
        assert result is False
