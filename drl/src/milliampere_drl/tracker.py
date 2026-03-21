"""Evaluation progress tracking and CSV data aggregation."""

from __future__ import annotations

import csv
import json
import os
import time
import traceback


class EvaluationTracker:
    """Persistent JSON-based tracker for resumable model evaluation."""

    def __init__(self, eval_dir: str) -> None:
        self.tracker_file = os.path.join(eval_dir, "evaluation_progress.json")
        self.progress = self._load_progress()
        self._print_current_progress()

    def _print_current_progress(self) -> None:
        print(f"Current tracker state: {self.progress}")

    def _load_progress(self) -> dict:
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file) as f:
                    return json.load(f)
        except Exception as e:
            print(
                f"Error loading progress file: {e}\n"
                "Starting with fresh progress tracking"
            )
        return {"completed_models": [], "current_model": None, "completed_episodes": 0}

    def save_progress(self) -> None:
        time.sleep(0.1)
        try:
            os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
            with open(self.tracker_file, "w") as f:
                json.dump(self.progress, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            current = self.progress["current_model"]
            name = os.path.basename(current) if current else "None"
            print(
                f"Progress saved: model={name}, "
                f"episodes={self.progress['completed_episodes']}"
            )
        except Exception as e:
            print(f"Error saving progress: {e}")
            traceback.print_exc()

    def is_model_completed(self, model_path: str) -> bool:
        return model_path in self.progress["completed_models"]

    def get_current_model(self) -> str | None:
        return self.progress["current_model"]

    def get_completed_episodes(self) -> int:
        return self.progress["completed_episodes"]

    def start_model_evaluation(self, model_path: str) -> None:
        if self.progress["current_model"] != model_path:
            print(f"Tracker: Starting evaluation for {os.path.basename(model_path)}")
            self.progress["current_model"] = model_path
            self.progress["completed_episodes"] = 0
            self.save_progress()
        else:
            print(f"Tracker: Resuming evaluation for {os.path.basename(model_path)}")

    def complete_episode(self) -> None:
        self.progress["completed_episodes"] += 1
        self.save_progress()

    def complete_model_evaluation(self) -> None:
        current_model = self.progress["current_model"]
        if current_model:
            print(
                f"Tracker: Marking model {os.path.basename(current_model)} as complete."
            )
            if current_model not in self.progress["completed_models"]:
                self.progress["completed_models"].append(current_model)
            self.progress["current_model"] = None
            self.progress["completed_episodes"] = 0
            self.save_progress()

    def get_all_completed_models(self) -> list[str]:
        return self.progress["completed_models"]


def append_monitor_data(temp_csv_path: str, main_csv_path: str) -> bool:
    """Append rows from a temporary monitor CSV into the main CSV file."""
    print(f"Attempting to append data from {temp_csv_path} to {main_csv_path}")
    if not os.path.exists(temp_csv_path):
        print(f"Temporary monitor file {temp_csv_path} not found.")
        return False

    temp_data_rows: list = []
    header = None
    try:
        with open(temp_csv_path) as temp_f:
            first_line = temp_f.readline()
            if first_line.startswith("#"):
                header_line = temp_f.readline().strip()
            else:
                header_line = first_line.strip()
            if header_line:
                header = header_line.split(",")
                reader = csv.reader(temp_f)
                temp_data_rows = list(reader)
            else:
                print(f"Warning: Temp file {temp_csv_path} empty/header missing.")
    except Exception as e:
        print(f"Error reading temp file {temp_csv_path}: {e}")
        traceback.print_exc()
        return False

    if not temp_data_rows:
        print(f"No data rows found in {temp_csv_path}. Cleaning up.")
        try:
            os.remove(temp_csv_path)
            temp_dir = os.path.dirname(temp_csv_path)
            os.rmdir(temp_dir)
            print(f"Removed empty temp file/dir: {temp_dir}")
        except OSError as e:
            print(f"Error removing temp file/dir {temp_csv_path}: {e}")
        return True

    main_dir = os.path.dirname(main_csv_path)
    os.makedirs(main_dir, exist_ok=True)
    needs_header = True
    if os.path.exists(main_csv_path):
        try:
            if os.path.getsize(main_csv_path) > 0:
                needs_header = False
        except OSError as e:
            print(
                f"Warning: size check failed {main_csv_path}: {e}. "
                "Assuming header needed."
            )

    try:
        with open(main_csv_path, "a", newline="") as main_f:
            writer = csv.writer(main_f)
            if needs_header and header:
                print(f"Writing header to {main_csv_path}")
                writer.writerow(header)
            print(f"Appending {len(temp_data_rows)} data rows to {main_csv_path}")
            writer.writerows(temp_data_rows)
    except Exception as e:
        print(f"Error writing main file {main_csv_path}: {e}")
        traceback.print_exc()
        return False

    try:
        print(f"Removing temporary monitor file: {temp_csv_path}")
        os.remove(temp_csv_path)
        temp_dir = os.path.dirname(temp_csv_path)
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"Removed empty temp dir: {temp_dir}")
    except OSError as e:
        print(f"Error cleaning up temp {temp_csv_path}: {e}")

    return True
