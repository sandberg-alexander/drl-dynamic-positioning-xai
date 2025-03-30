#!/bin/bash

# Get the directory of the script itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Configuration ---
# Find the latest training run directory automatically, or set manually
# LATEST_RUN_DIR=$(ls -td /app/models/training_* | head -n 1)
# OR specify directly:
RUN_DIR="/app/models/training_20250328_145357" # <<< CHANGE THIS IF NEEDED
NUM_EPISODES=5                             # <<< SET EPISODES HERE

# Construct the full path to the python script
PYTHON_SCRIPT="${SCRIPT_DIR}/eval.py"

# --- Argument Check ---
if [ -z "$RUN_DIR" ]; then
  echo "Error: RUN_DIR is not set or could not be found automatically."
  exit 1
elif [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory '$RUN_DIR' does not exist."
  exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

echo "Starting evaluation loop..."
echo "Run Directory: $RUN_DIR"
echo "Episodes per model: $NUM_EPISODES"
echo "-------------------------------------"

# Trap SIGINT (Ctrl+C)
trap "echo; echo '>>> Ctrl+C detected in shell script. Stopping evaluation loop. <<<'; exit 1" SIGINT

# --- Evaluation Loop ---
while true; do
    echo "*** Running Python script for next model... ***"
    # Execute the python script
    python3 "$PYTHON_SCRIPT" --dir "$RUN_DIR" --episodes "$NUM_EPISODES"

    # Check the exit code of the python script
    exit_code=$?
    echo "*** Python script finished with exit code: $exit_code ***"

    if [ $exit_code -eq 0 ]; then
        # Exit code 0 means:
        # - Successfully evaluated a model to completion OR
        # - Skipped an already completed model OR
        # - No more models left to evaluate.
        # We need to check the tracker file to see if we are truly done.

        eval_progress_file="${RUN_DIR}/evaluation/evaluation_progress.json"
        if [ ! -f "$eval_progress_file" ]; then
             echo "Evaluation progress file not found, assuming completion or error."
             break
        fi

        # Check if 'current_model' is null and 'completed_models' list contains all potential models (approx check)
        # A more robust check might involve parsing the JSON properly
        current_model=$(grep '"current_model":' "$eval_progress_file" | grep -o 'null')
        # Simplistic check: if current_model is null, maybe we are done?
        if [ "$current_model" == "null" ]; then
             echo "Tracker indicates no current model. Checking if any models remain..."
             # Rerun script one last time; if it exits 0 again, assume done.
             python3 "$PYTHON_SCRIPT" --dir "$RUN_DIR" --episodes "$NUM_EPISODES"
             last_run_exit_code=$?
             if [ $last_run_exit_code -eq 0 ]; then
                 echo "Python script confirms no more models. Evaluation complete."
                 break
             else
                 echo "Python script indicated more work (exit code $last_run_exit_code) despite null current_model? Continuing loop."
             fi
        fi
        echo "Continuing to next model..."

    elif [ $exit_code -eq 1 ]; then
        # Exit code 1 means:
        # - Evaluation for the model was interrupted (Ctrl+C inside Python) OR
        # - Evaluation failed mid-way for that model (but not critically)
        echo "Python script indicated model evaluation incomplete or interrupted. Continuing loop for resume..."
        # Loop will automatically pick up the same model again based on tracker
    else
        # Exit code 2 (or others) indicates a more critical error
        echo "Python script exited with critical error code ($exit_code). Stopping."
        break
    fi

    echo "-------------------------------------"
    sleep 1 # Small delay before next iteration
done

echo "Evaluation loop finished."
exit 0