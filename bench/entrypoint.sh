#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the models to benchmark
MODELS=(qwen2_5_32b deepseek-v3-54b qwen2_5_72b) # As requested

# Create results directory if it doesn't exist
RESULTS_DIR="/app/results"
mkdir -p "$RESULTS_DIR"

echo "Starting benchmark run..."
echo "Models to test: ${MODELS[@]}"
echo "Quality tests will be skipped if SKIP_TESTS=true (current: SKIP_TESTS=${SKIP_TESTS:-false})"
echo "--------------------------------------------------"

# Loop through each model
for model in "${MODELS[@]}"; do
  echo ""
  echo "===== Running benchmarks for: $model ====="

  # Define output file paths
  QUALITY_OUTPUT_FILE="$RESULTS_DIR/${model}.json"
  PERF_OUTPUT_FILE="$RESULTS_DIR/perf_${model}.json"

  # 1. Run lm-evaluation-harness (Quality)
  echo "--- Running lm_eval (Quality) ---"
  export ENGINE=$model # Set environment variable for lm_eval config and bench.py
  lm_eval --model=hf-chat \
          --model_args "engine=hf-chat,base_url=http://localhost:8000/v1,model=$model,max_gen_toks=64" \
          --tasks=hellaswag,arc_easy,boolq \
          --device="cpu" \
          --num_fewshot=0 \
          --seed=42 \
          --output_path "$QUALITY_OUTPUT_FILE" \
          --log_samples # Optional: remove if too verbose

  # Check if quality results file was created
  if [ ! -f "$QUALITY_OUTPUT_FILE" ]; then
      echo "ERROR: Quality results file '$QUALITY_OUTPUT_FILE' not found for model '$model'. Skipping further steps for this model."
      continue # Skip to the next model
  fi

  # 2. Run bench.py (Performance)
  echo "--- Running bench.py (Performance) ---"
  # Ensure ENGINE is still set for bench.py
  python /app/bench.py # Output is handled internally by the script

   # Check if performance results file was created
  if [ ! -f "$PERF_OUTPUT_FILE" ]; then
      echo "ERROR: Performance results file '$PERF_OUTPUT_FILE' not found for model '$model'. Skipping summary for this model."
      # We might still want to proceed to the next model or tests
      # depending on requirements. For now, let's just warn and continue summary attempt.
      # It will fail gracefully in summarize.py if the file is missing.
  fi

  # 3. Run summarize.py (Summary)
  echo "--- Generating Summary ---"
  # Check if both files exist before summarizing
  if [ -f "$QUALITY_OUTPUT_FILE" ] && [ -f "$PERF_OUTPUT_FILE" ]; then
      python /app/summarize.py "$QUALITY_OUTPUT_FILE" "$PERF_OUTPUT_FILE"
  else
      echo "Skipping summary for $model due to missing result files."
  fi


  echo "===== Finished benchmarks for: $model ====="
done

echo "--------------------------------------------------"
echo "Benchmark run complete for all models."
echo ""

# 4. Run pytest (Quality Gates)
if [ -z "$SKIP_TESTS" ] || [ "$SKIP_TESTS" = "false" ]; then
  echo "--- Running pytest Quality Gates ---"
  # Ensure pytest runs from the /app directory where results/ exists
  cd /app
  # Update tests dynamically before running pytest
  # This ensures test_quality.py picks up the latest results files
  # (This is handled internally by test_quality.py's get_model_names())

  # Run pytest, exit code will reflect test success/failure
  pytest -q tests/
  PYTEST_EXIT_CODE=$?
  if [ $PYTEST_EXIT_CODE -eq 0 ]; then
      echo "Quality Gates: PASSED"
  else
      echo "Quality Gates: FAILED (exit code $PYTEST_EXIT_CODE)"
  fi
  # Exit with the pytest status code
  exit $PYTEST_EXIT_CODE
else
  echo "--- Skipping pytest Quality Gates (SKIP_TESTS is set) ---"
  # Exit with 0 if tests are skipped
  exit 0
fi
