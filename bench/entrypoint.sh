#!/bin/bash
set -e

# Record start time
SCRIPT_START_TIME=$SECONDS

# Base directory for all results
BASE_RESULTS_DIR="/app/results"
mkdir -p "$BASE_RESULTS_DIR"

# Static identifier for the single model being tested
MODEL_IDENTIFIER="vllm_model"

# Default: No pytest run
RUN_TESTS=false

# Parse command-line arguments (only --run-tests relevant now)
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-tests)
      RUN_TESTS=true
      shift
      ;;
    *)
      # Allow unknown arguments to pass through
      echo "Ignoring unknown option: $1"
      shift
      ;;
  esac
done

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "===== Running benchmarks for model served by vLLM ====="

# Define paths for output directory and performance file
QUALITY_DIR="$BASE_RESULTS_DIR/${MODEL_IDENTIFIER}_${TIMESTAMP}"
PERF_OUTPUT_FILE="$BASE_RESULTS_DIR/perf_${MODEL_IDENTIFIER}_${TIMESTAMP}.json"

# Compose model args for local-completions driver
# Using multi-line definition as requested
MODEL_ARGS="completions_url=http://192.168.1.35:8002/v1/completions,tokenizer=/models,model=/models,num_concurrent=1"

# Print the arguments being passed
echo "Using MODEL_ARGS: $MODEL_ARGS"

# 1. Run lm-evaluation-harness (Quality)
echo "--- Running lm_eval (Quality) ---"
lm_eval --model local-completions \
        --model_args "$MODEL_ARGS" \
        --tasks hellaswag,arc_easy,boolq \
        --num_fewshot=0 \
        --seed=42 \
        --output_path "$QUALITY_DIR" \
        --log_samples \
        --trust_remote_code

# Check if quality results directory was created
if [ ! -d "$QUALITY_DIR" ]; then
    echo "ERROR: Quality results directory '$QUALITY_DIR' not found. Exiting."
    exit 1
fi
echo "Quality results saved to: $QUALITY_DIR"

# 2. Run bench.py (Performance)
echo "--- Running bench.py (Performance) ---"
# export PERF_OUTPUT_FILE # No longer needed, passing as arg
python /app/bench.py "$PERF_OUTPUT_FILE" # Pass output file path as argument

# Check if performance results file was created
if [ ! -f "$PERF_OUTPUT_FILE" ]; then
    echo "ERROR: Performance results file '$PERF_OUTPUT_FILE' not found. Exiting."
    exit 1
fi
echo "Performance results saved to: $PERF_OUTPUT_FILE"

# ---- Generate Summary (including duration) --------------------------
echo "--- Generating Summary ---"

# Compute elapsed time SO FAR
ELAPSED_SEC=$(( SECONDS - SCRIPT_START_TIME ))

# Format as DD:HH:MM:SS
printf -v DURATION_STR '%02d:%02d:%02d:%02d' \
        $((ELAPSED_SEC/86400)) \
        $((ELAPSED_SEC%86400/3600)) \
        $((ELAPSED_SEC%3600/60)) \
        $((ELAPSED_SEC%60))

# Pass quality dir, perf file, and formatted duration string
python /app/summarize.py "$QUALITY_DIR" "$PERF_OUTPUT_FILE" "$DURATION_STR"
echo "Summary generated (Duration so far: $DURATION_STR)."
# ----------------------------------------------------------------------

# 4. Optional: Run Pytest Quality Gates
if [ "$RUN_TESTS" = true ]; then
  echo ""
  echo "--- Running pytest Quality Gates ---"
  pytest /app/tests -v
  echo "Quality Gates: Check pytest output above."
else
  echo "Skipping pytest Quality Gates (--run-tests not specified)."
fi

echo "--------------------------------------------------"
echo "Benchmark run complete."
echo "Results are in: $BASE_RESULTS_DIR"
