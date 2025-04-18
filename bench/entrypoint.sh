#!/bin/bash
set -e

# -----------------------------------------
# Path where all JSONs will go
BASE_RESULTS_DIR="/models/results"      # <= inside the mapped volume
mkdir -p "$BASE_RESULTS_DIR"
# -----------------------------------------

# Record start time
SCRIPT_START_TIME=$SECONDS

# Static identifier for the single model being tested
MODEL_IDENTIFIER="vllm_model"

# Default: No pytest run
RUN_TESTS=false

# --- parse CLI flags -------------------------------------------------
FAST_MODE=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST_MODE=1 ;;
    --run-tests) RUN_TESTS=true ;;
    *)
      # Allow unknown arguments to pass through
      echo "Ignoring unknown option: $arg"
      ;;
  esac
done

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "===== Running benchmarks for model served by vLLM ====="

# --- Setup Output Paths ---
# NOTE: Use fixed paths inside the container
# Results are mounted to the host via docker volume
QUALITY_DIR="/models/results/${TIMESTAMP}"
PERF_FILE="${QUALITY_DIR}/perf_results.json"

# Ensure the results directory exists
mkdir -p "$QUALITY_DIR"

echo "Results directory: $QUALITY_DIR"

# Compose model args for local-completions driver
# Using multi-line definition as requested
# Working args: MODEL_ARGS="base_url=http://192.168.1.35:8002/v1/completions,model=/models/,num_concurrent=1,trust_remote_code=True"
MODEL_ARGS="base_url=http://192.168.1.35:8002/v1/completions,model=/models/,num_concurrent=1"

# Print the arguments being passed
echo "Using MODEL_ARGS: $MODEL_ARGS"

# --- build optional flag --------------------------------------------
LIMIT_FLAG=()                # << array (empty by default)
if [[ $FAST_MODE == 1 ]]; then
  echo "INFO: --fast detected; limiting each task to 25 samples."
  LIMIT_FLAG=(--limit 25)
fi

# 1. Run lm-evaluation-harness (Quality)
echo "--- Running lm_eval (Quality) ---"

lm_eval --model local-completions \
        --model_args "$MODEL_ARGS" \
        --tasks hellaswag,arc_easy,boolq,openbookqa,winogrande,piqa,commonsense_qa,truthfulqa_mc1,truthfulqa_mc2,humaneval,mbpp \
        --num_fewshot=0 \
        --seed=42 \
        --output_path "$QUALITY_DIR" \
        --log_samples \
        --trust_remote_code \
        "${LIMIT_FLAG[@]}"

# ---------- find the results JSON (recursively) ----------
RESULT_JSON=$(find "$QUALITY_DIR" -type f -name 'results_*.json' | head -n1)
if [[ -z "$RESULT_JSON" ]]; then
  echo "ERROR: lm_eval did not create a results JSON." ; exit 1
fi
cp "$RESULT_JSON" "$QUALITY_DIR/aggregated.json"
echo "Quality results saved to: $QUALITY_DIR/aggregated.json"
# --------------------------------------------------------

# Check if quality results directory was created
if [ ! -d "$QUALITY_DIR" ]; then
    echo "ERROR: Quality results directory '$QUALITY_DIR' not found. Exiting."
    exit 1
fi

# 2. Run bench.py (Performance)
echo "--- Running bench.py (Performance) ---"
export MODEL_ARGS              # ← ensure MODEL_ARGS is exported for bench.py
python /app/bench.py "$PERF_FILE"

# Check if performance results file was created
if [ ! -f "$PERF_FILE" ]; then
    echo "ERROR: Performance results file '$PERF_FILE' not found. Exiting."
    exit 1
fi
echo "Performance results saved to: $PERF_FILE"

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
python /app/summarize.py "$QUALITY_DIR/aggregated.json" "$PERF_FILE" "$DURATION_STR"
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
