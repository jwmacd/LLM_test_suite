import requests
import time
import json
import os
import statistics
import sys

# --- Configuration from MODEL_ARGS --- 
def parse_model_args(args_str):
    """Parses the MODEL_ARGS string into a dictionary."""
    if not args_str:
        return {}
    try:
        return dict(arg.split('=', 1) for arg in args_str.split(','))
    except ValueError:
        print(f"ERROR: Invalid format in MODEL_ARGS: '{args_str}'. Expected 'key1=value1,key2=value2,...'", file=sys.stderr)
        sys.exit(1)

# Get MODEL_ARGS from environment variable
model_args_str = os.environ.get("MODEL_ARGS")
if not model_args_str:
    print("ERROR: MODEL_ARGS environment variable not set.", file=sys.stderr)
    sys.exit(1)

model_args = parse_model_args(model_args_str)

# Extract required parameters
BASE_URL = model_args.get("base_url")
# Fallback to VLLM_BASE_URL environment variable if base_url not in MODEL_ARGS
if not BASE_URL:
    print("INFO: 'base_url' not found in MODEL_ARGS, checking VLLM_BASE_URL environment variable.")
    BASE_URL = os.environ.get("VLLM_BASE_URL")

MODEL_NAME = model_args.get("model") # Use 'model' key like lm-eval

if not BASE_URL:
    print("ERROR: Could not determine base URL. Set 'base_url' in MODEL_ARGS or set VLLM_BASE_URL environment variable.", file=sys.stderr)
    sys.exit(1)
if not MODEL_NAME:
    print("ERROR: 'model' not found in MODEL_ARGS.", file=sys.stderr)
    sys.exit(1)

# Construct the completions URL
VLLM_URL = BASE_URL.rstrip('/')

print(f"Using VLLM URL: {VLLM_URL}")
print(f"Using Model Name: {MODEL_NAME}")

# --- NEW: Timing variables ---
RAW_TOTAL_TIME   = 0.0      # excludes sleeps
EFFECTIVE_PAUSE  = 0.5      # the current sleep, keep it configurable
SLEEPS_USED      = 0        # counter
# -------------------------

# --- Output path ---
# If the caller provided a filename use it,
# otherwise fall back to the timestamp‑free default.
if len(sys.argv) > 1 and sys.argv[1]:
    OUTPUT_FILE = sys.argv[1]
else:
    OUTPUT_FILE = f"results/perf_{MODEL_NAME.replace('/', '_')}.json"

# --- Static Configuration ---
NUM_REQUESTS = 5 # Send multiple requests for more stable metrics
PROMPT = "Explain the concept of Large Language Models in one sentence."

def run_benchmark():
    """Runs performance benchmark against the vLLM server."""
    global RAW_TOTAL_TIME, SLEEPS_USED # Allow modification of globals
    latencies = []
    total_tokens = 0
    start_times = []
    end_times = []

    print(f"Starting performance benchmark for model: {MODEL_NAME}...")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": 100, # Limit generated tokens for perf test consistency
        "temperature": 0.1 # Low temp for deterministic output length (mostly)
    }

    for i in range(NUM_REQUESTS):
        try:
            start_time = time.perf_counter()
            start_times.append(start_time)
            response = requests.post(VLLM_URL, headers=headers, json=data, timeout=60) # Added timeout
            end_time = time.perf_counter()
            end_times.append(end_time)

            response.raise_for_status() # Raise exception for bad status codes

            # --- Timing update ---
            latency = end_time - start_time          # wall-time for this request
            RAW_TOTAL_TIME += latency                # accumulate *raw* time
            # -------------------
            latencies.append(latency)

            response_data = response.json()
            generated_tokens = 0
            # Prioritize usage stats for accurate token count (vLLM >= 0.4)
            if 'usage' in response_data and isinstance(response_data['usage'], dict) and \
               'total_tokens' in response_data['usage'] and 'prompt_tokens' in response_data['usage']:
                try:
                    generated_tokens = int(response_data['usage']['total_tokens']) - int(response_data['usage']['prompt_tokens'])
                except (TypeError, ValueError):
                    print("Warning: Could not parse token counts from 'usage' field.")
                    # Fall through to text-based estimation if parsing fails

            # Fallback if usage info is missing or incomplete
            if generated_tokens <= 0 and response_data.get('choices'):
                 try:
                     # Crude estimate if usage not provided (less accurate)
                     generated_tokens = len(response_data['choices'][0]['text'].split())
                     print("Warning: Estimating generated tokens from text length. Accuracy may vary.")
                 except (IndexError, KeyError):
                     print("Warning: Could not estimate tokens from response text.")

            total_tokens += generated_tokens
            print(f"Request {i+1}/{NUM_REQUESTS}: Latency={latency:.3f}s, Tokens={generated_tokens}")
            time.sleep(EFFECTIVE_PAUSE) # Small delay between requests
            SLEEPS_USED += 1

        except requests.exceptions.RequestException as e:
            print(f"Error during request {i+1}: {e}")
            # Decide how to handle errors: skip, retry, or fail? Let's skip for now.
            continue
        except Exception as e:
            print(f"An unexpected error occurred during request {i+1}: {e}")
            continue

    if not latencies:
        print("ERROR: No successful requests were made. Cannot calculate performance metrics.", file=sys.stderr)
        # Exit with error code 1 if all requests failed
        sys.exit(1)
    else:
        median_latency = statistics.median(latencies)
        # Calculate overall throughput based on total time and total tokens
        eff_total_time = RAW_TOTAL_TIME + SLEEPS_USED * EFFECTIVE_PAUSE
        raw_tps = total_tokens / RAW_TOTAL_TIME if RAW_TOTAL_TIME else 0
        eff_tps = total_tokens / eff_total_time if eff_total_time else 0

        results = {
            "model": MODEL_NAME,
            "median_latency_s": round(median_latency, 3),
            "raw_tps": round(raw_tps, 2),
            "effective_tps": round(eff_tps, 2),
            "sleep_per_call_s": EFFECTIVE_PAUSE,
            "total_requests": NUM_REQUESTS,
            "successful_requests": len(latencies),
            "total_tokens_generated": total_tokens,
            "total_time_s": round(eff_total_time, 3)
        }
        print(f"Benchmark complete for {MODEL_NAME}.")
        print(f"  Median Latency: {results['median_latency_s']:.3f}s")
        print(f"  Raw Throughput: {results['raw_tps']:.2f} tok/s")
        print(f"  Effective Throughput: {results['effective_tps']:.2f} tok/s")

    # Write results to JSON file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Performance results saved to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing performance results to {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    run_benchmark()
