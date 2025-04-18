import requests
import time
import json
import os
import statistics
import argparse

# Configuration
VLLM_URL = "http://192.168.1.35:8002/v1/completions" # Use completions endpoint for openai-completions
NUM_REQUESTS = 5 # Send multiple requests for more stable metrics
PROMPT = "Explain the concept of Large Language Models in one sentence."

def run_benchmark(output_file_path):
    """Runs performance benchmark against the vLLM server.

    Args:
        output_file_path (str): The full path where the JSON results should be saved.
    """
    latencies = []
    total_tokens = 0
    start_times = []
    end_times = []

    print(f"Starting performance benchmark...")
    print(f"Saving results to: {output_file_path}")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    headers = {"Content-Type": "application/json"}
    data = {
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

            latency = end_time - start_time
            latencies.append(latency)

            response_data = response.json()
            # Calculate generated tokens based on usage info if available
            generated_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
            if generated_tokens == 0 and response_data.get('choices'):
                 # Fallback: crude estimate if usage not provided (less accurate)
                 generated_tokens = len(response_data['choices'][0]['text'].split())

            total_tokens += generated_tokens
            print(f"Request {i+1}/{NUM_REQUESTS}: Latency={latency:.3f}s, Tokens={generated_tokens}")
            time.sleep(0.5) # Small delay between requests

        except requests.exceptions.RequestException as e:
            print(f"Error during request {i+1}: {e}")
            # Decide how to handle errors: skip, retry, or fail? Let's skip for now.
            continue
        except Exception as e:
            print(f"An unexpected error occurred during request {i+1}: {e}")
            continue

    if not latencies:
        print("No successful requests were made. Cannot calculate performance metrics.")
        results = {
            "error": "No successful requests completed."
        }
    else:
        median_latency = statistics.median(latencies)
        # Calculate overall throughput based on total time and total tokens
        overall_time = end_times[-1] - start_times[0] if len(start_times)>0 and len(end_times)>0 else sum(latencies)
        mean_tps = total_tokens / overall_time if overall_time > 0 else 0

        results = {
            "median_latency_s": round(median_latency, 3),
            "mean_tps": round(mean_tps, 2),
            "total_requests": NUM_REQUESTS,
            "successful_requests": len(latencies),
            "total_tokens_generated": total_tokens,
            "total_time_s": round(overall_time, 3)
        }
        print(f"Benchmark complete.")
        print(f"  Median Latency: {results['median_latency_s']:.3f}s")
        print(f"  Mean Throughput: {results['mean_tps']:.2f} tok/s")

    # Write results to JSON file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Performance results saved to {output_file_path}")
    except IOError as e:
        print(f"Error writing performance results to {output_file_path}: {e}")

# --- Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run vLLM performance benchmark.')
    parser.add_argument('output_file', type=str, help='Path to save the performance results JSON file.')
    args = parser.parse_args()

    run_benchmark(args.output_file)
