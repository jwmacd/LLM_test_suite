import json
import sys
import os
from tabulate import tabulate

# --- Configuration ---
# Define quality thresholds (same as in test_quality.py)
THRESHOLDS = {
    "hellaswag": 0.80,
    "arc_easy": 0.75,
    "boolq": 0.80
}
TASKS = ["hellaswag", "arc_easy", "boolq"]
# --- End Configuration ---

def load_json(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from - {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None

def format_quality(task, score, threshold):
    """Formats the quality score with a pass/fail indicator."""
    if score is None:
        return f"{task} -- ✗" # Indicate missing score
    passed = score >= threshold
    marker = "✓" if passed else "✗"
    return f"{task} {score:.2f} {marker}"

def main():
    if len(sys.argv) != 3:
        print("Usage: python summarize.py <quality_results.json> <perf_results.json>", file=sys.stderr)
        sys.exit(1)

    quality_file = sys.argv[1]
    perf_file = sys.argv[2]

    # Extract model name from filename (assuming results/<model>.json format)
    model_name = os.path.basename(quality_file).replace('.json', '')
    if model_name.startswith("perf_"): # Handle perf file name possibility
         model_name = model_name.replace('perf_', '')


    quality_data = load_json(quality_file)
    perf_data = load_json(perf_file)

    if quality_data is None or perf_data is None:
        print(f"[{model_name}] Could not generate summary due to errors loading results.", file=sys.stderr)
        sys.exit(1) # Exit if results are missing/invalid

    # --- Extract Quality Scores ---
    quality_scores = {}
    if 'results' in quality_data:
        for task in TASKS:
            if task in quality_data['results']:
                # Look for 'acc,norm' or 'acc' for accuracy
                score = quality_data['results'][task].get('acc,norm', quality_data['results'][task].get('acc', None))
                quality_scores[task] = score
            else:
                quality_scores[task] = None # Mark as missing if task not in results
    else:
         print(f"Warning: 'results' key not found in {quality_file}", file=sys.stderr)
         for task in TASKS:
             quality_scores[task] = None


    # --- Extract Performance Metrics ---
    mean_tps = perf_data.get("mean_tps", None)
    median_latency = perf_data.get("median_latency_s", None)

    # --- Format Output ---
    quality_summary_parts = [
        format_quality(task, quality_scores.get(task), THRESHOLDS.get(task, 0)) for task in TASKS
    ]
    quality_summary = "  ".join(quality_summary_parts)

    perf_summary_parts = []
    if mean_tps is not None:
        perf_summary_parts.append(f"{mean_tps:.1f} tok/s")
    else:
        perf_summary_parts.append("-- tok/s")

    if median_latency is not None:
         perf_summary_parts.append(f"p50 {median_latency:.2f}s")
    else:
         perf_summary_parts.append("p50 --s")

    perf_summary = "  ".join(perf_summary_parts)

    # Print the combined summary line
    print(f"{model_name:<15} {quality_summary:<45} {perf_summary}")


if __name__ == "__main__":
    main()
