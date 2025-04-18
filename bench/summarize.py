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
    if len(sys.argv) != 4:
        print("Usage: python summarize.py <quality_results.[json|dir]> <perf_results.json> <duration_str>", file=sys.stderr)
        sys.exit(1)

    quality_arg = sys.argv[1] 
    perf_file = sys.argv[2]
    duration_str = sys.argv[3]

    # If the first arg is a dir, read aggregated.json inside it
    if os.path.isdir(quality_arg):
        quality_file = os.path.join(quality_arg, "aggregated.json")
        model_name = os.path.basename(quality_arg) 
    else:
        quality_file = quality_arg 
        model_name = os.path.splitext(os.path.basename(quality_file))[0]

    # Handle potential 'perf_' prefix if quality derived from perf file accidentally
    if model_name.startswith("perf_"): 
         model_name = model_name.replace('perf_', '')

    quality_data = load_json(quality_file)
    perf_data = load_json(perf_file)

    if quality_data is None or perf_data is None:
        print(f"[{model_name}] Could not generate summary due to errors loading results.", file=sys.stderr)
        sys.exit(1) 

    # --- Extract Quality Scores ---
    quality_scores = {}
    if 'results' in quality_data:
        for task in TASKS:
            if task in quality_data['results']:
                try:
                    entry = quality_data["results"][task]
                    score = (entry.get("acc,norm") or 
                             entry.get("acc_norm") or 
                             entry.get("acc")      or
                             next(iter([v for v in entry.values() if isinstance(v, (int,float))]), None))

                    quality_scores[task] = score
                except KeyError:
                    quality_scores[task] = None
            else:
                quality_scores[task] = None 
    else:
         print(f"Warning: 'results' key not found in {quality_file}", file=sys.stderr)
         for task in TASKS:
             quality_scores[task] = None


    # --- Extract Performance Metrics ---
    mean_tps = perf_data.get("mean_tps", None)
    median_latency = perf_data.get("median_latency_s", None)

    # --- Generate Summary Table --- #
    summary_header = [f"Metric ({model_name})", "Value"]
    table_data = [
        ["Model", model_name], # Added explicit model name row
        ["-" * 10, "-" * 10], # Separator
        ["Quality Scores", ""],
    ]
    # Add formatted quality scores
    for task in TASKS:
        score = quality_scores.get(task, None)
        threshold = THRESHOLDS.get(task, 0)
        table_data.append([f"  {format_quality(task, score, threshold)}", ""])

    table_data.append(["-" * 10, "-" * 10]) # Separator
    table_data.append(["Performance", ""])
    # Add performance metrics
    table_data.append([f"  Median Latency", f"{perf_data.get('median_latency_s', 'N/A'):.3f} s" if isinstance(perf_data.get('median_latency_s'), (int, float)) else "N/A"])
    table_data.append([f"  Mean Throughput", f"{perf_data.get('mean_tps', 'N/A'):.2f} tok/s" if isinstance(perf_data.get('mean_tps'), (int, float)) else "N/A"])
    table_data.append(["  Total Tokens", f"{perf_data.get('total_tokens_generated', 'N/A')}"])

    table_data.append(["-" * 10, "-" * 10]) # Separator
    table_data.append(["Total Duration", f"{duration_str} (DD:HH:MM:SS)"])

    print("\n--- Benchmark Summary ---")
    print(tabulate(table_data, headers=summary_header, tablefmt="grid"))


if __name__ == "__main__":
    main()
