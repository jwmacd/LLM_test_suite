import json
import sys
import os
from tabulate import tabulate

# --- Configuration ---
TASKS = [
    "hellaswag", "arc_easy", "boolq",
    "openbookqa", "winogrande", "piqa", "commonsense_qa", "truthfulqa_mc1", "truthfulqa_mc2",
    "humaneval", "mbpp"
]
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

def format_score(task, score):
    """
    Return a plain string like 'hellaswag 0.64'.
    If the score is missing, show '--'.
    """
    return f"{task} {score:.2f}" if score is not None else f"{task} --"

def main():
    # Accept 2 or 3 positional args
    if len(sys.argv) not in (3, 4):
        print("Usage: python summarize.py <quality_results.json> <perf_results.json> [duration_str]", file=sys.stderr)
        sys.exit(1)

    quality_file = sys.argv[1]
    perf_file = sys.argv[2]
    duration_str = sys.argv[3] if len(sys.argv) == 4 else None

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

    # NEW: find where the "results" live  ────────────────────────────────
    qdata = quality_data
    if "results" not in qdata:
        # many eval runs wrap everything under a single model‑name key
        for v in qdata.values():
            if isinstance(v, dict) and "results" in v:
                qdata = v
                break
    # qdata now guaranteed to have "results" or we’ll warn below
    # --------------------------------------------------------------------

    if "results" in qdata:
        for task in TASKS:
            entry = qdata["results"].get(task, {}) # Use qdata
            # Prioritize keys: specific acc -> task-specific -> generic acc
            score = next((s for s in [
                entry.get("acc,none"),       # Standard acc (0-shot)
                entry.get("acc_norm,none"),  # Standard normalized acc (0-shot)
                entry.get("mc2"),            # Specific: truthfulqa_mc
                entry.get("acc@1"),          # Specific: apps
                entry.get("pass@1"),         # Specific: humaneval, mbpp
                entry.get("acc,norm"),       # Fallback normalized (e.g., hellaswag few-shot?)
                entry.get("acc_norm"),       # Fallback standard normalized
                entry.get("acc")             # Fallback standard accuracy
            ] if s is not None), None)
            quality_scores[task] = score
    elif quality_data: # Only warn if quality_data was actually loaded
        print(f"Warning: 'results' key not found in {quality_file}", file=sys.stderr)
        quality_scores = {t: None for t in TASKS}

    # --- Extract Performance Metrics ---
    raw_tps = perf_data.get("raw_tps", None)
    effective_tps = perf_data.get("effective_tps", None)
    sleep_per_call_s = perf_data.get("sleep_per_call_s", None)
    median_latency = perf_data.get("median_latency_s", None)
    total_tokens = perf_data.get("total_tokens_generated", None)

    # --- Generate Summary Table --- #
    summary_header = [f"Metric ({model_name})", "Value"]
    table_data = [
        ["Model", model_name],
        ["-" * 10, "-" * 10],
        ["Quality Scores", ""],
    ]
    quality_summary_parts = [
        format_score(task, quality_scores.get(task))
        for task in TASKS
    ]
    for part in quality_summary_parts:
        table_data.append([part.split(" ")[0], " ".join(part.split(" ")[1:])])
    table_data.append(["-" * 10, "-" * 10])
    table_data.append(["Performance", ""])
    if raw_tps is not None:
        table_data.append(["  Mean TPS (raw)",      f"{raw_tps:.1f} tok/s"])
    if effective_tps is not None and sleep_per_call_s is not None:
         table_data.append(["  Mean TPS (overall)",  f"{effective_tps:.1f} tok/s (sleep {sleep_per_call_s} s)"])
    else:
         # Fallback if new keys aren't present (optional, for backward compatibility)
         if perf_data.get("mean_tps") is not None: # Check if the old key exists
            table_data.append(["  Mean TPS", f"{perf_data.get('mean_tps'):.1f} tok/s"])
    if median_latency is not None:
        table_data.append(["  Median Latency", f"p50 {median_latency:.2f}s"])
    if total_tokens is not None:
        table_data.append(["  Total Tokens", total_tokens])

    # -------- duration row ----------
    if duration_str:
        table_data.append(["-" * 10, "-" * 10])
        table_data.append(["Total Duration", f"{duration_str} (DD:HH:MM:SS)"])

    # Print the summary table to stdout
    print("\n--- Generating Summary ---")
    table_string_stdout = tabulate(table_data, headers=summary_header, tablefmt="pipe") # Use pipe for stdout
    print(table_string_stdout)

    # --- Save Summary to File ---
    try:
        output_dir = os.path.dirname(perf_file)
        summary_file_path = os.path.join(output_dir, "SUMMARY.txt")
        # Generate table string with grid format for the file
        table_string_file = tabulate(table_data, headers=summary_header, tablefmt="grid")
        with open(summary_file_path, "w") as fp:
            fp.write(table_string_file)
        print(f"Summary saved to {summary_file_path}")
    except Exception as e:
        print(f"Error saving summary file: {e}", file=sys.stderr)

    if duration_str:
        print(f"Summary generated (Duration so far: {duration_str}).")
    else:
        print("Summary generated.")

if __name__ == "__main__":
    main()
