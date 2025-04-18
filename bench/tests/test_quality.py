# tests/test_quality.py
import pytest
import json
import os
import glob

# --- Configuration ---
# Define quality thresholds
THRESHOLDS = {
    "hellaswag": 0.80,
    "arc_easy": 0.75,
    "boolq": 0.80
}
RESULTS_DIR = "results"
# List of models to test (can be inferred from results files)
# MODELS = ["qwen2_5_32b", "deepseek-v3-54b", "qwen2_5_72b"] # Or infer dynamically
# --- End Configuration ---

# Check if tests should be skipped
SKIP_TESTS = os.environ.get("SKIP_TESTS", "false").lower() == "true"

def load_results(model_name):
    """Loads the quality results JSON for a given model."""
    filepath = os.path.join(RESULTS_DIR, f"{model_name}.json")
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Results file not found: {filepath}")
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from: {filepath}")
    except Exception as e:
        pytest.fail(f"Error loading {filepath}: {e}")

def get_model_names():
    """Dynamically finds model result files in the results directory."""
    if not os.path.exists(RESULTS_DIR):
        return []
    # Find files matching *.json but exclude perf_*.json
    pattern = os.path.join(RESULTS_DIR, "*.json")
    all_files = glob.glob(pattern)
    model_files = [f for f in all_files if not os.path.basename(f).startswith("perf_")]
    # Extract model names from filenames
    model_names = [os.path.basename(f).replace('.json', '') for f in model_files]
    return model_names

# Dynamically generate test parameters based on found result files
MODEL_NAMES = get_model_names()

@pytest.mark.skipif(SKIP_TESTS, reason="SKIP_TESTS environment variable is set")
@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("task, threshold", list(THRESHOLDS.items()))
def test_quality_thresholds(model_name, task, threshold):
    """Tests if a specific task score for a model meets the threshold."""
    results_data = load_results(model_name)

    assert 'results' in results_data, f"'results' key missing in {model_name}.json"
    assert task in results_data['results'], f"Task '{task}' missing in results for model {model_name}"

    task_result = results_data['results'][task]
    # Look for 'acc,norm' first, then 'acc'
    score = task_result.get('acc,norm', task_result.get('acc', None))

    assert score is not None, f"Accuracy score ('acc' or 'acc,norm') missing for task '{task}' in model {model_name}"
    assert score >= threshold, f"Model '{model_name}' failed task '{task}': Score {score:.3f} < Threshold {threshold:.3f}"

# Optional: Add a test to ensure all expected models were processed if needed
@pytest.mark.skipif(SKIP_TESTS or not MODEL_NAMES, reason="SKIP_TESTS set or no results found")
def test_all_models_present():
    """Checks if results for expected models are present (if using a fixed list)."""
    # If you have a predefined list of models you *expect* to run,
    # you could compare MODEL_NAMES against that list here.
    # For now, we just ensure *some* models were found if tests aren't skipped.
    print(f"Found results for models: {', '.join(MODEL_NAMES)}")
    pass # Simple check that the parametrization worked
