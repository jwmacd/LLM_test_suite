# LLM Test Suite

A self-contained benchmark harness for evaluating LLM models using a vLLM server.

## Overview

This benchmark harness connects to a running vLLM server at http://localhost:8000/v1 (OpenAI API compatible) and evaluates models on both quality and performance metrics.

## Features

### Quality Testing
- Uses lm-evaluation-harness in "local-completions" mode
- Evaluates on standard benchmarks: hellaswag, arc_easy, and boolq
- Zero-shot evaluation with configurable parameters
- Outputs detailed JSON results

### Performance Testing
- Measures latency and throughput via API calls to the vLLM server
- Computes median latency and mean tokens-per-second
- Outputs performance metrics in JSON format

### Summary Reporting
- Generates concise console summaries for each model run
- Format: `<model> hellaswag xx ✓/✗ arc_easy xx ✓/✗ boolq xx ✓/✗ <mean TPS> tok/s p50 <latency>s`

### Quality Gates
- Optional threshold testing for model accuracy
- Configurable via environment variables

## Usage

```bash
# Run the benchmark suite
docker compose up --build

# Skip quality gate tests
SKIP_TESTS=true docker compose up --build
```

## Requirements

- Docker and Docker Compose
- A running vLLM server (http://localhost:8000/v1) with OpenAI-compatible API

## Directory Structure

- `bench/`: Contains all benchmark harness files
  - `Dockerfile`: Container definition
  - `docker-compose.yml`: Service configuration
  - `quick_suite.yaml`: lm-evaluation-harness configuration
  - `bench.py`: Performance testing script
  - `summarize.py`: Result parsing and reporting
  - `entrypoint.sh`: Main orchestration script
  - `tests/`: Quality gate test definitions
