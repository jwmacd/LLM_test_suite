# Dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
# Install git first as lm-eval harness might need it
RUN apt-get update && apt-get install -y git --no-install-recommends && \
    pip install --no-cache-dir \
        "lm-eval[huggingface] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git" \
        pytest \
        requests \
        tabulate && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/*

# Copy necessary files from the build context
# Copy the scripts first
COPY bench.py .
COPY summarize.py .
COPY entrypoint.sh .
# Copy configuration and tests
COPY quick_suite.yaml .
COPY tests/ ./tests/

# Make the entrypoint script executable (within the container build process)
RUN chmod +x /app/entrypoint.sh

# Create the results directory and set permissions if needed
# (The entrypoint script also creates it, but doing it here ensures ownership)
RUN mkdir /app/results && chown nobody:nogroup /app/results
# Note: Adjust user/group if running as non-root is required later

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Optional: Specify a default command if needed (usually not needed with ENTRYPOINT script)
# CMD [""]
