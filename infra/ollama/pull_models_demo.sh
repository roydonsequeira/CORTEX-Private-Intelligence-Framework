#!/bin/bash
# Pull the tiny models used by the low-resource demo stack (CPU-friendly).
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
models=("llama3.2:1b" "nomic-embed-text")

for model in "${models[@]}"; do
    echo "Pulling $model..."
    curl -s -X POST "${OLLAMA_URL}/api/pull" \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$model\"}" | jq '.status' -r
    echo "Done: $model"
done

echo "Demo models pulled."
