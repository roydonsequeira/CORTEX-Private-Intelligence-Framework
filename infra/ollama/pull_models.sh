#!/bin/bash
# Pull all required models into the Ollama container
set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
models=("llama3.1:8b" "nomic-embed-text" "deepseek-r1:8b")

for model in "${models[@]}"; do
    echo "Pulling $model..."
    curl -s -X POST "${OLLAMA_URL}/api/pull" \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$model\"}" | jq '.status' -r
    echo "Done: $model"
done

echo "All models pulled."
