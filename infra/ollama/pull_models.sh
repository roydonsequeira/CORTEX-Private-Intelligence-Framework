#!/usr/bin/env bash
# Pull the models CORTEX uses by default into Ollama (native or the compose stack).
set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
models=("qwen2.5:7b" "nomic-embed-text")

for model in "${models[@]}"; do
    echo "Pulling $model (this can take a while)..."
    # stream=false returns one JSON object when the pull finishes; -f fails on HTTP errors.
    curl -sf -X POST "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\", \"stream\": false}" > /dev/null
    echo "Done: $model"
done

echo "All models pulled."
