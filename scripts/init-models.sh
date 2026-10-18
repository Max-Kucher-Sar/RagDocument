#!/bin/sh
set -eu

for model in "$CHAT_MODEL" "$EMBEDDING_MODEL"; do
    if ollama show "$model" >/dev/null 2>&1; then
        printf 'Model is already available: %s\n' "$model"
    else
        printf 'Downloading model: %s\n' "$model"
        ollama pull "$model"
    fi
done
printf 'Both models are ready.\n'
