# ADR-002: Why ChromaDB Over Qdrant or Weaviate

## Context

CORTEX needs semantic and procedural memory that works locally without a separate managed service. The default experience should run from Docker Compose or an embedded local path and remain easy to inspect during development.

## Decision

CORTEX uses ChromaDB for vector memory. Semantic and procedural memory use local Chroma collections with Ollama embeddings.

## Consequences

ChromaDB keeps the local developer path simple and Python-native. It is sufficient for a reference implementation and does not require operating a heavier vector database. Qdrant or Weaviate may be better for large deployments, but they add operational weight that conflicts with CORTEX's default offline-first setup.
