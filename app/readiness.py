"""Dependency checks shared by the container entry point and its health check."""
import httpx

from app.config import Settings


def check_dependencies(settings: Settings) -> bool:
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            ready = client.get(settings.qdrant_url.rstrip('/') + '/readyz')
            ready.raise_for_status()
            tags = client.get(settings.ollama_base_url.rstrip('/') + '/api/tags')
            tags.raise_for_status()
            names = {entry['name'] for entry in tags.json()['models']}
            return all(name in names or name + ':latest' in names
                       for name in [settings.chat_model, settings.embedding_model])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return False
