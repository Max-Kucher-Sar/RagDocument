"""Wait for dependencies, then replace this process with the application server."""
import os
import time

from app.config import Settings
from app.readiness import check_dependencies


def wait_until_ready(settings, *, timeout=180, interval=3):
    deadline = time.monotonic() + timeout
    while True:
        if check_dependencies(settings):
            print('Ollama models and Qdrant are ready.', flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Dependencies are not ready. Check logs: docker-compose logs ollama-init ollama qdrant')
        print('Waiting for Ollama models and Qdrant...', flush=True)
        time.sleep(interval)


def main():
    wait_until_ready(Settings())
    os.execvp('uvicorn', ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'])


if __name__ == '__main__':
    main()
