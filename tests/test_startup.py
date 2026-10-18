import httpx
import pytest

from app import bootstrap, readiness
from app.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.mark.parametrize('db_status,models,expected', [
    (200, ['qwen3:4b-instruct', 'qwen3-embedding:0.6b'], True),
    (503, ['qwen3:4b-instruct', 'qwen3-embedding:0.6b'], False),
    (200, ['qwen3:4b-instruct'], False),
    (200, [], False),
])
def test_readiness_requires_database_and_both_models(monkeypatch, db_status, models, expected):
    settings = Settings(_env_file=None)
    requests = []
    def respond(request):
        requests.append(str(request.url))
        if request.url.host == 'qdrant':
            return httpx.Response(db_status, text='ready')
        return httpx.Response(200, json={'models':[{'name':name} for name in models]})
    original = httpx.Client
    monkeypatch.setattr(readiness.httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(respond)))
    assert readiness.check_dependencies(settings) is expected
    assert requests[0] == 'http://qdrant:6333/readyz'


def test_bootstrap_waits_until_dependencies_are_ready(monkeypatch):
    attempts = iter([False, False, True])
    sleeps = []
    monkeypatch.setattr(bootstrap, 'check_dependencies', lambda settings: next(attempts))
    monkeypatch.setattr(bootstrap.time, 'sleep', sleeps.append)
    bootstrap.wait_until_ready(Settings(_env_file=None), interval=3)
    assert sleeps == [3, 3]


def test_bootstrap_exits_with_error_if_dependencies_do_not_start(monkeypatch):
    monkeypatch.setattr(bootstrap, 'check_dependencies', lambda settings: False)
    with pytest.raises(RuntimeError, match='docker-compose logs'):
        bootstrap.wait_until_ready(Settings(_env_file=None), timeout=0)


def test_health_endpoint_reports_readiness_without_app_token(monkeypatch, tmp_path):
    from app import main
    settings = Settings(_env_file=None, data_dir=tmp_path, app_token='secret')
    with TestClient(create_app(settings)) as client:
        monkeypatch.setattr(main, 'check_dependencies', lambda settings: False)
        assert client.get('/healthz').status_code == 503
        monkeypatch.setattr(main, 'check_dependencies', lambda settings: True)
        assert client.get('/healthz').json() == {'status':'ok'}
        assert client.get('/api/documents').status_code == 401
