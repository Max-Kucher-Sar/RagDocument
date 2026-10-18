"""Small client for the native local Ollama protocol; no cloud fallback."""
import json

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.errors import ServiceError


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.ollama_base_url.rstrip('/') + '/',
            timeout=httpx.Timeout(settings.ollama_timeout, connect=5), trust_env=False)

    def close(self):
        self.client.close()

    def _post(self, path, body):
        try:
            response = self.client.post(path, json=body)
            if response.status_code == 404:
                raise ServiceError(f'Модель {body["model"]} не найдена в Ollama. '
                                   'Выполните: docker-compose run --rm ollama-init', 503)
            if response.is_error:
                raise ServiceError('Ollama отклонила запрос. Проверьте модель, свободную память '
                                   'и журнал Ollama. Для эмбеддингов нужна embedding-модель.', 502)
            data = response.json()
            if not isinstance(data, dict):
                raise ServiceError('Ollama вернула неверную структуру ответа.', 502)
            return data
        except httpx.TimeoutException as exc:
            raise ServiceError('Ollama не успела ответить. Проверьте доступную память, '
                               'загрузку компьютера или увеличьте OLLAMA_TIMEOUT.', 504) from exc
        except httpx.RequestError as exc:
            raise ServiceError('Нет соединения с Ollama. Проверьте сервис: docker-compose logs ollama', 503) from exc
        except ValueError as exc:
            raise ServiceError('Ollama вернула некорректный JSON.', 502) from exc

    def status(self):
        try:
            response = self.client.get('api/tags', timeout=5)
            response.raise_for_status()
            names = {entry['name'] for entry in response.json()['models']}
            required = [self.settings.chat_model, self.settings.embedding_model]
            missing = [name for name in required if name not in names and name + ':latest' not in names]
            return dict(ollama_available=True, missing_models=missing)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return dict(ollama_available=False, missing_models=[])

    def ensure_models(self, *, chat=False):
        status = self.status()
        if not status['ollama_available']:
            raise ServiceError('Нет соединения с Ollama. Проверьте сервис: docker-compose logs ollama', 503)
        missing = status['missing_models']
        needed = missing if chat else [name for name in missing if name == self.settings.embedding_model]
        if needed:
            raise ServiceError('В Ollama нет нужных моделей: ' + ', '.join(needed) +
                               '. Выполните: docker-compose run --rm ollama-init', 503)

    def embed(self, texts: list[str], *, query=False):
        # Qwen3 Embedding recommends an instruction on queries, but not on passages.
        if query and self.settings.embedding_model.startswith('qwen3-embedding'):
            texts = ['Instruct: Given a question, retrieve passages that answer the question.\nQuery: ' + text
                     for text in texts]
        data = self._post('api/embed', dict(model=self.settings.embedding_model, input=texts,
                                           truncate=False, keep_alive='5m'))
        vectors = data.get('embeddings')
        if not isinstance(vectors, list):
            raise ServiceError('Ollama не вернула векторы эмбеддингов.', 502)
        return vectors

    def generate(self, instructions, question, sources, response_model):
        schema = response_model.model_json_schema()
        data = self._post('api/chat', dict(
            model=self.settings.chat_model, stream=False, think=False, format=schema,
            messages=[dict(role='system', content=instructions + '\nВерни JSON по схеме: ' +
                           json.dumps(schema, ensure_ascii=False)),
                      dict(role='user', content=json.dumps(dict(question=question, sources=sources),
                                                          ensure_ascii=False))],
            options=dict(temperature=0, num_ctx=self.settings.ollama_num_ctx,
                         num_predict=self.settings.ollama_num_predict), keep_alive='5m'))
        if data.get('done_reason') == 'length':
            raise ServiceError('Ответ модели обрезан. Уточните вопрос или увеличьте OLLAMA_NUM_PREDICT.', 502)
        try:
            return response_model.model_validate_json(data['message']['content'])
        except (ValidationError, KeyError, TypeError) as exc:
            raise ServiceError('Модель вернула ответ неверного формата. Уточните вопрос и повторите запрос.', 502) from exc

