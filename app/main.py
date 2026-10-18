import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.service import RagService, ServiceError
from app.readiness import check_dependencies

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = Field(default=None, max_length=500)


def create_app(settings: Settings | None = None, service: RagService | None = None):
    settings = settings or Settings()
    service = service or RagService(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title='RagDocument', version='1.0.0', lifespan=lifespan)
    app.state.service = service

    def authorize(request: Request):
        if settings.app_token:
            supplied = request.headers.get('authorization', '')
            if not hmac.compare_digest(supplied.encode(), ('Bearer ' + settings.app_token).encode()):
                raise HTTPException(401, 'Введите токен доступа к приложению.')

    secured = [Depends(authorize)]

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(status_code=exc.status, content={'detail': exc.message})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        log.error('Request failed (%s)', type(exc).__name__)
        return JSONResponse(status_code=503, content={'detail':
            'Операция не выполнена. Проверьте подключение к Qdrant и настройки сервера.'})

    @app.get('/api/status', dependencies=secured)
    def status():
        return dict(**service.status(), provider='ollama',
                    qdrant_mode='server',
                    chat_model=settings.chat_model, embedding_model=settings.embedding_model,
                    max_upload_mb=settings.max_upload_mb)

    @app.get('/healthz', include_in_schema=False)
    def health():
        if not check_dependencies(settings):
            return JSONResponse(status_code=503, content={'status': 'not_ready'})
        return {'status': 'ok'}

    @app.get('/api/documents', dependencies=secured)
    def list_documents():
        return service.list_documents()

    @app.post('/api/documents', dependencies=secured)
    async def upload(file: UploadFile = File(...)):
        content = bytearray()
        try:
            while chunk := await file.read(1024 * 1024):
                content.extend(chunk)
                if len(content) > settings.max_upload_mb * 1024 * 1024:
                    raise ServiceError(f'Максимальный размер файла: {settings.max_upload_mb} МБ.', 413)
            return await run_in_threadpool(service.ingest, file.filename or '', bytes(content))
        finally:
            await file.close()

    @app.delete('/api/documents/{document_id}', dependencies=secured)
    def delete(document_id: str):
        service.delete_document(document_id)
        return {'deleted': True}

    @app.get('/api/documents/{document_id}/chunks/{chunk_index}', dependencies=secured)
    def source(document_id: str, chunk_index: int):
        return service.get_source(document_id, chunk_index)

    @app.get('/api/documents/{document_id}/file', dependencies=secured)
    def original(document_id: str):
        path, filename = service.get_original(document_id)
        media_type = 'application/pdf' if path.suffix == '.pdf' else 'application/octet-stream'
        return FileResponse(path, filename=filename, media_type=media_type,
                            content_disposition_type='inline' if path.suffix == '.pdf' else 'attachment',
                            headers={'X-Content-Type-Options': 'nosniff'})

    @app.post('/api/ask', dependencies=secured)
    def ask(body: AskRequest):
        if not body.question.strip():
            raise ServiceError('Введите вопрос.')
        return service.ask(body.question.strip(), body.document_ids)

    app.mount('/', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='ui')
    return app


app = create_app()
