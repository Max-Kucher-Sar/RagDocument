"""End-to-end check against the Compose Ollama and Qdrant services.

Run: docker-compose --profile smoke run --rm smoke
No production documents are read or changed.
"""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from openpyxl import Workbook

from app.config import Settings
from app.service import RagService


def main():
    with TemporaryDirectory(prefix='smoke-ollama-') as directory:
        root = Path(directory)
        collection = 'smoke_' + uuid4().hex
        settings = Settings(data_dir=root / 'documents', app_token='', qdrant_collection=collection)
        service = RagService(settings)
        try:
            status = service.status()
            assert status['ollama_available'] and not status['missing_models'], status
            book = Workbook(); sheet = book.active; sheet.title = 'Возвраты'
            sheet.append(['Товар', 'Срок возврата'])
            sheet.append(['Чайник', '14 дней'])
            sheet.append(['Холодильник', '30 дней'])
            stream = BytesIO(); book.save(stream)
            doc = service.ingest('demo.xlsx', stream.getvalue())
            result = service.ask('Какой срок возврата чайника?', [doc['id']])
            print(result['answer'], flush=True)
            assert not result['insufficient_evidence'], result
            assert '14' in result['answer'], result
            assert any(source['sheet'] == 'Возвраты' and 'A2:B2' in source['cell_range']
                       for source in result['sources']), result
            for source in result['sources']:
                print(source['location'], source['url'], flush=True)
            missing = service.ask('Как зовут директора магазина?', [doc['id']])
            assert missing['insufficient_evidence'], missing
            print('PASS: real Ollama + Qdrant + XLSX + citations + no-evidence answer.', flush=True)
        finally:
            try:
                if service.qdrant is not None and service.qdrant.collection_exists(collection):
                    service.qdrant.delete_collection(collection)
            finally:
                service.close()


if __name__ == '__main__':
    main()
