from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Compose reads optional .env settings and passes an explicit environment.
    model_config = SettingsConfigDict(extra='ignore')

    ollama_base_url: str = 'http://ollama:11434'
    ollama_timeout: float = Field(300, ge=1)
    ollama_num_ctx: int = Field(8192, ge=2048)
    ollama_num_predict: int = Field(1200, ge=128)
    chat_model: str = 'qwen3:4b-instruct'
    embedding_model: str = 'qwen3-embedding:0.6b'
    embedding_dimensions: int = Field(1024, ge=1)
    qdrant_url: str = Field('http://qdrant:6333', min_length=1)
    qdrant_collection: str = 'documents_ollama_qwen3'
    data_dir: Path = Path('data/documents')
    max_upload_mb: int = Field(20, ge=1, le=200)
    chunk_size: int = Field(1200, ge=200, le=4000)
    chunk_overlap: int = Field(180, ge=0)
    top_k: int = Field(6, ge=1, le=20)
    score_threshold: float = Field(0.25, ge=-1, le=1)
    app_token: str = ''

    @model_validator(mode='after')
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('CHUNK_OVERLAP должен быть меньше CHUNK_SIZE')
        return self

