from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Промежуточная модель для сырого документа после чтения Frontmatter."""
    page_content: str = Field(description="Сырой Markdown-текст (без YAML, но еще с таблицами)")
    metadata: dict[str, Any] = Field(description="Словарь метаданных, нормализованный из Frontmatter")

class ChunkPayload(BaseModel):
    """Финальный дата-контракт для Qdrant (Payload вектора с поддержкой Hybrid и Contextual Chunks)."""
    # 1. Системные идентификаторы и атрибуция
    chunk_id: str = Field(description="Уникальный ID чанка (UUID4)")
    source_file: str = Field(description="Имя исходного файла (например, '214-fz-compliance.md')")
    
    # 2. Текстовое наполнение (Разделение для эмбеддингов и генерации)
    page_content: str = Field(description="Чистый текст чанка/таблицы для подачи в Gemini")
    contextual_text: str = Field(description="Обогащенный текст (Хедеры + Текст) для Dense & Sparse векторизации")
    
    # 3. Метаданные из Frontmatter (для жесткой фильтрации в Qdrant)
    domain: str = Field(description="Домен знаний: hr, legal, construction, итд")
    access: str = Field(description="Уровень доступа: public, internal, confidential, restricted")
    status: str = Field(description="Статус актуальности: active, needs-review, archived")
    audience: list[str] = Field(description="Целевая аудитория: employees, management, sgp_workers")
    
    # 4. Структурный контекст (от MarkdownHeaderTextSplitter)
    header_1: str | None = Field(default=None, description="Заголовок H1 верхнего уровня")
    header_2: str | None = Field(default=None, description="Заголовок H2")
    header_3: str | None = Field(default=None, description="Заголовок H3")
    
    # 5. Multi-hop навигация (Связи между вершинами графа знаний)
    linked_docs: list[str] = Field(default_factory=list, description="Массив связанных .md файлов")
    
    # 6. Parent-Child логика (Для продвинутой обработки таблиц)
    chunk_type: str = Field(description="Тип контента: 'text' или 'table_summary'")
    parent_id: str | None = Field(default=None, description="ID сырой Markdown-таблицы (если chunk_type == 'table_summary')")