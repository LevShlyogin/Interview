from typing import Any

from pydantic import BaseModel, Field


class RAGState(BaseModel):
    """Глобальное состояние графа для обработки одного запроса."""
    
    # Входные данные
    original_query: str = Field(description="Сырой вопрос от пользователя")
    chat_history: list[dict[str, str]] = Field(default_factory=list, description="История диалога (роль, текст)")
    user_access_level: str = Field(default="internal", description="Уровень доступа пользователя (из токена)")
    
    # Промежуточные данные
    rewritten_query: str | None = Field(default=None, description="Переписанный вопрос (без анафоры)")
    search_domain: str = Field(default="all", description="Выявленный домен для фильтрации в Qdrant")
    
    # Результаты поиска
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list, description="Найденные чанки из базы")
    context_warnings: list[str] = Field(default_factory=list, description="Предупреждения от Grader (например, архивные доки)")
    
    # Финальная генерация
    final_answer: str | None = Field(default=None, description="Сгенерированный ответ Gemini")
    sources: list[str] = Field(default_factory=list, description="Список файлов-источников")
    
    # Флаги маршрутизации
    is_context_relevant: bool = Field(default=False, description="Нашел ли Grader полезную информацию?")
    needs_clarification: bool = Field(default=False, description="Требуется ли уточнение от пользователя?")