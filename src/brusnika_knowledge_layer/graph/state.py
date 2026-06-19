from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RAGState(BaseModel):
    """Глобальное состояние графа для обработки одного запроса."""
    
    # Входные данные
    original_query: str = Field(description="Сырой вопрос от пользователя")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="История диалога (роль, текст)")
    user_access_level: str = Field(default="internal", description="Уровень доступа пользователя (из токена)")
    
    # Промежуточные данные (Результат работы узлов)
    rewritten_query: Optional[str] = Field(default=None, description="Переписанный вопрос (без анафоры)")
    search_domain: str = Field(default="all", description="Выявленный домен для фильтрации в Qdrant")
    
    # Результаты поиска
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Найденные чанки из базы")
    context_warnings: List[str] = Field(default_factory=list, description="Предупреждения от Grader (например, архивные доки)")
    
    # Финальная генерация
    final_answer: Optional[str] = Field(default=None, description="Сгенерированный ответ Gemini")
    sources: List[str] = Field(default_factory=list, description="Список файлов-источников")
    
    # Флаги маршрутизации (Управляют графом)
    is_context_relevant: bool = Field(default=False, description="Нашел ли Grader полезную информацию?")
    needs_clarification: bool = Field(default=False, description="Требуется ли уточнение от пользователя?")