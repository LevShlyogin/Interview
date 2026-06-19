from brusnika_knowledge_layer.graph.state import RAGState
from brusnika_knowledge_layer.retrieval.hybrid_search import HybridRetriever

# Инициализируем наш ретривер
retriever = HybridRetriever()

class OrchestratorNodes:
    """Узлы первичной обработки запроса (Работают на быстрой локальной LLM)."""
    
    @staticmethod
    def rewrite_query(state: RAGState) -> RAGState:
        """
        Узел 1: Разрешение кореференции (Анафоры).
        Заменяет местоимения на конкретные термины из истории.
        """
        print(f"[Узел: Rewriter] Анализ запроса: '{state.original_query}'")
        
        # Эмуляция логики LLM:
        # Если вопрос "А кто ЕГО проверяет?", а в истории говорили про ДДУ
        # LLM должна вернуть "Кто проверяет договор ДДУ?"
        
        # В реальной жизни тут будет вызов: llm.invoke(prompt)
        # Пока мы просто копируем запрос для демо
        state.rewritten_query = state.original_query
        return state

    @staticmethod
    def route_query(state: RAGState) -> RAGState:
        """
        Узел 2: Семантический роутинг.
        Определяет домен для жесткой фильтрации в Qdrant.
        """
        print(f"[Узел: Router] Маршрутизация запроса: '{state.rewritten_query}'")
        
        query_lower = state.rewritten_query.lower()
        
        # Простая эмуляция роутинга через ключевые слова
        if any(word in query_lower for word in ["отпуск", "командировка", "грейд", "суточные"]):
            state.search_domain = "hr"
        elif any(word in query_lower for word in ["дду", "договор", "суд", "иск"]):
            state.search_domain = "legal"
        elif any(word in query_lower for word in ["стройка", "цемент", "подрядчик", "сгп"]):
            state.search_domain = "construction"
        else:
            state.search_domain = "general"
            
        print(f"  -> Выбран домен поиска: {state.search_domain}")
        return state

    @staticmethod
    def retrieve_documents(state: RAGState) -> RAGState:
        """
        Узел 3: Вызов векторной базы.
        """
        print(f"[Узел: Retriever] Извлечение контекста из базы...")
        
        chunks = retriever.search(
            query=state.rewritten_query,
            domain=state.search_domain,
            access_level=state.user_access_level
        )
        state.retrieved_chunks = chunks
        return state

    @staticmethod
    def grade_documents(state: RAGState) -> RAGState:
        """
        Узел 4: Проверка контекста (Context Grader).
        Фильтрует мусор и ловит архивные документы.
        """
        print(f"[Узел: Grader] Оценка качества извлеченных чанков...")
        
        valid_chunks = []
        warnings = []
        
        for chunk in state.retrieved_chunks:
            # Если база вернула архивный документ, мы не используем его для генерации ответа,
            # но обязательно сохраняем предупреждение для финальной LLM.
            if chunk.get("status") == "archived":
                warning_msg = f"Документ {chunk['source_file']} помечен как 'archived'."
                warnings.append(warning_msg)
                print(f"  [!] Обнаружен устаревший документ: {chunk['source_file']}")
            else:
                valid_chunks.append(chunk)
                
        # Если после фильтрации остались хорошие документы
        if valid_chunks:
            state.is_context_relevant = True
            state.retrieved_chunks = valid_chunks
        else:
            state.is_context_relevant = False
            
        state.context_warnings = warnings
        return state
    
    @staticmethod
    def generate_answer(state: RAGState) -> RAGState:
        """
        Узел 5: Финальная генерация (Gemini 2.5 Flash).
        Собирает контекст, системный промпт и отдает ответ.
        """
        print(f"[Узел: Generator] Сборка контекста и генерация ответа...")
        
        # Склеиваем отфильтрованные тексты чанков
        context_texts = "\n\n".join([f"Источник: {chunk['source_file']}\n{chunk['content']}" for chunk in state.retrieved_chunks])
        
        # Формируем блок предупреждений
        warnings_text = "\n".join(state.context_warnings)
        warnings_prompt = f"\nВНИМАНИЕ: {warnings_text}" if warnings_text else ""
        
        # Эмуляция ответа тяжелой LLM (позже заменим на реальный вызов Gemini API)
        # В реальной жизни промпт выглядит так: "Используя контекст: {context_texts}, ответь на вопрос: {state.rewritten_query}. Учти предупреждения: {warnings_prompt}"
        
        mock_response = (
            "Согласно внутренним правилам (business-travel-policy.md), суточные для поездок "
            "на объекты строительства (СГП) составляют 1500 рублей в сутки.\n"
        )
        if warnings_prompt:
            mock_response += f"\n*Системное примечание: {warnings_text}*"
            
        state.final_answer = mock_response
        state.sources = [chunk["source_file"] for chunk in state.retrieved_chunks]
        
        return state

    @staticmethod
    def handle_fallback(state: RAGState) -> RAGState:
        """
        Узел 6: Альтернативная ветка (Фоллбек).
        Срабатывает, если контекст не найден или заблокирован.
        """
        print(f"[Узел: Fallback] Релевантный контекст отсутствует. Переход в режим уточнения.")
        state.final_answer = "К сожалению, я не нашел актуальной информации по вашему запросу в базе знаний. Пожалуйста, переформулируйте вопрос или уточните детали."
        return state