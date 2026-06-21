from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from brusnika_knowledge_layer.graph.state import RAGState
from brusnika_knowledge_layer.retrieval.hybrid_search import HybridRetriever

# Инициализируем локальную Qwen 2.5 3B (temperature=0 для точной логики)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# Инициализация боевого гибридного ретривера с Reranker'ом
retriever = HybridRetriever()

class RouteDecision(BaseModel):
    domain: str = Field(
        description="Домен базы знаний. Варианты: 'hr', 'legal', 'construction', 'it', 'finance', 'company', 'commercial', 'development', 'exploitation'. Если запрос кросс-доменный или непонятно куда его отнести, верни 'all'."
    )

class OrchestratorNodes:
    
    @staticmethod
    def rewrite_query(state: RAGState) -> RAGState:
        if not state.chat_history:
            print(f"[Узел: Rewriter] Первый вопрос: '{state.original_query}'")
            state.rewritten_query = state.original_query
            return state

        print(f"[Узел: Rewriter] Анализ запроса с учетом истории...")
        
        # Берем только 2 последних сообщения, чтобы не перегружать контекст 3B-модели
        history_text = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in state.chat_history[-2:]])
        
        # ЖЕСТКИЙ ПРОМПТ С ПРИМЕРАМИ (FEW-SHOT)
        prompt = (
            "Ты — строгий лингвистический анализатор. Твоя задача — переписать текущий вопрос пользователя так, чтобы он был понятен без контекста предыдущей беседы.\n\n"
            "ПРИМЕРЫ (FEW-SHOT):\n"
            "История: User: Какие лимиты по суточным?\nAI: 1500 рублей.\nТекущий вопрос: А для руководителей?\n-> Какие лимиты по суточным для руководителей?\n\n"
            "История: User: Что такое ПОС?\nAI: Проект организации строительства.\nТекущий вопрос: Кто его составляет?\n-> Кто составляет Проект организации строительства (ПОС)?\n\n"
            "История: User: Как запросить доступ к 1С?\nAI: Напишите заявку.\nТекущий вопрос: Я новый инженер ПТО, что делать в первую неделю?\n-> Что нужно сделать новому инженеру ПТО в первую неделю?\n\n"
            f"ИСТОРИЯ ТЕКУЩЕГО ДИАЛОГА:\n{history_text}\n\n"
            f"Текущий вопрос пользователя: {state.original_query}\n"
            "В ответ напиши ТОЛЬКО переписанный вопрос, без кавычек, пояснений и лишних слов. Если вопрос уже самостоятельный, верни его как есть."
        )
        
        response = llm.invoke([("human", prompt)])
        state.rewritten_query = response.content.strip()
        print(f"  -> Переписанный запрос: '{state.rewritten_query}'")
        
        return state

    @staticmethod
    def route_query(state: RAGState) -> RAGState:
        print(f"[Узел: Router] Классификация запроса через Qwen 2.5...")
        
        router_llm = llm.with_structured_output(RouteDecision)
        
        prompt = (
            f"Ты — строгий классификатор корпоративной базы знаний 'Брусника'.\n"
            f"Определи домен для вопроса: '{state.rewritten_query}'\n\n"
            f"ПРАВИЛА КЛАССИФИКАЦИИ:\n"
            f"- 'it': доступы, VPN, 1С, CRM, компьютеры, информационная безопасность.\n"
            f"- 'construction': ПОС, ППР, СГП, материалы, бетон, РНС, подрядчики.\n"
            f"- 'hr': отпуска, суточные, командировки, онбординг, зарплата.\n"
            f"ВАЖНОЕ ИСКЛЮЧЕНИЕ: Любые вопросы про 'первую неделю', 'адаптацию', 'нового сотрудника' — это всегда 'hr' или 'all', даже если указана строительная должность (ПТО).\n"
            f"Если вопрос затрагивает несколько тем или ты сомневаешься, верни 'all'. Верни только JSON."
        )
        
        try:
            decision = router_llm.invoke(prompt)
            state.search_domain = decision.domain
        except Exception as e:
            print(f"  [Ошибка роутера] Локальная модель сломала JSON: {e}. Переключаем на 'all'.")
            state.search_domain = "all"
            
        print(f"  -> Qwen выбрала домен для поиска: {state.search_domain.upper()}")
        return state

    @staticmethod
    def retrieve_documents(state: RAGState) -> RAGState:
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
        print(f"[Узел: Grader] Оценка качества извлеченных чанков...")
        valid_chunks = []
        warnings = []
        
        for chunk in state.retrieved_chunks:
            if chunk.get("status") == "archived":
                warning_msg = f"Документ {chunk['source_file']} устарел (archived). Информация может быть недействительной."
                warnings.append(warning_msg)
                print(f"  [!] Отсеян устаревший документ: {chunk['source_file']}")
            else:
                valid_chunks.append(chunk)
                
        if valid_chunks:
            state.is_context_relevant = True
            state.retrieved_chunks = valid_chunks
        else:
            state.is_context_relevant = False
            
        state.context_warnings = warnings
        return state

    @staticmethod
    def generate_answer(state: RAGState) -> RAGState:
        print(f"[Узел: Generator] Чтение базы и генерация ответа...")
        
        context_texts = "\n\n---\n\n".join([f"Источник [{chunk['source_file']}]:\n{chunk['content']}" for chunk in state.retrieved_chunks])
        warnings_text = "\n".join(state.context_warnings)
        
        # ЖЕСТКИЙ АНТИ-ГАЛЛЮЦИНОГЕННЫЙ ПРОМПТ С НАСТРОЙКОЙ ТОНА
        sys_prompt = (
            "Ты — заботливый, но строгий корпоративный ИИ-ассистент компании Брусника. "
            "Твоя главная задача — абсолютная достоверность.\n"
            "ПРАВИЛА:\n"
            "1. Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
            "2. Адаптируй текст под пользователя! Если это инструкция, пиши от лица компании к сотруднику (например: 'Вам нужно получить СИЗ', а не 'Выдать СИЗ сотруднику').\n"
            "3. Строго соблюдай запрошенный формат (если просят чек-лист, используй Markdown чек-боксы '- [ ]').\n"
            "4. Отвечай точно на заданный вопрос. Если спрашивают про первую неделю, не пиши про 90 дней.\n"
            "5. Если ответа нет в тексте, пиши: 'В базе знаний нет точной информации'. ЗАПРЕЩАЕТСЯ выдумывать факты.\n\n"
        )
        
        if warnings_text:
            sys_prompt += f"ВНИМАНИЕ! Упомяни эти предупреждения:\n{warnings_text}\n\n"
            
        sys_prompt += f"КОНТЕКСТ:\n{context_texts}"
        
        messages = [("system", sys_prompt), ("human", state.rewritten_query)]
        response = llm.invoke(messages)
        
        state.final_answer = response.content
        state.sources = list(set([chunk["source_file"] for chunk in state.retrieved_chunks]))
        
        return state

    @staticmethod
    def handle_fallback(state: RAGState) -> RAGState:
        print(f"[Узел: Fallback] Релевантный контекст отсутствует. Переход в режим уточнения.")
        state.final_answer = "К сожалению, я не нашел актуальной информации по вашему запросу в базе знаний. Пожалуйста, переформулируйте вопрос."
        return state