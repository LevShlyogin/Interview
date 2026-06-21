from langgraph.graph import StateGraph, START, END
from brusnika_knowledge_layer.graph.state import RAGState
from brusnika_knowledge_layer.graph.nodes import OrchestratorNodes

def route_after_grader(state: RAGState) -> str:
    """Функция маршрутизации. Решает, куда идти после оценки контекста."""
    if state.is_context_relevant:
        return "generate"  # Переход к узлу generator
    return "fallback"      # Переход к узлу fallback

def build_rag_graph():
    """Сборка и компиляция графа состояний."""
    workflow = StateGraph(RAGState)

    # 1. Регистрируем все узлы
    workflow.add_node("query_rewriter", OrchestratorNodes.rewrite_query)
    workflow.add_node("semantic_router", OrchestratorNodes.route_query)
    workflow.add_node("retriever", OrchestratorNodes.retrieve_documents)
    workflow.add_node("grader", OrchestratorNodes.grade_documents)
    workflow.add_node("generator", OrchestratorNodes.generate_answer)
    workflow.add_node("fallback", OrchestratorNodes.handle_fallback)

    # 2. Выстраиваем жесткие связи (Линейный участок)
    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "semantic_router")
    workflow.add_edge("semantic_router", "retriever")
    workflow.add_edge("retriever", "grader")

    # 3. УСЛОВНЫЙ ПЕРЕХОД (Ветвление логики)
    workflow.add_conditional_edges(
        "grader",               # Из какого узла выходим
        route_after_grader,     # Функция, которая принимает решение
        {
            "generate": "generator", # Если вернула "generate", идем в "generator"
            "fallback": "fallback"   # Если вернула "fallback", идем в "fallback"
        }
    )

    # 4. Сводим обе ветки к завершению
    workflow.add_edge("generator", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_rag_graph()
    
    print("\n" + "="*60)
    print("🤖 БРУСНИКА: AGENTIC RAG (Локальный режим | Qwen 2.5 3B)")
    print("Для выхода введите 'exit' или 'выход'")
    print("="*60 + "\n")

    chat_memory = []

    while True:
        user_query = input("\n📝 Ваш вопрос: ")
        
        if user_query.lower() in ['exit', 'выход', 'quit']:
            print("Завершение работы. До свидания!")
            break
            
        if not user_query.strip():
            continue

        initial_state = RAGState(
            original_query=user_query,
            chat_history=chat_memory, 
            user_access_level="internal"
        )
        
        print("\n[🚀] Обработка запроса...\n")
        final_state = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print(f"🏢 ДОМЕН ПОИСКА: {final_state['search_domain'].upper()}")
        print("-" * 60)
        print(f"🤖 ОТВЕТ:\n{final_state['final_answer']}")
        print("-" * 60)
        print(f"📚 ИСТОЧНИКИ: {', '.join(final_state['sources']) if final_state['sources'] else 'Нет источников'}")
        print("=" * 60)
        
        chat_memory.append({
            "user": user_query,
            "ai": final_state['final_answer']
        })
        if len(chat_memory) > 5:
            chat_memory.pop(0)