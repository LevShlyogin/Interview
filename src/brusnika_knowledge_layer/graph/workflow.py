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
    
    initial_state = RAGState(
        original_query="Какие лимиты по суточным для поездки на стройку СГП?",
        chat_history=[],
        user_access_level="internal"
    )
    
    print("\n[🚀] Запуск Agentic RAG графа...\n")
    final_state = app.invoke(initial_state)
    
    print("\n[✅] Выполнение графа завершено. Итоговое состояние:")
    print("=" * 50)
    print(f"Сырой вопрос:       {final_state['original_query']}")
    print(f"Домен поиска:       {final_state['search_domain'].upper()}")
    print("-" * 50)
    print(f"Ответ пользователю: {final_state['final_answer']}")
    print(f"Источники:          {final_state['sources']}")
    print("=" * 50)