import streamlit as st

from brusnika_knowledge_layer.graph.state import RAGState
from brusnika_knowledge_layer.graph.workflow import build_rag_graph

# Настройка страницы
st.set_page_config(page_title="БРУСНИКА | База Знаний", page_icon="🏢", layout="centered")
st.title("🏢 БРУСНИКА: Corporate AI-Assistant")
st.markdown("Задайте вопрос по регламентам, процессам и базе знаний (работает локально).")

# Инициализация графа (кэшируем, чтобы модели не грузились при каждом обновлении страницы)
@st.cache_resource
def get_graph():
    return build_rag_graph()

app = get_graph()

# Инициализация состояния сессии для хранения истории UI и графа
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Отрисовка предыдущих сообщений в интерфейсе
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Обработка нового ввода
if prompt := st.chat_input("Ваш вопрос (например, 'Как оформить командировку на СГП?'):"):
    
    # Показываем вопрос пользователя
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Вызываем граф и показываем спиннер загрузки
    with st.spinner('Анализ базы знаний...'):
        initial_state = RAGState(
            original_query=prompt,
            chat_history=st.session_state.chat_history,
            user_access_level="internal"
        )
        
        final_state = app.invoke(initial_state)
        
        answer = final_state['final_answer']
        sources = final_state.get('sources', [])
        domain = final_state.get('search_domain', 'ALL')

        # Форматируем красивый подвал с источниками
        source_text = "\n\n---\n**📚 Источники:** " + ", ".join(sources) if sources else "\n\n---\n*Источники не найдены*"
        domain_text = f"\n*🔍 Маршрутизация: {domain.upper()}*"
        
        full_response = f"{answer}{source_text}{domain_text}"

    # Ответ ИИ
    with st.chat_message("assistant"):
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Обновляем память графа
    st.session_state.chat_history.append({
        "user": prompt,
        "ai": answer
    })
    # Храним только 5 последних сообщений для стабильности
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history.pop(0)