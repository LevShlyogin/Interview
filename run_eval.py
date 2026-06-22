import os
import re
from tqdm import tqdm
from brusnika_knowledge_layer.graph.workflow import build_rag_graph
from brusnika_knowledge_layer.graph.state import RAGState

# Пути к файлам
INPUT_FILE = "test-questions.md"
OUTPUT_FILE = "test-answers.md"

def extract_questions(file_path: str) -> list[str]:
    """Парсит файл с вопросами, игнорируя YAML-фронтматтер и вступления."""
    questions = []
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден!")
        return questions

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Разбиваем весь текст по заголовкам "## Вопрос 1", "## Вопрос 2" и т.д.
    # parts[0] будет содержать весь YAML и вступление (мы его пропустим).
    parts = re.split(r'##\s*Вопрос\s*\d+', content, flags=re.IGNORECASE)

    for part in parts[1:]:
        question = part.strip()
        if question:
            # Убираем переносы строк внутри самого вопроса, чтобы не сломать таблицу
            clean_question = question.replace("\n", " ")
            questions.append(clean_question)

    return questions

def run_evaluation():
    print("[1] Инициализация RAG-графа...")
    app = build_rag_graph()
    
    questions = extract_questions(INPUT_FILE)
    if not questions:
        return

    print(f"[2] Найдено вопросов для теста: {len(questions)}")
    print("[3] Начинаем прогон (это может занять время)...\n")

    # Подготавливаем шапку для финального Markdown-файла
    md_content = [
        "# 📊 Результаты тестирования RAG-системы",
        "",
        "| № | Вопрос | Фактический ответ | Использованные источники | Оценка качества |",
        "|---|---|---|---|---|"
    ]

    # Прогоняем каждый вопрос через граф
    for i, question in enumerate(tqdm(questions, desc="Обработка запросов"), start=1):
        try:
            # Инициализируем стейт (передаем пустую историю и стандартный доступ)
            state = RAGState(
                original_query=question,
                chat_history=[],
                user_access_level="internal"
            )
            
            # Запускаем граф
            result_state = app.invoke(state)
            
            # Извлекаем данные
            answer = result_state.get("final_answer", "ОШИБКА ГЕНЕРАЦИИ").strip()
            sources = result_state.get("sources", [])
            sources_str = ", ".join(sources) if sources else "Нет"
            
            # Очищаем ответ от переносов строк, чтобы не сломать Markdown-таблицу
            clean_answer = answer.replace("\n", " <br> ").replace("|", "&#124;")
            
            # Формируем строку таблицы (Оценку оставляем пустой для ручного заполнения)
            row = f"| {i} | {question} | {clean_answer} | {sources_str} | ⬜ Ожидает оценки |"
            md_content.append(row)
            
        except Exception as e:
            error_msg = f"ОШИБКА: {str(e)}"
            row = f"| {i} | {question} | {error_msg} | - | ❌ Fail |"
            md_content.append(row)

    # Записываем результаты в файл
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"\n[+] ✅ Готово! Таблица с ответами сохранена в {OUTPUT_FILE}")
    print("[!] Зайдите в файл и заполните колонку 'Оценка качества' вручную.")

if __name__ == "__main__":
    run_evaluation()