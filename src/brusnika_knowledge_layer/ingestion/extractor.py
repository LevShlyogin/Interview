import re
import uuid

from brusnika_knowledge_layer.schema import RawDocument


class TableExtractor:
    def __init__(self, llm_client=None):
        """
        Инициализация экстрактора. 
        llm_client - клиент модели (LangChain/кастомный) для генерации саммари.
        """
        self.llm_client = llm_client
        # Регулярка ищет блоки текста, где строки начинаются и заканчиваются на пайп (|)
        self.table_regex = re.compile(r"(^\|.+?\|$\n)+^\|.+?\|$", re.MULTILINE)

    def generate_summary(self, raw_table: str) -> str:
            """Метод для обращения к LLM за саммаризацией таблицы."""
            if not self.llm_client:
                return "Сгенерированное текстовое описание данных из таблицы (Mock)."
            
            print("      [LLM] Генерирую саммари для таблицы...")
            prompt = f"Сделай подробное текстовое описание данных из этой Markdown-таблицы. Пиши только описание, без лишних вступлений:\n\n{raw_table}"
            
            # Реальный вызов локальной модели
            response = self.llm_client.invoke(prompt)
            return response.content

    def process_document(self, doc: RawDocument) -> tuple[RawDocument, list[dict[str, str]]]:
        """
        Находит таблицы, заменяет их на ID и возвращает измененный документ + список извлеченных таблиц.
        """
        extracted_tables = []
        text_content = doc.page_content

        # Ищем все совпадения таблиц
        matches = list(self.table_regex.finditer(text_content))
        
        # Идем с конца, чтобы индексы не смещались при замене текста
        for match in reversed(matches):
            raw_table = match.group(0)
            table_id = f"tbl_{uuid.uuid4().hex[:8]}"
            placeholder = f"{{{{TABLE_ID:{table_id}}}}}"
            
            # Генерируем саммари (сработает Qwen3/Gemini)
            summary = self.generate_summary(raw_table)
            
            extracted_tables.append({
                "table_id": table_id,
                "raw_markdown": raw_table,
                "summary": summary
            })
            
            # Заменяем таблицу на плейсхолдер в сыром тексте
            start, end = match.span()
            text_content = text_content[:start] + placeholder + text_content[end:]

        # Обновляем контент документа
        doc.page_content = text_content
        return doc, extracted_tables
        