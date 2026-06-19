import uuid
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter
from brusnika_knowledge_layer.schema import RawDocument, ChunkPayload

class HierarchicalSplitter:
    def __init__(self):
        # Настраиваем отслеживание заголовков 1, 2 и 3 уровней
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    def _build_contextual_text(self, headers: Dict[str, str], content: str) -> str:
        """Склеивает иерархию заголовков с текстом чанка для Dense/Sparse векторизации."""
        path = " -> ".join(headers.values())
        if path:
            return f"[{path}]\n{content.strip()}"
        return content.strip()

    def split_document(self, doc: RawDocument, extracted_tables: List[Dict[str, str]]) -> List[ChunkPayload]:
        """Разбивает документ на чанки и формирует финальные Payload-объекты."""
        chunks = self.splitter.split_text(doc.page_content)
        payloads = []

        # Создаем словарь таблиц для быстрого поиска по ID
        tables_dict = {t["table_id"]: t for t in extracted_tables}

        for chunk in chunks:
            # chunk.metadata здесь содержит только заголовки от LangChain
            headers = chunk.metadata
            h1 = headers.get("Header 1")
            h2 = headers.get("Header 2")
            h3 = headers.get("Header 3")

            content = chunk.page_content
            contextual_text = self._build_contextual_text(headers, content)

            # Проверяем, есть ли в этом чанке плейсхолдер таблицы
            is_table = False
            parent_id = None
            
            # Ищем маркер {{TABLE_ID:xxx}}
            if "{{TABLE_ID:" in content:
                # В реальном коде лучше использовать регулярку, но для простоты берем строковые методы
                start_idx = content.find("{{TABLE_ID:") + 11
                end_idx = content.find("}}", start_idx)
                if end_idx != -1:
                    t_id = content[start_idx:end_idx]
                    if t_id in tables_dict:
                        is_table = True
                        parent_id = t_id
                        # Подменяем плейсхолдер на сгенерированное текстовое саммари
                        content = tables_dict[t_id]["summary"]
                        # Обновляем contextual_text для таблицы
                        contextual_text = self._build_contextual_text(headers, content)

            # Формируем финальный объект дата-контракта
            payload = ChunkPayload(
                chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
                source_file=doc.metadata["source_file"],
                page_content=content,
                contextual_text=contextual_text,
                domain=doc.metadata["domain"],
                access=doc.metadata["access"],
                status=doc.metadata["status"],
                audience=doc.metadata["audience"],
                header_1=h1,
                header_2=h2,
                header_3=h3,
                linked_docs=doc.metadata["linked_docs"],
                chunk_type="table_summary" if is_table else "text",
                parent_id=parent_id
            )
            payloads.append(payload)

        return payloads