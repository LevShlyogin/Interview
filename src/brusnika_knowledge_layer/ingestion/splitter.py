import uuid

from langchain_text_splitters import MarkdownHeaderTextSplitter

from brusnika_knowledge_layer.schema import ChunkPayload, RawDocument


class HierarchicalSplitter:
    def __init__(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    def _build_contextual_text(self, headers: dict[str, str], content: str) -> str:
        path = " -> ".join(headers.values())
        if path:
            return f"[{path}]\n{content.strip()}"
        return content.strip()

    def split_document(self, doc: RawDocument, extracted_tables: list[dict[str, str]]) -> list[ChunkPayload]:
        chunks = self.splitter.split_text(doc.page_content)
        payloads = []
        tables_dict = {t["table_id"]: t for t in extracted_tables}

        for chunk in chunks:
            headers = chunk.metadata
            h1 = headers.get("Header 1")
            h2 = headers.get("Header 2")

            content = chunk.page_content
            contextual_text = self._build_contextual_text(headers, content)

            is_table = False
            parent_id = None
            
            if "{{TABLE_ID:" in content:
                start_idx = content.find("{{TABLE_ID:") + 11
                end_idx = content.find("}}", start_idx)
                if end_idx != -1:
                    t_id = content[start_idx:end_idx]
                    if t_id in tables_dict:
                        is_table = True
                        parent_id = t_id
                        content = tables_dict[t_id]["summary"]
                        contextual_text = self._build_contextual_text(headers, content)

            payload = ChunkPayload(
                chunk_id=str(uuid.uuid4()),
                source_file=doc.metadata["source_file"],
                page_content=content,
                contextual_text=contextual_text,
                domain=doc.metadata["domain"],
                access=doc.metadata["access"],
                status=doc.metadata["status"],
                audience=doc.metadata["audience"],
                header_1=h1,
                header_2=h2,
                header_3=None, # Заглушка
                linked_docs=doc.metadata["linked_docs"],
                chunk_type="table_summary" if is_table else "text",
                parent_id=parent_id
            )
            payloads.append(payload)

        return payloads