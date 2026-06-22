import os
from pathlib import Path

import frontmatter

from src.brusnika_knowledge_layer.schema import RawDocument


class DocumentParser:
    def __init__(self, data_dir: str = "data/knowledge_base"):
        self.data_dir = Path(data_dir)

    def parse_file(self, file_path: Path) -> RawDocument:
        """Считывает файл, извлекает YAML-frontmatter и нормализует под дата-контракт."""
        with open(file_path, encoding="utf-8") as f:
            parsed_file = frontmatter.load(f)

        raw_metadata = parsed_file.metadata
        content = parsed_file.content

        # Извлечение связанных документов
        related = raw_metadata.get("related", [])
        if isinstance(related, str):
            linked_docs = [related]
        elif isinstance(related, list):
            linked_docs = [str(doc) for doc in related]
        else:
            linked_docs = []

        # Нормализация метаданных под Pydantic-схему
        normalized_metadata = {
            "source_file": file_path.name,
            "domain": str(raw_metadata.get("domain", "general")).lower(),
            "access": str(raw_metadata.get("access", "internal")).lower(),
            "status": str(raw_metadata.get("status", "active")).lower(),
            "audience": raw_metadata.get("audience", ["employees"]),
            "linked_docs": linked_docs
        }

        return RawDocument(
            page_content=content,
            metadata=normalized_metadata
        )

    def load_all_documents(self) -> list[RawDocument]:
        """Итерирует по директории и собирает все валидные файлы .md."""
        documents = []
        if not self.data_dir.exists():
            print(f"[-] Директория {self.data_dir} не обнаружена. Создаю пустую структуру...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return documents

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = Path(root) / file
                    try:
                        doc = self.parse_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        print(f"[!] Ошибка парсинга файла {file}: {e}")
        
        return documents

if __name__ == "__main__":
    # Быстрый смоук-тест компонента
    parser = DocumentParser()
    docs = parser.load_all_documents()
    print(f"[+] Шаг 1 завершен. Успешно загружено документов: {len(docs)}")
    