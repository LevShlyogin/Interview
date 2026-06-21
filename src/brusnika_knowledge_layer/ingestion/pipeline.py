from brusnika_knowledge_layer.ingestion.parser import DocumentParser
from brusnika_knowledge_layer.ingestion.extractor import TableExtractor
from brusnika_knowledge_layer.ingestion.splitter import HierarchicalSplitter
from brusnika_knowledge_layer.database.qdrant_manager import QdrantManager

def run_full_ingestion():
    """Полный цикл загрузки всей базы знаний в Qdrant."""
    
    print("[1] Инициализация компонентов пайплайна...")
    parser = DocumentParser()
    extractor = TableExtractor()
    splitter = HierarchicalSplitter()
    
    # Инициализация подключения к БД и моделей векторизации
    db_manager = QdrantManager()

    print("\n[2] Загрузка Markdown-документов с диска (Этап 1)...")
    docs = parser.load_all_documents()
    
    if not docs:
        print("[-] Документы не найдены. Проверьте папку data/knowledge_base")
        return

    print(f"[+] Найдено документов: {len(docs)}")

    all_chunks = []
    
    print("\n[3] Обработка текстов и извлечение таблиц (Этапы 2 и 3)...")
    for doc in docs:
        # 1. Вырезаем таблицы и заменяем на плейсхолдеры
        processed_doc, tables = extractor.process_document(doc)
        
        # 2. Режем документ на умные иерархические чанки
        chunks = splitter.split_document(processed_doc, tables)
        
        # 3. Добавляем чанки текущего документа в общий котел
        all_chunks.extend(chunks)

    print(f"[+] База знаний успешно нарезана на {len(all_chunks)} чанков.")

    if all_chunks:
        print("\n[4] Векторизация (Dense + Sparse) и загрузка в Qdrant (Этап 4)...")
        # Передаем весь массив чанков в менеджер БД
        db_manager.upsert_chunks(all_chunks)
        print("\n[+] 🚀 Пайплайн ингестии успешно завершен! База данных готова к поиску.")
    else:
        print("[-] Не удалось сгенерировать чанки.")

if __name__ == "__main__":
    run_full_ingestion()