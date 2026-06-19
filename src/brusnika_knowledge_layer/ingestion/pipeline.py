from brusnika_knowledge_layer.ingestion.parser import DocumentParser
from brusnika_knowledge_layer.ingestion.extractor import TableExtractor
from brusnika_knowledge_layer.ingestion.splitter import HierarchicalSplitter

def test_ingestion_pipeline():
    print("[1] Инициализация пайплайна...")
    parser = DocumentParser()
    extractor = TableExtractor()
    splitter = HierarchicalSplitter()

    print("[2] Загрузка документов (Этап 1)...")
    docs = parser.load_all_documents()
    
    if not docs:
        print("[-] Документы не найдены. Проверьте папку data/knowledge_base")
        return

    # Для теста возьмем первый загруженный документ
    target_doc = docs[0]
    print(f"[+] Выбран документ для теста: {target_doc.metadata['source_file']}")
    print(f"    Связанные документы (linked_docs): {target_doc.metadata['linked_docs']}")

    print("\n[3] Изоляция таблиц (Этап 2)...")
    processed_doc, tables = extractor.process_document(target_doc)
    print(f"[+] Найдено таблиц: {len(tables)}")
    if tables:
        print(f"    ID первой таблицы: {tables[0]['table_id']}")

    print("\n[4] Иерархический сплиттинг (Этап 3)...")
    chunks = splitter.split_document(processed_doc, tables)
    print(f"[+] Документ нарезан на {len(chunks)} чанков.")

    if chunks:
        print("\n[5] Инспекция первого чанка (Финальный Payload для Qdrant):")
        first_chunk = chunks[0]
        
        print("="*50)
        print(f"CHUNK ID:      {first_chunk.chunk_id}")
        print(f"SOURCE:        {first_chunk.source_file}")
        print(f"HEADERS:       H1: {first_chunk.header_1} | H2: {first_chunk.header_2} | H3: {first_chunk.header_3}")
        print(f"ACCESS LEVEL:  {first_chunk.access}")
        print(f"CHUNK TYPE:    {first_chunk.chunk_type}")
        print("-" * 50)
        print("CONTEXTUAL TEXT (То, что пойдет в Dense & Sparse векторы):")
        print(first_chunk.contextual_text)
        print("="*50)

if __name__ == "__main__":
    test_ingestion_pipeline()