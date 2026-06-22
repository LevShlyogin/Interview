from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")
collection_name = "brusnika_knowledge"

print("[🔍] Ищу сгенерированные описания таблиц в Qdrant...\n")

# Ищем чанки, у которых мы проставили тип "table_summary"
results = client.scroll(
    collection_name=collection_name,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchValue(value="table_summary")
            )
        ]
    ),
    limit=5,
    with_payload=True
)

points = results[0]

if not points:
    print("❌ Таблицы не найдены! Либо в .md файлах нет таблиц, либо пайплайн не сработал.")
else:
    print(f"✅ Найдено таблиц: {len(points)} (показываю первые 5)\n")
    for point in points:
        print(f"📄 Файл: {point.payload.get('source_file')}")
        print(f"📝 Текст от LLM: {point.payload.get('page_content')}")
        print("-" * 50)