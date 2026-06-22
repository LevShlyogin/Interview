
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from brusnika_knowledge_layer.schema import ChunkPayload


class QdrantManager:
    def __init__(self, collection_name: str = "brusnika_knowledge", recreate_collection: bool = True):
        self.collection_name = collection_name
        # Подключаемся к локальному Qdrant, поднятому в Docker
        self.client = QdrantClient(url="http://localhost:6333", timeout=60.0)
        
        print("[Qdrant] Инициализация моделей векторизации (может занять минуту при первом запуске)...")
        # Dense модель (семантика)
        self.dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # Sparse модель (лексика/ключевые слова)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Передаем флаг очистки
        self._ensure_collection_exists(recreate_collection)

    def _ensure_collection_exists(self, recreate_collection: bool):
        """Создает коллекцию с поддержкой Hybrid Search. Удаляет старую, если нужно."""
        
        # 1. Сносим старую коллекцию (на 1024 измерения), если она есть
        if recreate_collection and self.client.collection_exists(self.collection_name):
            print(f"[Qdrant] 🧹 Удаляю старую коллекцию '{self.collection_name}' для чистой заливки...")
            self.client.delete_collection(collection_name=self.collection_name)

        # 2. Создаем новую чистую коллекцию на 384 измерения
        if not self.client.collection_exists(self.collection_name):
            print(f"[Qdrant] 🏗 Создаю новую коллекцию: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384, # Размерность для paraphrase-multilingual-MiniLM-L12-v2
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            # Настраиваем индексы для мгновенной фильтрации (Agentic Routing)
            self._create_payload_indices()

    def _create_payload_indices(self):
        """Индексирование полей для ускорения жесткой фильтрации."""
        fields_to_index = ["domain", "access", "status"]
        for field in fields_to_index:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
    def upsert_chunks(self, payloads: list[ChunkPayload]):
        """Векторизует тексты и загружает их в БД мелкими батчами (защита от OOM)."""
        if not payloads:
            return

        total_chunks = len(payloads)
        print(f"[Qdrant] Начинаю безопасную векторизацию и загрузку {total_chunks} чанков...")
        
        # Микро-батч для защиты оперативной памяти от SPLADE
        upload_batch_size = 8 
        
        for i in range(0, total_chunks, upload_batch_size):
            batch_payloads = payloads[i : i + upload_batch_size]
            texts = [p.contextual_text for p in batch_payloads]
            
            # Векторизуем мелкими порциями
            dense_embeddings = list(self.dense_model.embed(texts, batch_size=upload_batch_size))
            sparse_embeddings = list(self.sparse_model.embed(texts, batch_size=upload_batch_size))

            points = []
            for j, payload in enumerate(batch_payloads):
                point_id = payload.chunk_id
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_embeddings[j].tolist(),
                            "sparse": models.SparseVector(
                                indices=sparse_embeddings[j].indices.tolist(),
                                values=sparse_embeddings[j].values.tolist()
                            )
                        },
                        payload=payload.model_dump()
                    )
                )

            # Отправляем этот микро-батч в базу
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"  -> Загружено {min(i + upload_batch_size, total_chunks)} / {total_chunks} чанков...")
            
        print("[Qdrant] ✅ Все данные успешно загружены в базу!")