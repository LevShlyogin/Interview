from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """Боевой интерфейс для работы с Qdrant: Hybrid Search + Reranking (Cross-Encoder)."""
    
    def __init__(self, collection_name: str = "brusnika_knowledge"):
        self.collection_name = collection_name
        self.client = QdrantClient(url="http://localhost:6333", timeout=10.0)
        
        print("    [Retriever] Загрузка моделей векторизации...")
        self.dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        print("    [Retriever] Загрузка модели Reranker (Russian MSMARCO)...")
        # Используем легкую и быструю русскую модель (~700 МБ) для защиты от OOM
        self.reranker = CrossEncoder("DiTy/cross-encoder-russian-msmarco", max_length=512)

    def search(self, query: str, domain: str, access_level: str, final_limit: int = 5) -> list[dict[str, Any]]:
        print("    [Qdrant] Векторизация вопроса...")
        
        dense_query = next(iter(self.dense_model.embed([query])))
        sparse_query = next(iter(self.sparse_model.embed([query])))

        # Пользователь всегда имеет доступ к 'public' + к своему уровню
        allowed_access = ["public", access_level]

        filter_conditions = [
            models.FieldCondition(
                key="access", 
                match=models.MatchAny(any=allowed_access)
            )
        ]
        
        # --- ВАЖНОЕ ИЗМЕНЕНИЕ: Мультидоменная фильтрация ---
        # Если домен конкретный (например, construction), ищем в нём И в глобальном справочнике (company).
        if domain.lower() not in ["all", "general"]:
            filter_conditions.append(
                models.FieldCondition(
                    key="domain", 
                    match=models.MatchAny(any=[domain.lower(), "company"])
                )
            )
        
        query_filter = models.Filter(must=filter_conditions)

        # 1. ШИРОКИЙ ЗАХВАТ
        fetch_limit = final_limit * 3 
        print(f"    [Qdrant] Извлекаю топ-{fetch_limit} кандидатов для Reranker'а (domain={domain} + company)...")
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_query.tolist(), using="dense", limit=fetch_limit),
                models.Prefetch(query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()), using="sparse", limit=fetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=fetch_limit,
        )

        if not results.points:
            return []

        # 2. ПОДГОТОВКА К РАНЖИРОВАНИЮ
        candidates = []
        docs_for_reranking = [] 
        
        for point in results.points:
            payload = point.payload
            candidates.append({
                "chunk_id": payload.get("chunk_id"),
                "source_file": payload.get("source_file"),
                "content": payload.get("page_content", ""),
                "status": payload.get("status"),
            })
            docs_for_reranking.append([query, payload.get("page_content", "")])

        # 3. ПЕРЕРАНЖИРОВАНИЕ
        print(f"    [Reranker] Читаю тексты и расставляю {len(candidates)} кандидатов по идеальным местам...")
        rerank_scores = self.reranker.predict(docs_for_reranking)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 4. ФИНАЛЬНЫЙ СРЕЗ
        best_chunks = candidates[:final_limit]
        print(f"    [Reranker] Топ-1 документ получил оценку точности: {best_chunks[0]['rerank_score']:.4f}")
        
        return best_chunks