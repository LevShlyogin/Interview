from typing import List, Dict, Any

class HybridRetriever:
    """Интерфейс для работы с Qdrant (Dense + Sparse + RRF)."""
    
    def __init__(self):
        # В будущем здесь: self.client = QdrantClient(...)
        pass
        
    def search(self, query: str, domain: str, access_level: str) -> List[Dict[str, Any]]:
        """Эмуляция гибридного поиска с жесткими фильтрами Qdrant."""
        print(f"    [Qdrant] Выполняю Hybrid Search...")
        print(f"    [Qdrant] Применены фильтры -> domain: {domain} | access: {access_level}")
        
        # Мокаем ответ базы для нашего вопроса про суточные на стройке
        return [
            {
                "chunk_id": "chk_123",
                "source_file": "business-travel-policy.md",
                "content": "Суточные для поездок на объекты строительства (СГП) составляют 1500 рублей в сутки.",
                "status": "active"
            },
            {
                "chunk_id": "chk_456",
                "source_file": "old-travel-rules.md",
                "content": "Лимит суточных для инженеров - 700 рублей.",
                "status": "archived" # <-- Устаревший документ!
            }
        ]