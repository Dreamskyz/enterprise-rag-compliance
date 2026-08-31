"""基于 BGE-M3 + Qdrant 的 Dense Retriever。"""

from qdrant_client import QdrantClient

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.retrieval.models import (
    DenseSearchResult,
    RetrievalCandidate,
)
from enterprise_rag.vectorstore.qdrant_store import (
    COLLECTION_NAME,
    QDRANT_URL,
)


class DenseRetriever:
    """
    Dense 向量检索器。
    """

    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedding_service = embedding_service
        self.collection_name = collection_name

        self.client = QdrantClient(
            url=qdrant_url,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[DenseSearchResult]:
        """
        对 Query 执行 Dense Retrieval。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.astype(
                "float32"
            ).tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: list[
            DenseSearchResult
        ] = []

        for point in response.points:
            payload = point.payload

            if payload is None:
                raise RuntimeError(
                    "Qdrant 检索结果缺少 Payload"
                )

            candidate = RetrievalCandidate(
                chunk_id=str(
                    payload["chunk_id"]
                ),
                document_id=str(
                    payload["document_id"]
                ),
                title=str(
                    payload["title"]
                ),
                document_type=str(
                    payload["document_type"]
                ),
                language=str(
                    payload["language"]
                ),
                version=str(
                    payload["version"]
                ),
                chapter_number=str(
                    payload["chapter_number"]
                ),
                chapter_title=str(
                    payload["chapter_title"]
                ),
                article_number=str(
                    payload["article_number"]
                ),
                content=str(
                    payload["content"]
                ),
                retrieval_text=str(
                    payload["retrieval_text"]
                ),
                source_url=str(
                    payload["source_url"]
                ),
                access_level=str(
                    payload["access_level"]
                ),
                chunk_index=int(
                    payload["chunk_index"]
                ),
                content_hash=str(
                    payload["content_hash"]
                ),
            )

            results.append(
                DenseSearchResult(
                    candidate=candidate,
                    score=float(
                        point.score
                    ),
                )
            )

        return results