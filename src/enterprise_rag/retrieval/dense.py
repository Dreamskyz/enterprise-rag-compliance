"""基于 BGE-M3 + Qdrant 的 Dense Retriever。"""

from qdrant_client import QdrantClient

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.acl.qdrant_filter import (
    build_access_filter,
)
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

    当前职责：

    1. Query → BGE-M3 Dense Vector；
    2. 根据 UserRole 构造 ACL Payload Filter；
    3. 在 Qdrant 授权数据空间中执行 Dense Search；
    4. 返回统一 DenseSearchResult。
    """

    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedding_service = (
            embedding_service
        )

        self.collection_name = (
            collection_name
        )

        self.client = QdrantClient(
            url=qdrant_url,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        role: UserRole = UserRole.GUEST,
    ) -> list[DenseSearchResult]:
        """
        执行 ACL-aware Dense Retrieval。

        参数：
            query:
                用户自然语言问题。

            top_k:
                返回前 K 个候选。

            role:
                当前请求用户角色。

                默认：
                    guest

                即默认只允许访问：
                    public
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        # --------------------------------------------------
        # 1. Query Embedding
        # --------------------------------------------------

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        # --------------------------------------------------
        # 2. ACL → Qdrant Filter
        # --------------------------------------------------

        access_filter = (
            build_access_filter(
                role
            )
        )

        # --------------------------------------------------
        # 3. Qdrant ACL-aware Search
        # --------------------------------------------------

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.astype(
                "float32"
            ).tolist(),
            query_filter=access_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # --------------------------------------------------
        # 4. Qdrant Result → Business Result
        # --------------------------------------------------

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