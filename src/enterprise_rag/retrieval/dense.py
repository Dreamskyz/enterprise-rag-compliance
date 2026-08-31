"""基于 BGE-M3 + Qdrant 的 Dense Retriever。"""

from qdrant_client import QdrantClient

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.retrieval.models import (
    DenseSearchResult,
)
from enterprise_rag.vectorstore.qdrant_store import (
    COLLECTION_NAME,
    QDRANT_URL,
)


class DenseRetriever:
    """
    Dense 向量检索器。

    当前职责：

    1. 将用户 Query 转换为 Dense Embedding；
    2. 向 Qdrant 发起向量相似度检索；
    3. 将 Qdrant 原始结果转换成业务层
       DenseSearchResult。
    """

    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        初始化 Dense Retriever。

        参数：
            embedding_service:
                已初始化的 BGEEmbeddingService。

                由外部传入，而不是 Retriever 内部自己
                再创建一个模型，避免重复加载 BGE-M3。

            qdrant_url:
                Qdrant REST 地址。

            collection_name:
                当前 Dense Index Collection 名称。
        """

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
    ) -> list[DenseSearchResult]:
        """
        对 Query 执行 Dense Retrieval。

        参数：
            query:
                用户自然语言问题。

            top_k:
                返回最相关的前 K 个 Chunk。

        返回：
            按 Dense Score 从高到低排列的
            DenseSearchResult 列表。
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
        # 1. Query → BGE-M3 Dense Vector
        # --------------------------------------------------

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        # --------------------------------------------------
        # 2. Dense Vector → Qdrant Search
        # --------------------------------------------------
        #
        # 当前 Collection：
        #
        # vector size = 1024
        # distance    = Cosine
        #
        # query_points 会返回按相似度排序的 Point。
        #
        # with_payload=True：
        # 要求 Qdrant 将 Chunk metadata 一并返回。
        # 否则这里只能拿到 ID 和 score。

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.astype(
                "float32"
            ).tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # --------------------------------------------------
        # 3. Qdrant Result → Business Result
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

            result = DenseSearchResult(
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

                score=float(
                    point.score
                ),
            )

            results.append(result)

        return results