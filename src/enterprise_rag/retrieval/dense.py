"""基于 BGE-M3 + Qdrant 的 Dense Retriever。"""

from collections.abc import Mapping

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
    4. 将 Qdrant Payload 恢复为 RetrievalCandidate；
    5. 返回统一 DenseSearchResult。

    Day 6 开始：

    Payload 可能来自两类文档：

        法规
            → chapter / article

        OWASP / FastAPI / Qdrant
            → section

    所以 Payload → Candidate 的恢复过程
    必须正确处理 nullable metadata。
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

    @staticmethod
    def _read_optional_string(
        payload: Mapping[str, object],
        key: str,
    ) -> str | None:
        """
        从 Qdrant Payload 中读取可空字符串字段。

        为什么不能直接：

            str(payload[key])

        因为：

            str(None)

        得到的是：

            "None"

        而不是真正的：

            None

        同时使用 payload.get(key)，还能够兼容
        Day 6 之前已经存在的旧 Qdrant Payload：

            旧 Payload 中没有 section_title
            旧 Payload 中没有 section_path

        此时直接返回 None，而不会触发 KeyError。
        """

        value = payload.get(key)

        if value is None:
            return None

        return str(value)

    @classmethod
    def _build_candidate_from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> RetrievalCandidate:
        """
        将 Qdrant Payload 恢复为 RetrievalCandidate。

        这是 Vector Store → Retrieval Domain
        之间明确的数据转换边界。

        核心字段仍然使用：

            payload["field"]

        如果缺失则直接失败，
        因为这些属于 Payload Contract 的必需字段。

        可空 / 新增 Metadata 使用安全读取，
        从而兼容旧索引。
        """

        return RetrievalCandidate(
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

            # ----------------------------------------------
            # 法规结构。
            #
            # 技术文档中这些值可能是真正的 None。
            # 不能使用 str(None)。
            # ----------------------------------------------

            chapter_number=(
                cls._read_optional_string(
                    payload,
                    "chapter_number",
                )
            ),

            chapter_title=(
                cls._read_optional_string(
                    payload,
                    "chapter_title",
                )
            ),

            article_number=(
                cls._read_optional_string(
                    payload,
                    "article_number",
                )
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

            # ----------------------------------------------
            # Day 6 通用 Section Metadata。
            #
            # 使用 .get() 语义读取，
            # 因此同时兼容：
            #
            # 1. 新 Payload，有真实 section；
            # 2. 新法规 Payload，值为 None；
            # 3. Day 6 前旧 Payload，字段根本不存在。
            # ----------------------------------------------

            section_title=(
                cls._read_optional_string(
                    payload,
                    "section_title",
                )
            ),

            section_path=(
                cls._read_optional_string(
                    payload,
                    "section_path",
                )
            ),
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

            candidate = (
                self._build_candidate_from_payload(
                    payload
                )
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