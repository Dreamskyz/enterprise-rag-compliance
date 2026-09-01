"""Dense + BM25 + RRF Hybrid Retriever。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)
from enterprise_rag.retrieval.dense import (
    DenseRetriever,
)
from enterprise_rag.retrieval.models import (
    HybridSearchResult,
)
from enterprise_rag.retrieval.rrf import (
    RRF_K,
    reciprocal_rank_fusion,
)


class HybridRetriever:
    """
    ACL-aware Hybrid Retriever。

    当前流程：

        AccessContext
              │
              ├───────────────┐
              ▼               ▼
        Dense Retrieval    BM25 Retrieval
        Qdrant Filter      Authorized Corpus
              │               │
              └───────┬───────┘
                      ▼
                     RRF
                      │
                      ▼
                Hybrid Results

    注意：

    ACL 必须在 Dense / BM25 Candidate Generation
    之前生效。

    RRF 本身不负责权限过滤，
    它只融合已经授权的候选结果。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = RRF_K,
    ) -> None:
        """
        初始化 Hybrid Retriever。

        参数：
            dense_retriever:
                BGE-M3 + Qdrant Dense Retriever。

            bm25_retriever:
                Jieba + BM25 Retriever。

            dense_top_k:
                Dense 分支召回数量。

            bm25_top_k:
                BM25 分支召回数量。

            rrf_k:
                Reciprocal Rank Fusion 平滑参数。
        """

        if dense_top_k <= 0:
            raise ValueError(
                "dense_top_k 必须大于 0"
            )

        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k 必须大于 0"
            )

        self.dense_retriever = (
            dense_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.dense_top_k = (
            dense_top_k
        )

        self.bm25_top_k = (
            bm25_top_k
        )

        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        access_context: AccessContext | None = None,
    ) -> list[HybridSearchResult]:
        """
        执行 ACL-aware Hybrid Retrieval。

        参数：
            query:
                用户自然语言问题。

            top_k:
                RRF 融合后最多返回数量。

            access_context:
                当前请求的 ACL Context。

                如果调用方没有提供，
                默认使用 guest，
                遵循最小权限原则。
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
        # 默认最小权限：
        #
        # 没有 AccessContext
        # → guest
        # → public only
        # --------------------------------------------------

        if access_context is None:
            access_context = AccessContext(
                role=UserRole.GUEST
            )

        role = access_context.role

        # --------------------------------------------------
        # Dense：
        #
        # role
        # ↓
        # Qdrant Payload Filter
        # ↓
        # Authorized Dense Candidates
        # --------------------------------------------------

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=self.dense_top_k,
                role=role,
            )
        )

        # --------------------------------------------------
        # BM25：
        #
        # role
        # ↓
        # Role-specific Corpus
        # ↓
        # Authorized BM25 Candidates
        # --------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=self.bm25_top_k,
                role=role,
            )
        )

        # --------------------------------------------------
        # RRF 只负责融合两个已经授权的候选集合。
        #
        # ACL 不应该在这里才开始执行。
        # --------------------------------------------------

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )