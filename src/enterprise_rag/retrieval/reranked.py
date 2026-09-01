"""ACL-aware Hybrid Retrieval + BGE Reranker。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.reranking.bge_reranker import (
    BGERerankerService,
)
from enterprise_rag.retrieval.hybrid import (
    HybridRetriever,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
)


class RerankedRetriever:
    """
    ACL-aware Hybrid + Reranker 两阶段检索器。

    当前流程：

        AccessContext
             │
             ▼
       Hybrid Retriever
        /            \
     Dense           BM25
    ACL Filter     ACL Corpus
        |            /
             RRF
              ↓
        Hybrid Top-K
              ↓
        BGE Reranker
              ↓
         Final Top-K

    Reranker 本身不负责 ACL。

    它只精排 Hybrid Retriever
    已经授权的 Candidate。
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker_service: BGERerankerService,
        candidate_top_k: int = 20,
    ) -> None:
        """
        初始化 Reranked Retriever。

        参数：
            hybrid_retriever:
                ACL-aware Hybrid Retriever。

            reranker_service:
                BGE Cross-Encoder Reranker。

            candidate_top_k:
                送入 Reranker 的 Hybrid
                Candidate 数量。
        """

        if candidate_top_k <= 0:
            raise ValueError(
                "candidate_top_k 必须大于 0"
            )

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.reranker_service = (
            reranker_service
        )

        self.candidate_top_k = (
            candidate_top_k
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        access_context: AccessContext | None = None,
    ) -> list[RerankedSearchResult]:
        """
        执行 ACL-aware Hybrid Retrieval + Rerank。

        参数：
            query:
                用户自然语言问题。

            top_k:
                最终精排返回数量。

            access_context:
                当前请求访问控制上下文。

                未提供时默认 guest，
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

        if top_k > self.candidate_top_k:
            raise ValueError(
                "top_k 不能大于 candidate_top_k"
            )

        if access_context is None:
            access_context = AccessContext(
                role=UserRole.GUEST
            )

        # --------------------------------------------------
        # 1. ACL-aware Hybrid Retrieval
        # --------------------------------------------------

        hybrid_results = (
            self.hybrid_retriever.search(
                query=query,
                top_k=self.candidate_top_k,
                access_context=access_context,
            )
        )

        if not hybrid_results:
            return []

        # --------------------------------------------------
        # 2. 准备 Cross-Encoder 输入
        #
        # 注意：
        # 此时这些 Candidate 已经经过 ACL。
        # --------------------------------------------------

        passages = [
            result.candidate.retrieval_text
            for result in hybrid_results
        ]

        # --------------------------------------------------
        # 3. Query + Passage → Rerank Score
        # --------------------------------------------------

        rerank_scores = (
            self.reranker_service.compute_scores(
                query=query,
                passages=passages,
            )
        )

        if len(
            rerank_scores
        ) != len(
            hybrid_results
        ):
            raise RuntimeError(
                "Reranker Score 数量"
                "与 Candidate 数量不一致"
            )

        # --------------------------------------------------
        # 4. 构建精排结果
        # --------------------------------------------------

        reranked_results: list[
            RerankedSearchResult
        ] = []

        for original_rank, (
            hybrid_result,
            rerank_score,
        ) in enumerate(
            zip(
                hybrid_results,
                rerank_scores,
                strict=True,
            ),
            start=1,
        ):
            reranked_results.append(
                RerankedSearchResult(
                    candidate=(
                        hybrid_result.candidate
                    ),
                    rerank_score=float(
                        rerank_score
                    ),
                    original_rank=original_rank,
                    rrf_score=(
                        hybrid_result.rrf_score
                    ),
                    dense_rank=(
                        hybrid_result.dense_rank
                    ),
                    bm25_rank=(
                        hybrid_result.bm25_rank
                    ),
                )
            )

        # --------------------------------------------------
        # 5. Cross-Encoder 精排
        # --------------------------------------------------

        reranked_results.sort(
            key=lambda result: (
                result.rerank_score
            ),
            reverse=True,
        )

        return reranked_results[
            :top_k
        ]