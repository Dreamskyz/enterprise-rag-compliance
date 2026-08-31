"""Hybrid Retrieval + BGE Reranker。"""

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
    Hybrid + Reranker 两阶段检索器。

    当前流程：

        Dense Top20
            \
             → RRF Top20
            /
        BM25 Top20

             ↓

        BGE Reranker

             ↓

        Final Top5
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
                已构建的 Dense + BM25 + RRF
                检索器。

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
    ) -> list[RerankedSearchResult]:
        """
        执行 Hybrid Retrieval + Rerank。

        参数：
            query:
                用户问题。

            top_k:
                最终返回的精排结果数量。
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

        # --------------------------------------------------
        # 1. Hybrid Retrieval
        # --------------------------------------------------

        hybrid_results = (
            self.hybrid_retriever.search(
                query=query,
                top_k=self.candidate_top_k,
            )
        )

        if not hybrid_results:
            return []

        # --------------------------------------------------
        # 2. 准备 Cross-Encoder Passage
        # --------------------------------------------------

        passages = [
            result.candidate.retrieval_text
            for result in hybrid_results
        ]

        # --------------------------------------------------
        # 3. Query + Passage → Rerank Scores
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
        # 4. 构建 RerankedSearchResult
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
                    original_rank=(
                        original_rank
                    ),
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
        # 5. 根据 Cross-Encoder 分数重新排序
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