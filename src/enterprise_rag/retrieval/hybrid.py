"""Dense + BM25 + RRF Hybrid Retriever。"""

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
    Hybrid Retriever。

    当前流程：

        Query
          ├─ Dense Retriever
          └─ BM25 Retriever
                ↓
               RRF
                ↓
           Hybrid Top-K
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
    ) -> list[HybridSearchResult]:
        """
        执行 Hybrid Retrieval。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=self.dense_top_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=self.bm25_top_k,
            )
        )

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )