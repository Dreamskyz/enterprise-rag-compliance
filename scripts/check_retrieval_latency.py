"""检查完整 Retrieval Pipeline 各阶段延迟。"""

from pathlib import Path
from time import perf_counter

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)
from enterprise_rag.reranking.bge_reranker import (
    BGERerankerService,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)
from enterprise_rag.retrieval.dense import (
    DenseRetriever,
)
from enterprise_rag.retrieval.hybrid import (
    HybridRetriever,
)
from enterprise_rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)


QUERY = (
    "生成式人工智能服务"
    "处理训练数据需要遵守什么规定？"
)


def elapsed_ms(
    start: float,
    end: float,
) -> float:
    """将秒转换为毫秒。"""

    return (
        end - start
    ) * 1000


def main() -> None:
    """
    分阶段观测：

    1. Dense Retrieval
    2. BM25 Retrieval
    3. RRF Fusion
    4. Reranker

    注意：
        模型加载时间单独记录，
        不混入单次 Query latency。
    """

    chunks = read_chunks_jsonl(
        Path(
            "data/processed/chunks.jsonl"
        )
    )

    if not chunks:
        raise RuntimeError(
            "chunks.jsonl 中没有 Chunk"
        )

    print("=" * 80)
    print("Retrieval Latency Check")
    print("=" * 80)

    print(
        "Query:",
        QUERY,
    )

    # --------------------------------------------------
    # 1. 初始化阶段
    # --------------------------------------------------

    init_start = perf_counter()

    embedding_service = (
        BGEEmbeddingService()
    )

    dense_retriever = DenseRetriever(
        embedding_service=embedding_service
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    reranker_service = (
        BGERerankerService()
    )

    init_end = perf_counter()

    print()
    print(
        "Initialization:",
        f"{elapsed_ms(init_start, init_end):.2f} ms",
    )

    # --------------------------------------------------
    # 2. Dense
    # --------------------------------------------------

    dense_start = perf_counter()

    dense_results = (
        dense_retriever.search(
            query=QUERY,
            top_k=20,
        )
    )

    dense_end = perf_counter()

    # --------------------------------------------------
    # 3. BM25
    # --------------------------------------------------

    bm25_start = perf_counter()

    bm25_results = (
        bm25_retriever.search(
            query=QUERY,
            top_k=20,
        )
    )

    bm25_end = perf_counter()

    # --------------------------------------------------
    # 4. RRF
    # --------------------------------------------------

    rrf_start = perf_counter()

    hybrid_results = (
        reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=60,
            top_k=20,
        )
    )

    rrf_end = perf_counter()

    # --------------------------------------------------
    # 5. Rerank
    # --------------------------------------------------

    passages = [
        result.candidate.retrieval_text
        for result in hybrid_results
    ]

    rerank_start = perf_counter()

    rerank_scores = (
        reranker_service.compute_scores(
            query=QUERY,
            passages=passages,
        )
    )

    rerank_end = perf_counter()

    if len(rerank_scores) != len(
        hybrid_results
    ):
        raise RuntimeError(
            "Reranker Score 数量异常"
        )

    # --------------------------------------------------
    # 6. 输出
    # --------------------------------------------------

    dense_ms = elapsed_ms(
        dense_start,
        dense_end,
    )

    bm25_ms = elapsed_ms(
        bm25_start,
        bm25_end,
    )

    rrf_ms = elapsed_ms(
        rrf_start,
        rrf_end,
    )

    rerank_ms = elapsed_ms(
        rerank_start,
        rerank_end,
    )

    total_query_ms = (
        dense_ms
        + bm25_ms
        + rrf_ms
        + rerank_ms
    )

    print()
    print("=" * 80)
    print("Latency")
    print("=" * 80)

    print(
        "Dense:",
        f"{dense_ms:.2f} ms",
    )

    print(
        "BM25:",
        f"{bm25_ms:.2f} ms",
    )

    print(
        "RRF:",
        f"{rrf_ms:.2f} ms",
    )

    print(
        "Reranker:",
        f"{rerank_ms:.2f} ms",
    )

    print(
        "Total query pipeline:",
        f"{total_query_ms:.2f} ms",
    )

    print()
    print(
        "✅ Retrieval latency 检查完成"
    )


if __name__ == "__main__":
    main()