"""检查 Hybrid Retrieval + BGE Reranker。"""

from pathlib import Path

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
from enterprise_rag.retrieval.reranked import (
    RerankedRetriever,
)


QUERIES = [
    (
        "生成式人工智能服务"
        "处理训练数据需要遵守什么规定？"
    ),
    "数据标注质量评估",
]


def main() -> None:
    """
    检查完整检索链：

        Dense + BM25
             ↓
            RRF
             ↓
        Hybrid Top20
             ↓
        BGE Reranker
             ↓
          Final Top5
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
    print("Reranked Retrieval Check")
    print("=" * 80)

    print(
        "Chunk count:",
        len(chunks),
    )

    # --------------------------------------------------
    # 1. 初始化 Embedding
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    # --------------------------------------------------
    # 2. Dense
    # --------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_service=embedding_service
    )

    # --------------------------------------------------
    # 3. BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # --------------------------------------------------
    # 4. Hybrid RRF
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=20,
        bm25_top_k=20,
        rrf_k=60,
    )

    # --------------------------------------------------
    # 5. Reranker
    # --------------------------------------------------

    reranker_service = (
        BGERerankerService()
    )

    reranked_retriever = (
        RerankedRetriever(
            hybrid_retriever=(
                hybrid_retriever
            ),
            reranker_service=(
                reranker_service
            ),
            candidate_top_k=20,
        )
    )

    # --------------------------------------------------
    # 6. 两条真实 Query
    # --------------------------------------------------

    for query in QUERIES:
        results = (
            reranked_retriever.search(
                query=query,
                top_k=5,
            )
        )

        print()
        print("=" * 80)

        print(
            "Query:",
            query,
        )

        print("=" * 80)

        for rank, result in enumerate(
            results,
            start=1,
        ):
            candidate = (
                result.candidate
            )

            print()
            print(
                f"Top {rank}"
            )

            print(
                "Rerank Score:",
                round(
                    result.rerank_score,
                    4,
                ),
            )

            print(
                "Original RRF Rank:",
                result.original_rank,
            )

            print(
                "RRF Score:",
                round(
                    result.rrf_score,
                    6,
                ),
            )

            print(
                "Dense Rank:",
                result.dense_rank,
            )

            print(
                "BM25 Rank:",
                result.bm25_rank,
            )

            print(
                "Chunk ID:",
                candidate.chunk_id,
            )

            print(
                "Title:",
                candidate.title,
            )

            print(
                "Article:",
                candidate.article_number,
            )

            print(
                "Content:",
                candidate.content[:180],
            )

            print("-" * 80)

    print()
    print(
        "✅ Hybrid + Reranker "
        "完整检索链运行正常"
    )


if __name__ == "__main__":
    main()