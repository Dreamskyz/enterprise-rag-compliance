"""检查 Dense + BM25 + RRF Hybrid Retrieval。"""

from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
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


QUERIES = [
    (
        "生成式人工智能服务"
        "处理训练数据需要遵守什么规定？"
    ),
    "数据标注质量评估",
]


def main() -> None:
    """运行两条真实 Query 的 Hybrid Retrieval。"""

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
    print("Hybrid Retriever Check")
    print("=" * 80)

    print(
        "Chunk count:",
        len(chunks),
    )

    embedding_service = (
        BGEEmbeddingService()
    )

    dense_retriever = DenseRetriever(
        embedding_service=embedding_service
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=20,
        bm25_top_k=20,
        rrf_k=60,
    )

    for query in QUERIES:
        results = hybrid_retriever.search(
            query=query,
            top_k=10,
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
            candidate = result.candidate

            print()
            print(
                f"Top {rank}"
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
                "Dense Score:",
                (
                    round(
                        result.dense_score,
                        4,
                    )
                    if result.dense_score
                    is not None
                    else None
                ),
            )

            print(
                "BM25 Score:",
                (
                    round(
                        result.bm25_score,
                        4,
                    )
                    if result.bm25_score
                    is not None
                    else None
                ),
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
                candidate.content[:160],
            )

            print("-" * 80)

    print()
    print(
        "✅ Hybrid RRF Retrieval 运行正常"
    )


if __name__ == "__main__":
    main()