"""检查中文 BM25 Retriever 的真实检索结果。"""

from pathlib import Path

from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)


TRAINING_DATA_QUERY = (
    "生成式人工智能服务"
    "处理训练数据需要遵守什么规定？"
)

LABELING_QUERY = (
    "数据标注质量评估"
)


def print_results(
    query: str,
    results: list,
) -> None:
    """打印 BM25 Top-K 检索结果。"""

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
        print()

        print(
            f"Top {rank}"
        )

        print(
            "BM25 Score:",
            round(
                result.score,
                4,
            ),
        )

        print(
            "Chunk ID:",
            result.candidate.chunk_id,
        )

        print(
            "Title:",
            result.candidate.title,
        )

        print(
            "Chapter:",
            result.candidate.chapter_number,
            result.candidate.chapter_title,
        )

        print(
            "Article:",
            result.candidate.article_number,
        )

        print(
            "Content:",
            result.candidate.content[:200],
        )

        print("-" * 80)


def main() -> None:
    """
    使用两条真实 Query 检查 BM25：

    1. 训练数据相关自然语言 Query；
    2. 具有明显关键词特征的“数据标注质量评估”。
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
    print("BM25 Retriever Check")
    print("=" * 80)

    print(
        "Chunk count:",
        len(chunks),
    )

    retriever = BM25Retriever(
        chunks=chunks
    )

    # --------------------------------------------------
    # Query 1
    # 与 Dense Baseline 使用同一问题。
    # --------------------------------------------------

    training_results = (
        retriever.search(
            query=TRAINING_DATA_QUERY,
            top_k=5,
        )
    )

    print_results(
        TRAINING_DATA_QUERY,
        training_results,
    )

    # --------------------------------------------------
    # Query 2
    # 测试明显关键词型 Query。
    # --------------------------------------------------

    labeling_results = (
        retriever.search(
            query=LABELING_QUERY,
            top_k=5,
        )
    )

    print_results(
        LABELING_QUERY,
        labeling_results,
    )

    # --------------------------------------------------
    # Sanity Check
    # --------------------------------------------------

    if not training_results:
        raise RuntimeError(
            "训练数据 Query 没有返回结果"
        )

    if not labeling_results:
        raise RuntimeError(
            "数据标注 Query 没有返回结果"
        )

    print()
    print(
        "✅ BM25 Sparse Retrieval 运行正常"
    )


if __name__ == "__main__":
    main()