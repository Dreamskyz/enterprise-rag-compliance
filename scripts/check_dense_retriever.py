"""检查正式 DenseRetriever 的 Qdrant 检索结果。"""

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.retrieval.dense import (
    DenseRetriever,
)


QUERY = (
    "生成式人工智能服务"
    "处理训练数据需要遵守什么规定？"
)


def main() -> None:
    """
    使用真实 Query 检查：

    BGE-M3
        ↓
    Query Vector
        ↓
    Qdrant
        ↓
    Dense Top 5

    并验证之前 Brute-force Top 1
    对应的第七条是否仍然排在第一。
    """

    print("=" * 80)
    print("Dense Retriever Check")
    print("=" * 80)

    print(
        "Query:",
        QUERY,
    )

    # 模型只初始化一次，
    # 然后注入 DenseRetriever。
    embedding_service = (
        BGEEmbeddingService()
    )

    retriever = DenseRetriever(
        embedding_service=embedding_service
    )

    results = retriever.search(
        query=QUERY,
        top_k=5,
    )

    print()
    print(
        "Result count:",
        len(results),
    )

    print()
    print("=" * 80)
    print("Qdrant Dense Retrieval Top 5")
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
            "Score:",
            round(
                result.score,
                4,
            ),
        )

        print(
            "Chunk ID:",
            result.chunk_id,
        )

        print(
            "Title:",
            result.title,
        )

        print(
            "Chapter:",
            result.chapter_number,
            result.chapter_title,
        )

        print(
            "Article:",
            result.article_number,
        )

        print(
            "Access Level:",
            result.access_level,
        )

        print(
            "Content:",
            result.content[:200],
        )

        print("-" * 80)

    # --------------------------------------------------
    # Sanity Check
    # --------------------------------------------------

    if not results:
        raise RuntimeError(
            "Dense Retriever 没有返回任何结果"
        )

    top1 = results[0]

    if (
        top1.document_id
        != "cn_genai_interim_2023"
        or top1.article_number
        != "第七条"
    ):
        raise RuntimeError(
            "Dense Retrieval Top 1 "
            "与预期基线不一致："
            f"{top1.chunk_id}"
        )

    print()
    print(
        "✅ Qdrant Dense Retrieval "
        "与当前 Brute-force 基线一致"
    )


if __name__ == "__main__":
    main()