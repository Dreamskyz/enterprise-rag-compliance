"""检查正式 BGEEmbeddingService 和基础 Dense Retrieval。"""

from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)


QUERY = (
    "生成式人工智能服务"
    "处理训练数据需要遵守什么规定？"
)


def main() -> None:
    """
    验证：

    1. BGEEmbeddingService 可以处理真实 Chunk；
    2. 文档向量维度正确；
    3. Query 向量维度正确；
    4. 对全部 Chunk 做一次暴力 Dense Retrieval；
    5. 打印 Top 5 结果。
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
    print("BGEEmbeddingService Check")
    print("=" * 80)

    print(
        "Chunk count:",
        len(chunks),
    )

    print(
        "Query:",
        QUERY,
    )

    # 初始化正式 Embedding Service。
    service = BGEEmbeddingService()

    # --------------------------------------------------
    # 1. 先验证三个真实 Chunk
    # --------------------------------------------------

    sample_texts = [
        chunk.retrieval_text
        for chunk in chunks[:3]
    ]

    sample_vectors = (
        service.embed_documents(
            sample_texts,
            batch_size=3,
        )
    )

    query_vector = service.embed_query(
        QUERY
    )

    print()
    print(
        "Sample document vectors:",
        sample_vectors.shape,
    )

    print(
        "Query vector:",
        query_vector.shape,
    )

    print(
        "Dimension:",
        service.dimension,
    )

    # --------------------------------------------------
    # 2. 对全部 Chunk 生成 Dense Embedding
    # --------------------------------------------------

    all_texts = [
        chunk.retrieval_text
        for chunk in chunks
    ]

    document_vectors = (
        service.embed_documents(
            all_texts,
            batch_size=8,
        )
    )

    print()
    print(
        "All document vectors:",
        document_vectors.shape,
    )

    # --------------------------------------------------
    # 3. Brute-force Dense Retrieval
    # --------------------------------------------------
    #
    # document_vectors:
    #     (49, 1024)
    #
    # query_vector:
    #     (1024,)
    #
    # 矩阵乘法后：
    #     scores.shape == (49,)
    #
    # 每一个 score 对应一个 Chunk 的语义相似度。

    scores = (
        document_vectors
        @ query_vector
    )

    # argsort 默认从小到大。
    #
    # [::-1]
    # 将排序反转为从大到小。
    #
    # [:5]
    # 最后取得分最高的前 5 个 Chunk。
    top_indices = scores.argsort()[
        ::-1
    ][:5]

    print()
    print("=" * 80)
    print("Dense Retrieval Top 5")
    print("=" * 80)

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        # NumPy index 转普通 int，
        # 后面访问 Python list 更明确。
        chunk_index = int(index)

        chunk = chunks[
            chunk_index
        ]

        score = float(
            scores[chunk_index]
        )

        print()
        print(
            f"Top {rank}"
        )

        print(
            "Score:",
            round(score, 4),
        )

        print(
            "Title:",
            chunk.title,
        )

        print(
            "Chapter:",
            chunk.chapter_number,
            chunk.chapter_title,
        )

        print(
            "Article:",
            chunk.article_number,
        )

        print(
            "Content:",
            chunk.content[:200],
        )

        print("-" * 80)


if __name__ == "__main__":
    main()