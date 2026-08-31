"""构建 Qdrant Dense Vector Index。"""

from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)
from enterprise_rag.vectorstore.qdrant_store import (
    QdrantVectorStore,
)


CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


def main() -> None:
    """
    将 chunks.jsonl：

        retrieval_text
            ↓
        BGE-M3
            ↓
        Dense Vector
            ↓
        Qdrant

    构建为 Dense Vector Index。
    """

    print("=" * 80)
    print("Build Qdrant Dense Vector Index")
    print("=" * 80)

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    if not chunks:
        raise RuntimeError(
            "没有可用于建库的 Chunk"
        )

    print(
        "Chunk count:",
        len(chunks),
    )

    # --------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    texts = [
        chunk.retrieval_text
        for chunk in chunks
    ]

    print()
    print("Generating embeddings...")

    vectors = (
        embedding_service.embed_documents(
            texts,
            batch_size=8,
        )
    )

    print(
        "Embedding shape:",
        vectors.shape,
    )

    # --------------------------------------------------
    # 2. Qdrant
    # --------------------------------------------------

    store = QdrantVectorStore(
        vector_size=embedding_service.dimension
    )

    # 当前第一次正式构建 v1 Collection，
    # 明确允许重建。
    store.create_collection(
        recreate=True
    )

    print()
    print("Collection created.")

    # --------------------------------------------------
    # 3. Upsert
    # --------------------------------------------------

    store.upsert_chunks(
        chunks=chunks,
        vectors=vectors,
    )

    count = (
        store.get_collection_count()
    )

    print()
    print(
        "Qdrant point count:",
        count,
    )

    if count != len(chunks):
        raise RuntimeError(
            "Qdrant Point 数量与 Chunk 数量不一致："
            f"{count} != {len(chunks)}"
        )

    print()
    print(
        "✅ Qdrant Dense Vector Index 构建完成"
    )


if __name__ == "__main__":
    main()