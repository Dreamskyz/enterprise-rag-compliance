"""检查完整 Retrieval Pipeline 的 ACL 行为。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
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
from enterprise_rag.vectorstore.qdrant_store import (
    QdrantVectorStore,
)


ACL_COLLECTION = (
    "acl_pipeline_test_v1"
)

QUERY = "内部模型安全规范"


def build_test_chunks() -> list[
    KnowledgeChunk
]:
    """
    构造三个高度相关但权限不同的 Chunk。

    故意让三条都与 Query 高度相关，
    避免“没召回”其实只是因为语义不相关。
    """

    return [
        KnowledgeChunk(
            chunk_id="pipeline_public_chunk",
            document_id="acl_pipeline_doc",
            title="公开模型安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第一章",
            chapter_title="公开安全要求",
            article_number="第一条",
            content=(
                "内部模型安全规范要求"
                "研发人员遵守基础安全流程。"
            ),
            retrieval_text=(
                "公开模型安全规范\n"
                "第一章 公开安全要求\n"
                "第一条\n"
                "内部模型安全规范要求"
                "研发人员遵守基础安全流程。"
            ),
            source_url=(
                "https://example.com/public"
            ),
            access_level="public",
            chunk_index=0,
            content_hash=(
                "pipeline-public-hash"
            ),
        ),

        KnowledgeChunk(
            chunk_id=(
                "pipeline_developer_chunk"
            ),
            document_id="acl_pipeline_doc",
            title="开发人员内部模型安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第二章",
            chapter_title="开发安全要求",
            article_number="第二条",
            content=(
                "开发人员内部模型安全规范"
                "要求研发阶段执行模型输入输出"
                "安全检查。"
            ),
            retrieval_text=(
                "开发人员内部模型安全规范\n"
                "第二章 开发安全要求\n"
                "第二条\n"
                "开发人员内部模型安全规范"
                "要求研发阶段执行模型输入输出"
                "安全检查。"
            ),
            source_url=(
                "https://example.com/developer"
            ),
            access_level="developer",
            chunk_index=1,
            content_hash=(
                "pipeline-developer-hash"
            ),
        ),

        KnowledgeChunk(
            chunk_id="pipeline_admin_chunk",
            document_id="acl_pipeline_doc",
            title="管理员机密模型安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第三章",
            chapter_title="管理员安全要求",
            article_number="第三条",
            content=(
                "管理员机密内部模型安全规范"
                "包含高权限模型配置与"
                "安全控制要求。"
            ),
            retrieval_text=(
                "管理员机密模型安全规范\n"
                "第三章 管理员安全要求\n"
                "第三条\n"
                "管理员机密内部模型安全规范"
                "包含高权限模型配置与"
                "安全控制要求。"
            ),
            source_url=(
                "https://example.com/admin"
            ),
            access_level="admin",
            chunk_index=2,
            content_hash=(
                "pipeline-admin-hash"
            ),
        ),
    ]


def print_results(
    role: UserRole,
    results: list,
) -> None:
    """打印最终 Reranked 结果。"""

    print()
    print("=" * 80)

    print(
        "Role:",
        role.value,
    )

    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        candidate = (
            result.candidate
        )

        print(
            f"Top {rank}"
        )

        print(
            "Chunk:",
            candidate.chunk_id,
        )

        print(
            "Access:",
            candidate.access_level,
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
            "Dense Rank:",
            result.dense_rank,
        )

        print(
            "BM25 Rank:",
            result.bm25_rank,
        )

        print("-" * 80)


def get_levels(
    results: list,
) -> set[str]:
    """提取最终结果权限集合。"""

    return {
        result.candidate.access_level
        for result in results
    }


def main() -> None:
    """
    验证完整链路：

        ACL Context
            ↓
        Dense ACL
        BM25 ACL
            ↓
           RRF
            ↓
        Reranker
            ↓
        Final Results

    要求：

        guest
        → public only

        developer
        → public + developer

        admin
        → all
    """

    print("=" * 80)
    print(
        "Full ACL Retrieval Pipeline Check"
    )
    print("=" * 80)

    chunks = build_test_chunks()

    # --------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    vectors = (
        embedding_service.embed_documents(
            [
                chunk.retrieval_text
                for chunk in chunks
            ],
            batch_size=3,
        )
    )

    # --------------------------------------------------
    # 2. 创建独立 ACL Test Collection
    # --------------------------------------------------

    store = QdrantVectorStore(
        collection_name=ACL_COLLECTION,
        vector_size=(
            embedding_service.dimension
        ),
    )

    store.create_collection(
        recreate=True
    )

    store.upsert_chunks(
        chunks=chunks,
        vectors=vectors,
    )

    print(
        "Test point count:",
        store.get_collection_count(),
    )

    assert (
        store.get_collection_count()
        == 3
    )

    # --------------------------------------------------
    # 3. Dense
    # --------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        ),
        collection_name=ACL_COLLECTION,
    )

    # --------------------------------------------------
    # 4. BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # --------------------------------------------------
    # 5. Hybrid
    #
    # 测试数据只有 3 条，
    # 因此 top-k 设置大于数据量，
    # 确保不会因为截断造成假阳性。
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=(
            dense_retriever
        ),
        bm25_retriever=(
            bm25_retriever
        ),
        dense_top_k=10,
        bm25_top_k=10,
        rrf_k=60,
    )

    # --------------------------------------------------
    # 6. Reranker
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
            candidate_top_k=10,
        )
    )

    # --------------------------------------------------
    # 7. 三种角色分别查询
    # --------------------------------------------------

    guest_results = (
        reranked_retriever.search(
            query=QUERY,
            top_k=1,
            access_context=AccessContext(
                role=UserRole.GUEST
            ),
        )
    )

    developer_results = (
        reranked_retriever.search(
            query=QUERY,
            top_k=2,
            access_context=AccessContext(
                role=UserRole.DEVELOPER
            ),
        )
    )

    admin_results = (
        reranked_retriever.search(
            query=QUERY,
            top_k=3,
            access_context=AccessContext(
                role=UserRole.ADMIN
            ),
        )
    )

    print_results(
        UserRole.GUEST,
        guest_results,
    )

    print_results(
        UserRole.DEVELOPER,
        developer_results,
    )

    print_results(
        UserRole.ADMIN,
        admin_results,
    )

    # --------------------------------------------------
    # 8. 最终 ACL Assertions
    # --------------------------------------------------

    assert get_levels(
        guest_results
    ) == {
        "public",
    }

    assert get_levels(
        developer_results
    ) == {
        "public",
        "developer",
    }

    assert get_levels(
        admin_results
    ) == {
        "public",
        "developer",
        "admin",
    }

    # --------------------------------------------------
    # 9. 明确验证 0 条越权 Chunk
    # --------------------------------------------------

    guest_unauthorized = [
        result
        for result in guest_results
        if result.candidate.access_level
        not in {
            "public",
        }
    ]

    developer_unauthorized = [
        result
        for result in developer_results
        if result.candidate.access_level
        not in {
            "public",
            "developer",
        }
    ]

    admin_unauthorized = [
        result
        for result in admin_results
        if result.candidate.access_level
        not in {
            "public",
            "developer",
            "admin",
        }
    ]

    assert len(
        guest_unauthorized
    ) == 0

    assert len(
        developer_unauthorized
    ) == 0

    assert len(
        admin_unauthorized
    ) == 0

    print()
    print("=" * 80)

    print(
        "guest unauthorized chunks: 0"
    )

    print(
        "developer unauthorized chunks: 0"
    )

    print(
        "admin unauthorized chunks: 0"
    )

    print()

    print(
        "✅ ACL + Dense + BM25 + RRF + "
        "Reranker 全链路验证通过"
    )


if __name__ == "__main__":
    main()