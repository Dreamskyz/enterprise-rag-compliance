"""检查 BM25 Retrieval 的 ACL 行为。"""

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)


QUERY = "内部模型安全规范"


def build_test_chunks() -> list[
    KnowledgeChunk
]:
    """构造三个权限级别的测试数据。"""

    return [
        KnowledgeChunk(
            chunk_id="acl_public_chunk",
            document_id="acl_test_doc",
            title="公开 AI 安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第一章",
            chapter_title="公开规范",
            article_number="第一条",
            content=(
                "这是公开的内部模型安全规范说明。"
            ),
            retrieval_text=(
                "公开 AI 安全规范\n"
                "第一章 公开规范\n"
                "第一条\n"
                "这是公开的内部模型安全规范说明。"
            ),
            source_url="https://example.com/public",
            access_level="public",
            chunk_index=0,
            content_hash="acl-public-hash",
        ),

        KnowledgeChunk(
            chunk_id="acl_developer_chunk",
            document_id="acl_test_doc",
            title="开发人员模型安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第二章",
            chapter_title="开发规范",
            article_number="第二条",
            content=(
                "这是仅允许开发人员访问的"
                "内部模型安全规范。"
            ),
            retrieval_text=(
                "开发人员模型安全规范\n"
                "第二章 开发规范\n"
                "第二条\n"
                "这是仅允许开发人员访问的"
                "内部模型安全规范。"
            ),
            source_url=(
                "https://example.com/developer"
            ),
            access_level="developer",
            chunk_index=1,
            content_hash="acl-developer-hash",
        ),

        KnowledgeChunk(
            chunk_id="acl_admin_chunk",
            document_id="acl_test_doc",
            title="管理员机密模型安全规范",
            document_type="internal_policy",
            language="zh-CN",
            version="1",
            chapter_number="第三章",
            chapter_title="管理员规范",
            article_number="第三条",
            content=(
                "这是仅管理员可以访问的"
                "机密内部模型安全规范。"
            ),
            retrieval_text=(
                "管理员机密模型安全规范\n"
                "第三章 管理员规范\n"
                "第三条\n"
                "这是仅管理员可以访问的"
                "机密内部模型安全规范。"
            ),
            source_url="https://example.com/admin",
            access_level="admin",
            chunk_index=2,
            content_hash="acl-admin-hash",
        ),
    ]


def print_results(
    role: UserRole,
    results: list,
) -> None:
    """打印角色对应的 BM25 结果。"""

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
        candidate = result.candidate

        print(
            f"Top {rank} | "
            f"{candidate.chunk_id} | "
            f"access={candidate.access_level} | "
            f"score={result.score:.4f}"
        )


def main() -> None:
    """
    验证：

        guest
        → public

        developer
        → public + developer

        admin
        → public + developer + admin
    """

    print("=" * 80)
    print("BM25 ACL Retrieval Check")
    print("=" * 80)

    chunks = build_test_chunks()

    retriever = BM25Retriever(
        chunks=chunks
    )

    guest_results = retriever.search(
        query=QUERY,
        top_k=10,
        role=UserRole.GUEST,
    )

    developer_results = retriever.search(
        query=QUERY,
        top_k=10,
        role=UserRole.DEVELOPER,
    )

    admin_results = retriever.search(
        query=QUERY,
        top_k=10,
        role=UserRole.ADMIN,
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
    # ACL Assertions
    # --------------------------------------------------

    guest_levels = {
        result.candidate.access_level
        for result in guest_results
    }

    developer_levels = {
        result.candidate.access_level
        for result in developer_results
    }

    admin_levels = {
        result.candidate.access_level
        for result in admin_results
    }

    assert guest_levels == {
        "public",
    }

    assert developer_levels == {
        "public",
        "developer",
    }

    assert admin_levels == {
        "public",
        "developer",
        "admin",
    }

    assert len(
        guest_results
    ) == 1

    assert len(
        developer_results
    ) == 2

    assert len(
        admin_results
    ) == 3

    print()
    print(
        "✅ BM25 ACL-aware Retrieval "
        "验证通过"
    )


if __name__ == "__main__":
    main()