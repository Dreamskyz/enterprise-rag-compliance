"""测试 BM25 Retrieval 的 ACL 行为。"""

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)


def make_chunk(
    chunk_id: str,
    access_level: str,
    content: str,
    chunk_index: int,
) -> KnowledgeChunk:
    """构造 ACL 测试 Chunk。"""

    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="acl_test_doc",
        title="内部模型安全规范",
        document_type="internal_policy",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="安全规范",
        article_number=f"第{chunk_index + 1}条",
        content=content,
        retrieval_text=(
            "内部模型安全规范\n"
            "第一章 安全规范\n"
            f"第{chunk_index + 1}条\n"
            f"{content}"
        ),
        source_url=(
            f"https://example.com/{chunk_id}"
        ),
        access_level=access_level,
        chunk_index=chunk_index,
        content_hash=f"hash-{chunk_id}",
    )


def build_chunks() -> list[
    KnowledgeChunk
]:
    """构造 public / developer / admin 三种数据。"""

    return [
        make_chunk(
            chunk_id="public_chunk",
            access_level="public",
            content=(
                "这是公开的内部模型安全规范。"
            ),
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="developer_chunk",
            access_level="developer",
            content=(
                "这是开发人员内部模型安全规范。"
            ),
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="admin_chunk",
            access_level="admin",
            content=(
                "这是管理员机密内部模型安全规范。"
            ),
            chunk_index=2,
        ),
    ]


def result_levels(
    results: list,
) -> set[str]:
    """提取结果中的 access_level。"""

    return {
        result.candidate.access_level
        for result in results
    }


def test_guest_bm25_only_sees_public() -> None:
    """guest 的 BM25 corpus 只能包含 public。"""

    retriever = BM25Retriever(
        build_chunks()
    )

    results = retriever.search(
        query="内部模型安全规范",
        top_k=10,
        role=UserRole.GUEST,
    )

    assert len(results) == 1

    assert result_levels(
        results
    ) == {
        "public",
    }


def test_developer_bm25_sees_public_and_developer() -> None:
    """
    developer 的 BM25 corpus
    只能包含 public + developer。
    """

    retriever = BM25Retriever(
        build_chunks()
    )

    results = retriever.search(
        query="内部模型安全规范",
        top_k=10,
        role=UserRole.DEVELOPER,
    )

    assert len(results) == 2

    assert result_levels(
        results
    ) == {
        "public",
        "developer",
    }


def test_admin_bm25_sees_all_levels() -> None:
    """admin 可以检索全部测试 Chunk。"""

    retriever = BM25Retriever(
        build_chunks()
    )

    results = retriever.search(
        query="内部模型安全规范",
        top_k=10,
        role=UserRole.ADMIN,
    )

    assert len(results) == 3

    assert result_levels(
        results
    ) == {
        "public",
        "developer",
        "admin",
    }


def test_bm25_default_role_is_guest() -> None:
    """
    调用方未提供 role 时，
    必须遵循最小权限 guest。
    """

    retriever = BM25Retriever(
        build_chunks()
    )

    results = retriever.search(
        query="内部模型安全规范",
        top_k=10,
    )

    assert result_levels(
        results
    ) == {
        "public",
    }