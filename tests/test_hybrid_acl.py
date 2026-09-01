"""测试 HybridRetriever 是否正确传播 ACL Context。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.retrieval.hybrid import (
    HybridRetriever,
)


class FakeDenseRetriever:
    """记录 Hybrid 传入的角色。"""

    def __init__(self) -> None:
        self.last_role: UserRole | None = None

    def search(
        self,
        query: str,
        top_k: int,
        role: UserRole,
    ) -> list:
        self.last_role = role

        # 本测试只验证 role propagation，
        # 不验证 RRF 排序。
        return []


class FakeBM25Retriever:
    """记录 Hybrid 传入的角色。"""

    def __init__(self) -> None:
        self.last_role: UserRole | None = None

    def search(
        self,
        query: str,
        top_k: int,
        role: UserRole,
    ) -> list:
        self.last_role = role

        return []


def test_hybrid_propagates_developer_role() -> None:
    """
    developer AccessContext
    必须同时传给 Dense 和 BM25。
    """

    dense = FakeDenseRetriever()
    bm25 = FakeBM25Retriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    retriever.search(
        query="测试问题",
        access_context=AccessContext(
            role=UserRole.DEVELOPER
        ),
    )

    assert (
        dense.last_role
        == UserRole.DEVELOPER
    )

    assert (
        bm25.last_role
        == UserRole.DEVELOPER
    )


def test_hybrid_default_access_is_guest() -> None:
    """
    没有 AccessContext 时，
    Hybrid 必须默认 guest。

    这是 Least Privilege。
    """

    dense = FakeDenseRetriever()
    bm25 = FakeBM25Retriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    retriever.search(
        query="测试问题",
    )

    assert (
        dense.last_role
        == UserRole.GUEST
    )

    assert (
        bm25.last_role
        == UserRole.GUEST
    )