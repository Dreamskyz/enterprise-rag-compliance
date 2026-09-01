"""测试 RerankedRetriever 的纯业务逻辑。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.retrieval.models import (
    HybridSearchResult,
    RetrievalCandidate,
)
from enterprise_rag.retrieval.reranked import (
    RerankedRetriever,
)


def make_candidate(
    chunk_id: str,
) -> RetrievalCandidate:
    """创建测试用 Candidate。"""

    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc",
        title="测试文档",
        document_type="regulation",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="测试章节",
        article_number="第一条",
        content=f"{chunk_id} 正文",
        retrieval_text=f"{chunk_id} 检索文本",
        source_url="https://example.com",
        access_level="public",
        chunk_index=0,
        content_hash="hash",
    )


class FakeHybridRetriever:
    """
    固定返回 Hybrid Candidate，
    同时记录收到的 AccessContext。
    """

    def __init__(self) -> None:
        self.last_access_context: (
            AccessContext | None
        ) = None

    def search(
        self,
        query: str,
        top_k: int,
        access_context: AccessContext,
    ) -> list[HybridSearchResult]:
        self.last_access_context = (
            access_context
        )

        candidates = [
            make_candidate("a"),
            make_candidate("b"),
            make_candidate("c"),
        ]

        results = [
            HybridSearchResult(
                candidate=candidates[0],
                rrf_score=0.03,
                dense_rank=1,
                bm25_rank=2,
                dense_score=0.8,
                bm25_score=7.0,
            ),
            HybridSearchResult(
                candidate=candidates[1],
                rrf_score=0.02,
                dense_rank=2,
                bm25_rank=1,
                dense_score=0.7,
                bm25_score=8.0,
            ),
            HybridSearchResult(
                candidate=candidates[2],
                rrf_score=0.01,
                dense_rank=3,
                bm25_rank=3,
                dense_score=0.6,
                bm25_score=6.0,
            ),
        ]

        return results[:top_k]


class FakeRerankerService:
    """
    固定返回：

        a -> 1
        b -> 5
        c -> 3
    """

    def compute_scores(
        self,
        query: str,
        passages: list[str],
    ) -> list[float]:
        return [
            1.0,
            5.0,
            3.0,
        ][:len(passages)]


def build_retriever(
) -> tuple[
    RerankedRetriever,
    FakeHybridRetriever,
]:
    """创建测试用 RerankedRetriever。"""

    hybrid = FakeHybridRetriever()

    retriever = RerankedRetriever(
        hybrid_retriever=hybrid,
        reranker_service=(
            FakeRerankerService()
        ),
        candidate_top_k=3,
    )

    return retriever, hybrid


def test_reranker_reorders_hybrid_results() -> None:
    """Reranker 应重新排序 Hybrid Candidate。"""

    retriever, _ = build_retriever()

    results = retriever.search(
        query="测试问题",
        top_k=3,
    )

    assert [
        result.candidate.chunk_id
        for result in results
    ] == [
        "b",
        "c",
        "a",
    ]


def test_reranker_preserves_original_rrf_rank() -> None:
    """精排后仍应保留原始 RRF Rank。"""

    retriever, _ = build_retriever()

    results = retriever.search(
        query="测试问题",
        top_k=3,
    )

    top1 = results[0]

    assert (
        top1.candidate.chunk_id
        == "b"
    )

    assert top1.original_rank == 2

    assert top1.rerank_score == 5.0


def test_reranker_rejects_invalid_top_k() -> None:
    """最终 top_k 不能超过候选池。"""

    retriever, _ = build_retriever()

    try:
        retriever.search(
            query="测试问题",
            top_k=4,
        )
    except ValueError as exc:
        assert "candidate_top_k" in str(
            exc
        )
    else:
        raise AssertionError(
            "预期应抛出 ValueError"
        )


def test_reranker_propagates_access_context() -> None:
    """
    RerankedRetriever 必须将权限上下文
    继续传给 HybridRetriever。
    """

    retriever, hybrid = (
        build_retriever()
    )

    context = AccessContext(
        role=UserRole.ADMIN
    )

    retriever.search(
        query="测试问题",
        top_k=3,
        access_context=context,
    )

    assert (
        hybrid.last_access_context
        == context
    )


def test_reranker_default_access_is_guest() -> None:
    """
    RerankedRetriever 未收到 AccessContext 时，
    必须默认 guest。
    """

    retriever, hybrid = (
        build_retriever()
    )

    retriever.search(
        query="测试问题",
        top_k=3,
    )

    assert (
        hybrid.last_access_context
        is not None
    )

    assert (
        hybrid.last_access_context.role
        == UserRole.GUEST
    )