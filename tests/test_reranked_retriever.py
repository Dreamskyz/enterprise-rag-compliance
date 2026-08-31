"""测试 RerankedRetriever 的纯业务逻辑。"""

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
    """固定返回 Hybrid Candidate。"""

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[HybridSearchResult]:
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
    固定返回 rerank score。

    顺序对应：
        a -> 1.0
        b -> 5.0
        c -> 3.0
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


def test_reranker_reorders_hybrid_results() -> None:
    """
    Reranker 应根据 rerank_score
    改变原来的 RRF 排名。
    """

    retriever = RerankedRetriever(
        hybrid_retriever=FakeHybridRetriever(),
        reranker_service=FakeRerankerService(),
        candidate_top_k=3,
    )

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
    """
    精排后仍应保留原始 RRF Rank，
    方便 Debug 和评测。
    """

    retriever = RerankedRetriever(
        hybrid_retriever=FakeHybridRetriever(),
        reranker_service=FakeRerankerService(),
        candidate_top_k=3,
    )

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
    """top_k 不能大于候选池大小。"""

    retriever = RerankedRetriever(
        hybrid_retriever=FakeHybridRetriever(),
        reranker_service=FakeRerankerService(),
        candidate_top_k=3,
    )

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