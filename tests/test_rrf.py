"""测试 Reciprocal Rank Fusion。"""

import pytest

from enterprise_rag.retrieval.models import (
    BM25SearchResult,
    DenseSearchResult,
    RetrievalCandidate,
)
from enterprise_rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)


def make_candidate(
    chunk_id: str,
) -> RetrievalCandidate:
    """创建最小测试 Candidate。"""

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
        content="测试正文",
        retrieval_text="测试检索文本",
        source_url="https://example.com",
        access_level="public",
        chunk_index=0,
        content_hash="hash",
    )


def test_rrf_rewards_candidates_present_in_both_lists() -> None:
    """
    同时出现在 Dense 和 BM25 前列的 Candidate
    应获得更高融合分数。
    """

    candidate_a = make_candidate("a")
    candidate_b = make_candidate("b")
    candidate_c = make_candidate("c")

    dense_results = [
        DenseSearchResult(
            candidate=candidate_a,
            score=0.9,
        ),
        DenseSearchResult(
            candidate=candidate_b,
            score=0.8,
        ),
    ]

    bm25_results = [
        BM25SearchResult(
            candidate=candidate_b,
            score=10.0,
        ),
        BM25SearchResult(
            candidate=candidate_c,
            score=9.0,
        ),
    ]

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        rrf_k=60,
        top_k=3,
    )

    assert results[0].candidate.chunk_id == "b"

    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1


def test_rrf_uses_rank_not_raw_score() -> None:
    """
    RRF 不直接使用 Dense/BM25 原始分数做加法。
    """

    candidate_a = make_candidate("a")
    candidate_b = make_candidate("b")

    dense_results = [
        DenseSearchResult(
            candidate=candidate_a,
            score=0.51,
        ),
        DenseSearchResult(
            candidate=candidate_b,
            score=0.50,
        ),
    ]

    bm25_results = [
        BM25SearchResult(
            candidate=candidate_a,
            score=1.0,
        ),
        BM25SearchResult(
            candidate=candidate_b,
            score=1000.0,
        ),
    ]

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        rrf_k=60,
        top_k=2,
    )

    # A 两路都是 rank 1，
    # 所以即使 B 的 BM25 原始 score 极大，
    # 也不能因为 raw score 而超过 A。
    assert results[0].candidate.chunk_id == "a"


def test_rrf_rejects_invalid_top_k() -> None:
    """top_k 必须大于 0。"""

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        reciprocal_rank_fusion(
            dense_results=[],
            bm25_results=[],
            top_k=0,
        )


def test_rrf_rejects_negative_rrf_k() -> None:
    """rrf_k 不能为负数。"""

    with pytest.raises(
        ValueError,
        match="rrf_k",
    ):
        reciprocal_rank_fusion(
            dense_results=[],
            bm25_results=[],
            rrf_k=-1,
        )