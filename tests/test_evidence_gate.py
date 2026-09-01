"""测试 Evidence Gate 判定逻辑。"""

import math

import pytest

from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.evidence.models import (
    EvidenceDecisionReason,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
    RetrievalCandidate,
)


def make_candidate(
    chunk_id: str,
) -> RetrievalCandidate:
    """构造测试 Candidate。"""

    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="test_doc",
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
        content_hash="test-hash",
    )


def make_result(
    score: float,
) -> RerankedSearchResult:
    """构造指定 Rerank Score 的测试结果。"""

    return RerankedSearchResult(
        candidate=make_candidate(
            chunk_id=f"chunk-{score}"
        ),
        rerank_score=score,
        original_rank=1,
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
    )


def test_gate_rejects_no_results() -> None:
    """
    没有 Retrieval Result 时，
    必须直接拒绝进入 Generation。
    """

    gate = EvidenceGate(
        min_top_score=1.0
    )

    decision = gate.evaluate(
        []
    )

    assert decision.passed is False

    assert (
        decision.reason
        == EvidenceDecisionReason.NO_RESULTS
    )

    assert decision.top_score is None

    assert decision.threshold == 1.0


def test_gate_passes_score_above_threshold() -> None:
    """
    Top1 Score 高于阈值时，
    Gate 应通过。
    """

    gate = EvidenceGate(
        min_top_score=1.0
    )

    results = [
        make_result(3.0),
        make_result(0.5),
    ]

    decision = gate.evaluate(
        results
    )

    assert decision.passed is True

    assert (
        decision.reason
        == EvidenceDecisionReason.PASSED
    )

    assert decision.top_score == 3.0


def test_gate_passes_score_equal_to_threshold() -> None:
    """
    当前规则采用：

        score >= threshold

    所以恰好等于阈值时通过。
    """

    gate = EvidenceGate(
        min_top_score=1.0
    )

    decision = gate.evaluate(
        [
            make_result(1.0),
        ]
    )

    assert decision.passed is True


def test_gate_rejects_score_below_threshold() -> None:
    """
    Top1 Score 低于阈值时，
    Gate 应拒绝。
    """

    gate = EvidenceGate(
        min_top_score=1.0
    )

    decision = gate.evaluate(
        [
            make_result(0.9),
        ]
    )

    assert decision.passed is False

    assert (
        decision.reason
        == EvidenceDecisionReason
        .BELOW_THRESHOLD
    )

    assert decision.top_score == 0.9


def test_gate_supports_negative_threshold() -> None:
    """
    Reranker raw score 允许出现负数。

    因此 threshold 本身也不应该
    被代码强制要求 >= 0。

    最终阈值由真实评测数据决定。
    """

    gate = EvidenceGate(
        min_top_score=-2.0
    )

    decision = gate.evaluate(
        [
            make_result(-1.5),
        ]
    )

    assert decision.passed is True


def test_gate_rejects_non_finite_threshold() -> None:
    """阈值不能是 NaN / Infinity。"""

    with pytest.raises(
        ValueError,
        match="有限数值",
    ):
        EvidenceGate(
            min_top_score=math.nan
        )

    with pytest.raises(
        ValueError,
        match="有限数值",
    ):
        EvidenceGate(
            min_top_score=math.inf
        )


def test_gate_rejects_non_finite_top_score() -> None:
    """Rerank Score 出现 NaN 时应显式失败。"""

    gate = EvidenceGate(
        min_top_score=1.0
    )

    with pytest.raises(
        ValueError,
        match="rerank_score",
    ):
        gate.evaluate(
            [
                make_result(
                    math.nan
                ),
            ]
        )