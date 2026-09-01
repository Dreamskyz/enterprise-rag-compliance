"""检查 Evidence Gate 的基础控制逻辑。"""

from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
    RetrievalCandidate,
)


def make_result(
    chunk_id: str,
    score: float,
) -> RerankedSearchResult:
    """构造测试用 Reranked Result。"""

    candidate = RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="gate_test_doc",
        title="Evidence Gate 测试文档",
        document_type="regulation",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="测试章节",
        article_number="第一条",
        content="Evidence Gate 测试正文。",
        retrieval_text=(
            "Evidence Gate 测试文档\n"
            "Evidence Gate 测试正文。"
        ),
        source_url="https://example.com",
        access_level="public",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
    )

    return RerankedSearchResult(
        candidate=candidate,
        rerank_score=score,
        original_rank=1,
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
    )


def print_decision(
    name: str,
    decision,
) -> None:
    """打印 Gate 判定。"""

    print()
    print("=" * 80)

    print(
        "Case:",
        name,
    )

    print(
        "Passed:",
        decision.passed,
    )

    print(
        "Reason:",
        decision.reason.value,
    )

    print(
        "Top Score:",
        decision.top_score,
    )

    print(
        "Threshold:",
        decision.threshold,
    )


def main() -> None:
    """
    使用人为分数检查 Gate 逻辑。

    注意：

    threshold=1.0
    这里只是测试数据。

    它不是项目正式 Evidence Threshold。
    """

    gate = EvidenceGate(
        min_top_score=1.0
    )

    no_results = gate.evaluate(
        []
    )

    weak_evidence = gate.evaluate(
        [
            make_result(
                chunk_id="weak",
                score=0.2,
            )
        ]
    )

    strong_evidence = gate.evaluate(
        [
            make_result(
                chunk_id="strong",
                score=4.0,
            )
        ]
    )

    print_decision(
        "No Results",
        no_results,
    )

    print_decision(
        "Weak Evidence",
        weak_evidence,
    )

    print_decision(
        "Strong Evidence",
        strong_evidence,
    )

    assert (
        no_results.passed
        is False
    )

    assert (
        weak_evidence.passed
        is False
    )

    assert (
        strong_evidence.passed
        is True
    )

    print()
    print("=" * 80)

    print(
        "✅ Evidence Gate "
        "基础控制逻辑验证通过"
    )

    print()

    print(
        "⚠ 当前 threshold=1.0 "
        "仅用于逻辑测试，"
        "不是正式项目阈值。"
    )


if __name__ == "__main__":
    main()