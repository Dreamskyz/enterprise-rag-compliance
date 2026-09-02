"""测试 Full-RAG Snapshot Annotation Enrichment。"""

import pytest

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
)
from enterprise_rag.evaluation.answer_snapshot_enrichment import (
    enrich_answer_eval_results,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCategory,
)


def make_old_snapshot_result() -> (
    AnswerEvalCaseResult
):
    """
    模拟旧 Snapshot。

    旧 Snapshot 尚未拥有正式 Citation Gold，
    所以先临时使用 Retrieval Gold。
    """

    return AnswerEvalCaseResult(
        query_id="R001",
        query="测试问题",
        category=(
            RetrievalEvalCategory.DIRECT
        ),
        expected_answerable=True,

        gold_chunk_ids=(
            "retrieval-gold",
        ),

        citation_gold_chunk_ids=(
            "retrieval-gold",
        ),

        strict_citation_eval=False,

        actual_answerable=True,

        answer="真实历史回答",

        cited_chunk_ids=(
            "actual-citation",
        ),

        gate_reason="passed",

        reason="真实历史 Reason",

        retrieval_count=5,

        top_rerank_score=7.0,

        latency_ms=1234.0,
    )


def make_new_dataset_case() -> (
    RetrievalEvalCase
):
    """构造新的 Citation-aware Annotation。"""

    return RetrievalEvalCase(
        query_id="R001",
        query="测试问题",

        gold_chunk_ids=(
            "retrieval-gold",
        ),

        category=(
            RetrievalEvalCategory.DIRECT
        ),

        answerable=True,

        note="",

        citation_gold_chunk_ids=(
            "citation-gold-a",
            "citation-gold-b",
        ),

        strict_citation_eval=True,
    )


def test_enrichment_only_updates_annotations() -> None:
    """
    Enrichment 只能更新人工 Annotation。

    真实模型输出必须保持完全不变。
    """

    original = (
        make_old_snapshot_result()
    )

    case = (
        make_new_dataset_case()
    )

    enriched_results = (
        enrich_answer_eval_results(
            results=[
                original
            ],
            cases=[
                case
            ],
        )
    )

    assert len(
        enriched_results
    ) == 1

    enriched = (
        enriched_results[0]
    )

    # ------------------------------------------------------
    # 人工 Annotation 被更新。
    # ------------------------------------------------------

    assert (
        enriched.citation_gold_chunk_ids
        == (
            "citation-gold-a",
            "citation-gold-b",
        )
    )

    assert (
        enriched.strict_citation_eval
        is True
    )

    # ------------------------------------------------------
    # 真实模型输出必须不变。
    # ------------------------------------------------------

    assert (
        enriched.answer
        == original.answer
    )

    assert (
        enriched.actual_answerable
        == original.actual_answerable
    )

    assert (
        enriched.cited_chunk_ids
        == original.cited_chunk_ids
    )

    assert (
        enriched.reason
        == original.reason
    )

    assert (
        enriched.gate_reason
        == original.gate_reason
    )

    assert (
        enriched.retrieval_count
        == original.retrieval_count
    )

    assert (
        enriched.top_rerank_score
        == original.top_rerank_score
    )

    assert (
        enriched.latency_ms
        == original.latency_ms
    )


def test_enrichment_rejects_unknown_query_id() -> None:
    """
    Snapshot 中出现 Dataset 没有的 Query，
    不能静默忽略。
    """

    result = (
        make_old_snapshot_result()
    )

    with pytest.raises(
        ValueError,
        match="无法匹配",
    ):
        enrich_answer_eval_results(
            results=[
                result
            ],
            cases=[
                RetrievalEvalCase(
                    query_id="R999",
                    query="其它问题",
                    gold_chunk_ids=(
                        "A",
                    ),
                    category=(
                        RetrievalEvalCategory.DIRECT
                    ),
                    answerable=True,
                    note="",
                    citation_gold_chunk_ids=(
                        "A",
                    ),
                    strict_citation_eval=True,
                )
            ],
        )


def test_enrichment_rejects_changed_query() -> None:
    """
    同一个 query_id 的 Query 文本发生变化，
    不允许把旧模型输出强行绑定到新问题。
    """

    result = (
        make_old_snapshot_result()
    )

    changed_case = (
        RetrievalEvalCase(
            query_id="R001",
            query="已经修改过的问题",

            gold_chunk_ids=(
                "retrieval-gold",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="",

            citation_gold_chunk_ids=(
                "citation-gold",
            ),

            strict_citation_eval=True,
        )
    )

    with pytest.raises(
        ValueError,
        match="Query 与 Dataset 不一致",
    ):
        enrich_answer_eval_results(
            results=[
                result
            ],
            cases=[
                changed_case
            ],
        )