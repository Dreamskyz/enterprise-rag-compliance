"""测试 Answer / Refusal / Citation Evaluation Metrics。"""

import pytest

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
    aggregate_answer_metrics,
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCategory,
)


def make_result(
    *,
    query_id: str,
    category: RetrievalEvalCategory,
    expected_answerable: bool,
    actual_answerable: bool,
    gold_chunk_ids: tuple[
        str,
        ...,
    ],
    cited_chunk_ids: tuple[
        str,
        ...,
    ],
    citation_gold_chunk_ids: (
        tuple[
            str,
            ...,
        ]
        | None
    ) = None,
    strict_citation_eval: bool = True,
) -> AnswerEvalCaseResult:
    """
    构造测试 Result。

    默认情况下：

        Citation Gold
        =
        Retrieval Gold

    如果测试需要验证二者解耦，
    可以显式传 citation_gold_chunk_ids。
    """

    answer = (
        "测试回答"
        if actual_answerable
        else None
    )

    # ------------------------------------------------------
    # 默认复制 Retrieval Gold。
    #
    # 这只是 Test Helper 的方便写法，
    # 不代表生产系统认为两类 Gold 永远相同。
    # ------------------------------------------------------

    if citation_gold_chunk_ids is None:
        citation_gold_chunk_ids = (
            gold_chunk_ids
        )

    # 不可回答 Case 不进入 strict citation。
    if not expected_answerable:
        strict_citation_eval = False

    return AnswerEvalCaseResult(
        query_id=(
            query_id
        ),
        query="测试问题",
        category=(
            category
        ),
        expected_answerable=(
            expected_answerable
        ),

        # Retrieval Gold。
        gold_chunk_ids=(
            gold_chunk_ids
        ),

        # Citation Gold。
        citation_gold_chunk_ids=(
            citation_gold_chunk_ids
        ),

        strict_citation_eval=(
            strict_citation_eval
        ),

        actual_answerable=(
            actual_answerable
        ),

        answer=answer,

        cited_chunk_ids=(
            cited_chunk_ids
        ),

        gate_reason="passed",

        reason="测试",

        retrieval_count=5,

        top_rerank_score=1.0,

        latency_ms=100.0,
    )


def test_citation_precision() -> None:
    """
    引用 A+B，
    Citation Gold 只有 A。

    Precision = 1 / 2。
    """

    score = (
        citation_precision_for_case(
            cited_chunk_ids=(
                "A",
                "B",
            ),
            gold_chunk_ids=(
                "A",
            ),
        )
    )

    assert score == 0.5


def test_citation_recall() -> None:
    """
    Citation Gold A+B，
    实际只引用 A。

    Recall = 1 / 2。
    """

    score = (
        citation_recall_for_case(
            cited_chunk_ids=(
                "A",
            ),
            gold_chunk_ids=(
                "A",
                "B",
            ),
        )
    )

    assert score == 0.5


def test_aggregate_answer_metrics() -> None:
    """
    构造：

        TP = 1
        FN = 1
        TN = 2
        FP = 1
    """

    results = [
        # --------------------------------------------------
        # TP
        # --------------------------------------------------
        make_result(
            query_id="R001",
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            expected_answerable=True,
            actual_answerable=True,
            gold_chunk_ids=(
                "A",
            ),
            cited_chunk_ids=(
                "A",
            ),
        ),

        # --------------------------------------------------
        # FN
        # --------------------------------------------------
        make_result(
            query_id="R002",
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            expected_answerable=True,
            actual_answerable=False,
            gold_chunk_ids=(
                "B",
            ),
            cited_chunk_ids=(),
        ),

        # --------------------------------------------------
        # TN Hard Negative
        # --------------------------------------------------
        make_result(
            query_id="R003",
            category=(
                RetrievalEvalCategory.HARD_NEGATIVE
            ),
            expected_answerable=False,
            actual_answerable=False,
            gold_chunk_ids=(),
            cited_chunk_ids=(),
        ),

        # --------------------------------------------------
        # TN OOD
        # --------------------------------------------------
        make_result(
            query_id="R004",
            category=(
                RetrievalEvalCategory.OUT_OF_DOMAIN
            ),
            expected_answerable=False,
            actual_answerable=False,
            gold_chunk_ids=(),
            cited_chunk_ids=(),
        ),

        # --------------------------------------------------
        # FP Hard Negative
        # --------------------------------------------------
        make_result(
            query_id="R005",
            category=(
                RetrievalEvalCategory.HARD_NEGATIVE
            ),
            expected_answerable=False,
            actual_answerable=True,
            gold_chunk_ids=(),
            cited_chunk_ids=(
                "X",
            ),
        ),
    ]

    metrics = (
        aggregate_answer_metrics(
            results
        )
    )

    assert metrics.case_count == 5

    assert metrics.true_positive == 1

    assert metrics.false_negative == 1

    assert metrics.true_negative == 2

    assert metrics.false_positive == 1

    assert (
        metrics.overall_decision_accuracy
        == pytest.approx(
            3 / 5
        )
    )

    assert (
        metrics.answerable_accuracy
        == pytest.approx(
            1 / 2
        )
    )

    assert (
        metrics.refusal_accuracy
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics
        .hard_negative_refusal_accuracy
        == pytest.approx(
            1 / 2
        )
    )

    assert (
        metrics
        .out_of_domain_refusal_accuracy
        == 1.0
    )

    # ------------------------------------------------------
    # Citation 只统计：
    #
    # expected answerable
    # +
    # actual answered
    #
    # 因此这里只有 R001。
    # ------------------------------------------------------

    assert (
        metrics.citation_case_count
        == 1
    )

    assert (
        metrics.citation_precision
        == 1.0
    )

    assert (
        metrics.citation_recall
        == 1.0
    )

    assert (
        metrics.citation_hit_rate
        == 1.0
    )

    # R001 strict=true。
    assert (
        metrics.strict_citation_case_count
        == 1
    )

    assert (
        metrics.strict_citation_precision
        == 1.0
    )

    assert (
        metrics.strict_citation_recall
        == 1.0
    )

    assert (
        metrics.strict_citation_hit_rate
        == 1.0
    )


def test_citation_metrics_use_citation_gold() -> None:
    """
    非常关键：

    Citation Metrics 必须使用：

        citation_gold_chunk_ids

    而不是：

        gold_chunk_ids
    """

    result = make_result(
        query_id="R001",
        category=(
            RetrievalEvalCategory.DIRECT
        ),
        expected_answerable=True,
        actual_answerable=True,

        # Retrieval Gold 故意完全不同。
        gold_chunk_ids=(
            "retrieval-only",
        ),

        # Citation Gold。
        citation_gold_chunk_ids=(
            "citation-a",
            "citation-b",
        ),

        # 模型实际引用完整 Citation Gold。
        cited_chunk_ids=(
            "citation-a",
            "citation-b",
        ),

        strict_citation_eval=True,
    )

    metrics = (
        aggregate_answer_metrics(
            [result]
        )
    )

    assert (
        metrics.citation_precision
        == 1.0
    )

    assert (
        metrics.citation_recall
        == 1.0
    )

    assert (
        metrics.citation_hit_rate
        == 1.0
    )


def test_non_strict_case_excluded_from_strict_metrics() -> None:
    """
    non-strict Case：

        仍进入 All-case Citation Metrics

    但：

        不进入 Strict Citation Metrics。
    """

    result = make_result(
        query_id="R010",
        category=(
            RetrievalEvalCategory.SHORT
        ),
        expected_answerable=True,
        actual_answerable=True,
        gold_chunk_ids=(
            "A",
        ),
        citation_gold_chunk_ids=(
            "A",
            "B",
        ),
        cited_chunk_ids=(
            "A",
            "B",
        ),
        strict_citation_eval=False,
    )

    metrics = (
        aggregate_answer_metrics(
            [result]
        )
    )

    assert (
        metrics.citation_case_count
        == 1
    )

    assert (
        metrics.citation_precision
        == 1.0
    )

    assert (
        metrics.strict_citation_case_count
        == 0
    )

    assert (
        metrics.strict_citation_precision
        == 0.0
    )

    assert (
        metrics.strict_citation_recall
        == 0.0
    )

    assert (
        metrics.strict_citation_hit_rate
        == 0.0
    )


def test_strict_and_all_case_metrics_can_differ() -> None:
    """
    验证：

        All-case Citation
        和
        Strict Citation

    可以产生不同结果。
    """

    strict_result = make_result(
        query_id="R001",
        category=(
            RetrievalEvalCategory.DIRECT
        ),
        expected_answerable=True,
        actual_answerable=True,
        gold_chunk_ids=(
            "A",
        ),
        citation_gold_chunk_ids=(
            "A",
        ),
        cited_chunk_ids=(
            "A",
            "EXTRA",
        ),
        strict_citation_eval=True,
    )

    non_strict_result = make_result(
        query_id="R014",
        category=(
            RetrievalEvalCategory.AMBIGUOUS
        ),
        expected_answerable=True,
        actual_answerable=True,
        gold_chunk_ids=(
            "B",
        ),
        citation_gold_chunk_ids=(
            "B",
            "C",
        ),
        cited_chunk_ids=(
            "B",
            "C",
        ),
        strict_citation_eval=False,
    )

    metrics = aggregate_answer_metrics(
        [
            strict_result,
            non_strict_result,
        ]
    )

    # ------------------------------------------------------
    # All-case：
    #
    # R001 precision = 1/2
    # R014 precision = 1
    #
    # Macro = 0.75
    # ------------------------------------------------------

    assert (
        metrics.citation_case_count
        == 2
    )

    assert (
        metrics.citation_precision
        == pytest.approx(
            0.75
        )
    )

    # ------------------------------------------------------
    # Strict：
    #
    # 只有 R001。
    # ------------------------------------------------------

    assert (
        metrics.strict_citation_case_count
        == 1
    )

    assert (
        metrics.strict_citation_precision
        == pytest.approx(
            0.5
        )
    )


def test_empty_results_rejected() -> None:
    """空结果不能汇总。"""

    with pytest.raises(
        ValueError,
        match="results",
    ):
        aggregate_answer_metrics(
            []
        )