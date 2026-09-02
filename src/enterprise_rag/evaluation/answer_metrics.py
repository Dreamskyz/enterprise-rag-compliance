"""Answer / Refusal Evaluation Metrics。"""

from collections.abc import Sequence
from dataclasses import dataclass

from enterprise_rag.evaluation.models import (
    RetrievalEvalCategory,
)


@dataclass(frozen=True)
class AnswerEvalCaseResult:
    """
    单条 Full-RAG Evaluation 原始结果。
    """

    query_id: str

    query: str

    category: RetrievalEvalCategory

    expected_answerable: bool

    # Retrieval Gold。
    gold_chunk_ids: tuple[
        str,
        ...,
    ]

    # Citation Gold。
    citation_gold_chunk_ids: tuple[
        str,
        ...,
    ]

    # 是否进入 Strict Citation Metrics。
    strict_citation_eval: bool

    actual_answerable: bool

    answer: str | None

    cited_chunk_ids: tuple[
        str,
        ...,
    ]

    gate_reason: str

    reason: str

    retrieval_count: int

    top_rerank_score: float | None

    latency_ms: float


@dataclass(frozen=True)
class AnswerAggregateMetrics:
    """
    Dataset 级 Answer / Citation 指标。
    """

    case_count: int

    answerable_count: int

    unanswerable_count: int

    true_positive: int

    true_negative: int

    false_positive: int

    false_negative: int

    overall_decision_accuracy: float

    answerable_accuracy: float

    refusal_accuracy: float

    hard_negative_refusal_accuracy: float

    out_of_domain_refusal_accuracy: float

    # ======================================================
    # All-case Citation Metrics。
    #
    # 对所有：
    #
    # expected_answerable=true
    # actual_answerable=true
    #
    # 的 Case 计算。
    # ======================================================

    citation_case_count: int

    citation_precision: float

    citation_recall: float

    citation_hit_rate: float

    # ======================================================
    # Strict Citation Metrics。
    #
    # 在上面基础上进一步要求：
    #
    # strict_citation_eval=true
    # ======================================================

    strict_citation_case_count: int

    strict_citation_precision: float

    strict_citation_recall: float

    strict_citation_hit_rate: float


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """安全除法。"""

    if denominator == 0:
        return 0.0

    return numerator / denominator


def citation_precision_for_case(
    *,
    cited_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> float:
    """
    Citation Precision。

        正确引用
        /
        实际引用
    """

    if not cited_chunk_ids:
        return 0.0

    cited_set = set(
        cited_chunk_ids
    )

    gold_set = set(
        gold_chunk_ids
    )

    return (
        len(
            cited_set.intersection(
                gold_set
            )
        )
        / len(cited_set)
    )


def citation_recall_for_case(
    *,
    cited_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> float:
    """
    Citation Recall。

        被引用的 Citation Gold
        /
        Citation Gold 总数
    """

    if not gold_chunk_ids:
        raise ValueError(
            "gold_chunk_ids 不能为空"
        )

    cited_set = set(
        cited_chunk_ids
    )

    gold_set = set(
        gold_chunk_ids
    )

    return (
        len(
            cited_set.intersection(
                gold_set
            )
        )
        / len(gold_set)
    )


def citation_hit_for_case(
    *,
    cited_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> bool:
    """至少命中一个 Citation Gold。"""

    return bool(
        set(
            cited_chunk_ids
        ).intersection(
            gold_chunk_ids
        )
    )


def _aggregate_citation_subset(
    results: Sequence[
        AnswerEvalCaseResult
    ],
) -> tuple[
    int,
    float,
    float,
    float,
]:
    """
    汇总一个 Citation Evaluation 子集。

    返回：

        case_count
        macro precision
        macro recall
        hit rate
    """

    if not results:
        return (
            0,
            0.0,
            0.0,
            0.0,
        )

    precisions: list[
        float
    ] = []

    recalls: list[
        float
    ] = []

    hit_count = 0

    for result in results:
        precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result.citation_gold_chunk_ids
                ),
            )
        )

        recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result.citation_gold_chunk_ids
                ),
            )
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        if citation_hit_for_case(
            cited_chunk_ids=(
                result.cited_chunk_ids
            ),
            gold_chunk_ids=(
                result.citation_gold_chunk_ids
            ),
        ):
            hit_count += 1

    case_count = len(
        results
    )

    return (
        case_count,
        sum(precisions)
        / case_count,
        sum(recalls)
        / case_count,
        hit_count
        / case_count,
    )


def aggregate_answer_metrics(
    results: Sequence[
        AnswerEvalCaseResult
    ],
) -> AnswerAggregateMetrics:
    """汇总 Full-RAG Evaluation。"""

    if not results:
        raise ValueError(
            "results 不能为空"
        )

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    answerable_results: list[
        AnswerEvalCaseResult
    ] = []

    unanswerable_results: list[
        AnswerEvalCaseResult
    ] = []

    hard_negative_results: list[
        AnswerEvalCaseResult
    ] = []

    out_of_domain_results: list[
        AnswerEvalCaseResult
    ] = []

    for result in results:
        if result.expected_answerable:
            answerable_results.append(
                result
            )

            if result.actual_answerable:
                true_positive += 1
            else:
                false_negative += 1

        else:
            unanswerable_results.append(
                result
            )

            if result.actual_answerable:
                false_positive += 1
            else:
                true_negative += 1

            if (
                result.category
                == RetrievalEvalCategory.HARD_NEGATIVE
            ):
                hard_negative_results.append(
                    result
                )

            if (
                result.category
                == RetrievalEvalCategory.OUT_OF_DOMAIN
            ):
                out_of_domain_results.append(
                    result
                )

    case_count = len(
        results
    )

    answerable_count = len(
        answerable_results
    )

    unanswerable_count = len(
        unanswerable_results
    )

    overall_decision_accuracy = (
        safe_ratio(
            true_positive
            + true_negative,
            case_count,
        )
    )

    answerable_accuracy = (
        safe_ratio(
            true_positive,
            answerable_count,
        )
    )

    refusal_accuracy = (
        safe_ratio(
            true_negative,
            unanswerable_count,
        )
    )

    hard_negative_correct = sum(
        1
        for result
        in hard_negative_results
        if not result.actual_answerable
    )

    hard_negative_refusal_accuracy = (
        safe_ratio(
            hard_negative_correct,
            len(
                hard_negative_results
            ),
        )
    )

    ood_correct = sum(
        1
        for result
        in out_of_domain_results
        if not result.actual_answerable
    )

    out_of_domain_refusal_accuracy = (
        safe_ratio(
            ood_correct,
            len(
                out_of_domain_results
            ),
        )
    )

    # ======================================================
    # All-case Citation Subset。
    # ======================================================

    answered_answerable_results = [
        result
        for result in answerable_results
        if result.actual_answerable
    ]

    (
        citation_case_count,
        citation_precision,
        citation_recall,
        citation_hit_rate,
    ) = _aggregate_citation_subset(
        answered_answerable_results
    )

    # ======================================================
    # Strict Citation Subset。
    # ======================================================

    strict_results = [
        result
        for result
        in answered_answerable_results
        if result.strict_citation_eval
    ]

    (
        strict_citation_case_count,
        strict_citation_precision,
        strict_citation_recall,
        strict_citation_hit_rate,
    ) = _aggregate_citation_subset(
        strict_results
    )

    return AnswerAggregateMetrics(
        case_count=case_count,
        answerable_count=(
            answerable_count
        ),
        unanswerable_count=(
            unanswerable_count
        ),
        true_positive=(
            true_positive
        ),
        true_negative=(
            true_negative
        ),
        false_positive=(
            false_positive
        ),
        false_negative=(
            false_negative
        ),
        overall_decision_accuracy=(
            overall_decision_accuracy
        ),
        answerable_accuracy=(
            answerable_accuracy
        ),
        refusal_accuracy=(
            refusal_accuracy
        ),
        hard_negative_refusal_accuracy=(
            hard_negative_refusal_accuracy
        ),
        out_of_domain_refusal_accuracy=(
            out_of_domain_refusal_accuracy
        ),
        citation_case_count=(
            citation_case_count
        ),
        citation_precision=(
            citation_precision
        ),
        citation_recall=(
            citation_recall
        ),
        citation_hit_rate=(
            citation_hit_rate
        ),
        strict_citation_case_count=(
            strict_citation_case_count
        ),
        strict_citation_precision=(
            strict_citation_precision
        ),
        strict_citation_recall=(
            strict_citation_recall
        ),
        strict_citation_hit_rate=(
            strict_citation_hit_rate
        ),
    )