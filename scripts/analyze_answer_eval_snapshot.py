"""离线分析 Citation-aware Full-RAG Evaluation Snapshot。"""

from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.answer_result_store import (
    read_answer_eval_results_jsonl,
)


INPUT_PATH = Path(
    "data/eval/results/"
    "answer_eval_v1_run_001_audited.jsonl"
)


def print_summary(
    results,
) -> None:
    """
    打印 Decision + Citation Metrics。
    """

    metrics = (
        aggregate_answer_metrics(
            results
        )
    )

    print()
    print("=" * 100)
    print(
        "Offline Answer Evaluation Summary"
    )
    print("=" * 100)

    print(
        "Cases:",
        metrics.case_count,
    )

    print(
        "Answerable:",
        metrics.answerable_count,
    )

    print(
        "Unanswerable:",
        metrics.unanswerable_count,
    )

    print()

    print(
        "TP:",
        metrics.true_positive,
    )

    print(
        "TN:",
        metrics.true_negative,
    )

    print(
        "FP:",
        metrics.false_positive,
    )

    print(
        "FN:",
        metrics.false_negative,
    )

    print()

    print(
        "Overall Decision Accuracy:",
        f"{metrics.overall_decision_accuracy:.4f}",
    )

    print(
        "Answerable Accuracy:",
        f"{metrics.answerable_accuracy:.4f}",
    )

    print(
        "Refusal Accuracy:",
        f"{metrics.refusal_accuracy:.4f}",
    )

    print(
        "Hard Negative Refusal Accuracy:",
        (
            f"{metrics.hard_negative_refusal_accuracy:.4f}"
        ),
    )

    print(
        "OOD Refusal Accuracy:",
        (
            f"{metrics.out_of_domain_refusal_accuracy:.4f}"
        ),
    )

    # ======================================================
    # All-case Citation Metrics。
    # ======================================================

    print()
    print("-" * 100)
    print(
        "All-case Citation Metrics"
    )
    print("-" * 100)

    print(
        "Citation Cases:",
        metrics.citation_case_count,
    )

    print(
        "Citation Precision:",
        f"{metrics.citation_precision:.4f}",
    )

    print(
        "Citation Recall:",
        f"{metrics.citation_recall:.4f}",
    )

    print(
        "Citation Hit Rate:",
        f"{metrics.citation_hit_rate:.4f}",
    )

    # ======================================================
    # Strict Citation Metrics。
    # ======================================================

    print()
    print("-" * 100)
    print(
        "Strict Citation Metrics"
    )
    print("-" * 100)

    print(
        "Strict Citation Cases:",
        metrics.strict_citation_case_count,
    )

    print(
        "Strict Citation Precision:",
        f"{metrics.strict_citation_precision:.4f}",
    )

    print(
        "Strict Citation Recall:",
        f"{metrics.strict_citation_recall:.4f}",
    )

    print(
        "Strict Citation Hit Rate:",
        f"{metrics.strict_citation_hit_rate:.4f}",
    )


def print_all_citation_mismatches(
    results,
) -> None:
    """
    打印所有 Citation Mismatch。

    使用：

        citation_gold_chunk_ids

    而不是 Retrieval Gold。
    """

    print()
    print("=" * 100)
    print(
        "All Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in results:
        if not result.expected_answerable:
            continue

        if not result.actual_answerable:
            continue

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

        if (
            precision == 1.0
            and recall == 1.0
        ):
            continue

        mismatch_count += 1

        citation_gold_set = set(
            result.citation_gold_chunk_ids
        )

        cited_set = set(
            result.cited_chunk_ids
        )

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Strict:",
            result.strict_citation_eval,
        )

        print(
            "Citation Gold:",
            result.citation_gold_chunk_ids,
        )

        print(
            "Actual Citations:",
            result.cited_chunk_ids,
        )

        print(
            "Extra:",
            tuple(
                sorted(
                    cited_set
                    - citation_gold_set
                )
            ),
        )

        print(
            "Missing:",
            tuple(
                sorted(
                    citation_gold_set
                    - cited_set
                )
            ),
        )

        print(
            "Precision:",
            f"{precision:.4f}",
        )

        print(
            "Recall:",
            f"{recall:.4f}",
        )

    if mismatch_count == 0:
        print(
            "No citation mismatches."
        )


def print_strict_citation_mismatches(
    results,
) -> None:
    """
    只打印：

        strict_citation_eval=true

    的 Citation Mismatch。

    这一组最值得用于：

        Prompt Optimization
        Citation Selection Failure Analysis
    """

    print()
    print("=" * 100)
    print(
        "Strict Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in results:
        if not result.expected_answerable:
            continue

        if not result.actual_answerable:
            continue

        if not result.strict_citation_eval:
            continue

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

        if (
            precision == 1.0
            and recall == 1.0
        ):
            continue

        mismatch_count += 1

        citation_gold_set = set(
            result.citation_gold_chunk_ids
        )

        cited_set = set(
            result.cited_chunk_ids
        )

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Citation Gold:",
            result.citation_gold_chunk_ids,
        )

        print(
            "Actual Citations:",
            result.cited_chunk_ids,
        )

        print(
            "Extra:",
            tuple(
                sorted(
                    cited_set
                    - citation_gold_set
                )
            ),
        )

        print(
            "Missing:",
            tuple(
                sorted(
                    citation_gold_set
                    - cited_set
                )
            ),
        )

        print(
            "Precision:",
            f"{precision:.4f}",
        )

        print(
            "Recall:",
            f"{recall:.4f}",
        )

        print(
            "Reason:",
            result.reason,
        )

    if mismatch_count == 0:
        print(
            "No strict citation mismatches."
        )


def main() -> None:
    """
    离线分析已经 Audit 后的 Snapshot。

    不调用：

        GPU
        Qdrant
        SiliconFlow
        LLM
    """

    print("=" * 100)
    print(
        "Offline Citation-aware "
        "Answer Evaluation Analysis"
    )
    print("=" * 100)

    results = (
        read_answer_eval_results_jsonl(
            INPUT_PATH
        )
    )

    print(
        "Snapshot:",
        INPUT_PATH,
    )

    print(
        "Cases:",
        len(results),
    )

    print_summary(
        results
    )

    print_all_citation_mismatches(
        results
    )

    print_strict_citation_mismatches(
        results
    )

    print()
    print("=" * 100)

    print(
        "✅ Offline analysis completed."
    )

    print(
        "✅ No GPU / Qdrant / LLM "
        "inference was required."
    )


if __name__ == "__main__":
    main()