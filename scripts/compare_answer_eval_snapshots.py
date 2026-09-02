"""比较 Prompt v1 与 Prompt v2 Full-RAG Evaluation Snapshot。"""

from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.answer_result_store import (
    read_answer_eval_results_jsonl,
)


# ==========================================================
# Baseline：
# Prompt v1 + 最新人工 Citation Annotation。
# ==========================================================

BASELINE_PATH = Path(
    "data/eval/results/"
    "answer_eval_v1_run_001_audited.jsonl"
)


# ==========================================================
# Candidate：
# Prompt v2 Full-RAG Run。
# ==========================================================

CANDIDATE_PATH = Path(
    "data/eval/results/"
    "answer_eval_v1_run_002.jsonl"
)


def format_delta(
    baseline: float,
    candidate: float,
) -> str:
    """
    格式化 Candidate - Baseline。
    """

    delta = (
        candidate
        - baseline
    )

    return (
        f"{delta:+.4f}"
    )


def print_metric_comparison(
    *,
    name: str,
    baseline: float,
    candidate: float,
) -> None:
    """打印单个指标比较。"""

    print(
        f"{name:<38}"
        f"{baseline:>10.4f}"
        f"{candidate:>12.4f}"
        f"{format_delta(baseline, candidate):>12}"
    )


def print_summary_comparison(
    baseline_results,
    candidate_results,
) -> None:
    """
    比较 Aggregate Metrics。
    """

    baseline = (
        aggregate_answer_metrics(
            baseline_results
        )
    )

    candidate = (
        aggregate_answer_metrics(
            candidate_results
        )
    )

    print()
    print("=" * 100)
    print(
        "Prompt v1 vs Prompt v2 "
        "Aggregate Metrics"
    )
    print("=" * 100)

    print(
        f"{'Metric':<38}"
        f"{'Prompt v1':>10}"
        f"{'Prompt v2':>12}"
        f"{'Delta':>12}"
    )

    print(
        "-" * 72
    )

    print_metric_comparison(
        name=(
            "Overall Decision Accuracy"
        ),
        baseline=(
            baseline.overall_decision_accuracy
        ),
        candidate=(
            candidate.overall_decision_accuracy
        ),
    )

    print_metric_comparison(
        name="Answerable Accuracy",
        baseline=(
            baseline.answerable_accuracy
        ),
        candidate=(
            candidate.answerable_accuracy
        ),
    )

    print_metric_comparison(
        name="Refusal Accuracy",
        baseline=(
            baseline.refusal_accuracy
        ),
        candidate=(
            candidate.refusal_accuracy
        ),
    )

    print_metric_comparison(
        name=(
            "Hard Negative Refusal Accuracy"
        ),
        baseline=(
            baseline.hard_negative_refusal_accuracy
        ),
        candidate=(
            candidate.hard_negative_refusal_accuracy
        ),
    )

    print_metric_comparison(
        name="OOD Refusal Accuracy",
        baseline=(
            baseline.out_of_domain_refusal_accuracy
        ),
        candidate=(
            candidate.out_of_domain_refusal_accuracy
        ),
    )

    print_metric_comparison(
        name="Citation Precision",
        baseline=(
            baseline.citation_precision
        ),
        candidate=(
            candidate.citation_precision
        ),
    )

    print_metric_comparison(
        name="Citation Recall",
        baseline=(
            baseline.citation_recall
        ),
        candidate=(
            candidate.citation_recall
        ),
    )

    print_metric_comparison(
        name="Citation Hit Rate",
        baseline=(
            baseline.citation_hit_rate
        ),
        candidate=(
            candidate.citation_hit_rate
        ),
    )

    print_metric_comparison(
        name="Strict Citation Precision",
        baseline=(
            baseline.strict_citation_precision
        ),
        candidate=(
            candidate.strict_citation_precision
        ),
    )

    print_metric_comparison(
        name="Strict Citation Recall",
        baseline=(
            baseline.strict_citation_recall
        ),
        candidate=(
            candidate.strict_citation_recall
        ),
    )

    print_metric_comparison(
        name="Strict Citation Hit Rate",
        baseline=(
            baseline.strict_citation_hit_rate
        ),
        candidate=(
            candidate.strict_citation_hit_rate
        ),
    )

    print()

    print(
        "Prompt v1 TP/TN/FP/FN:",
        (
            baseline.true_positive,
            baseline.true_negative,
            baseline.false_positive,
            baseline.false_negative,
        ),
    )

    print(
        "Prompt v2 TP/TN/FP/FN:",
        (
            candidate.true_positive,
            candidate.true_negative,
            candidate.false_positive,
            candidate.false_negative,
        ),
    )


def print_case_level_changes(
    baseline_results,
    candidate_results,
) -> None:
    """
    输出发生变化的 Query。

    重点观察：

        Answerable Decision
        Citation Set
        Citation Precision / Recall
    """

    baseline_by_id = {
        result.query_id: result
        for result in baseline_results
    }

    candidate_by_id = {
        result.query_id: result
        for result in candidate_results
    }

    print()
    print("=" * 100)
    print(
        "Case-level Changes"
    )
    print("=" * 100)

    changed_count = 0

    for query_id in sorted(
        baseline_by_id
    ):
        baseline = (
            baseline_by_id[
                query_id
            ]
        )

        candidate = (
            candidate_by_id.get(
                query_id
            )
        )

        if candidate is None:
            raise ValueError(
                "Prompt v2 Snapshot "
                "缺少 query_id："
                f"{query_id}"
            )

        decision_changed = (
            baseline.actual_answerable
            != candidate.actual_answerable
        )

        citations_changed = (
            baseline.cited_chunk_ids
            != candidate.cited_chunk_ids
        )

        if (
            not decision_changed
            and not citations_changed
        ):
            continue

        changed_count += 1

        print()

        print(
            query_id,
            "|",
            baseline.query,
        )

        print(
            "Strict:",
            baseline.strict_citation_eval,
        )

        print(
            "Citation Gold:",
            baseline.citation_gold_chunk_ids,
        )

        print(
            "Prompt v1 Answerable:",
            baseline.actual_answerable,
        )

        print(
            "Prompt v2 Answerable:",
            candidate.actual_answerable,
        )

        print(
            "Prompt v1 Citations:",
            baseline.cited_chunk_ids,
        )

        print(
            "Prompt v2 Citations:",
            candidate.cited_chunk_ids,
        )

        # --------------------------------------------------
        # Citation Metrics 只对两边都回答的情况计算。
        # --------------------------------------------------

        if (
            baseline.actual_answerable
            and candidate.actual_answerable
            and baseline.expected_answerable
        ):
            baseline_precision = (
                citation_precision_for_case(
                    cited_chunk_ids=(
                        baseline.cited_chunk_ids
                    ),
                    gold_chunk_ids=(
                        baseline.citation_gold_chunk_ids
                    ),
                )
            )

            baseline_recall = (
                citation_recall_for_case(
                    cited_chunk_ids=(
                        baseline.cited_chunk_ids
                    ),
                    gold_chunk_ids=(
                        baseline.citation_gold_chunk_ids
                    ),
                )
            )

            candidate_precision = (
                citation_precision_for_case(
                    cited_chunk_ids=(
                        candidate.cited_chunk_ids
                    ),
                    gold_chunk_ids=(
                        candidate.citation_gold_chunk_ids
                    ),
                )
            )

            candidate_recall = (
                citation_recall_for_case(
                    cited_chunk_ids=(
                        candidate.cited_chunk_ids
                    ),
                    gold_chunk_ids=(
                        candidate.citation_gold_chunk_ids
                    ),
                )
            )

            print(
                "Citation Precision:",
                f"{baseline_precision:.4f}",
                "→",
                f"{candidate_precision:.4f}",
            )

            print(
                "Citation Recall:",
                f"{baseline_recall:.4f}",
                "→",
                f"{candidate_recall:.4f}",
            )

        print(
            "Prompt v1 Answer:",
            baseline.answer,
        )

        print(
            "Prompt v2 Answer:",
            candidate.answer,
        )

    if changed_count == 0:
        print(
            "No case-level changes."
        )


def print_prompt_v2_regressions(
    baseline_results,
    candidate_results,
) -> None:
    """
    专门寻找 Prompt v2 Regression。

    当前定义：

    1. Prompt v1 Decision 正确，
       Prompt v2 Decision 错误；

    2. Strict Citation Case 中，
       Prompt v2 Precision 或 Recall
       比 Prompt v1 更低。
    """

    baseline_by_id = {
        result.query_id: result
        for result in baseline_results
    }

    candidate_by_id = {
        result.query_id: result
        for result in candidate_results
    }

    print()
    print("=" * 100)
    print(
        "Prompt v2 Regression Check"
    )
    print("=" * 100)

    regression_count = 0

    for query_id, baseline in (
        baseline_by_id.items()
    ):
        candidate = (
            candidate_by_id.get(
                query_id
            )
        )

        if candidate is None:
            raise ValueError(
                "Prompt v2 Snapshot "
                "缺少 query_id："
                f"{query_id}"
            )

        baseline_decision_correct = (
            baseline.actual_answerable
            == baseline.expected_answerable
        )

        candidate_decision_correct = (
            candidate.actual_answerable
            == candidate.expected_answerable
        )

        if (
            baseline_decision_correct
            and not candidate_decision_correct
        ):
            regression_count += 1

            print()

            print(
                query_id,
                "| Decision Regression"
            )

            print(
                "Query:",
                baseline.query,
            )

            print(
                "Expected:",
                baseline.expected_answerable,
            )

            print(
                "Prompt v1:",
                baseline.actual_answerable,
            )

            print(
                "Prompt v2:",
                candidate.actual_answerable,
            )

            continue

        if not baseline.strict_citation_eval:
            continue

        if not baseline.expected_answerable:
            continue

        if not (
            baseline.actual_answerable
            and candidate.actual_answerable
        ):
            continue

        baseline_precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    baseline.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    baseline.citation_gold_chunk_ids
                ),
            )
        )

        candidate_precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    candidate.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    candidate.citation_gold_chunk_ids
                ),
            )
        )

        baseline_recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    baseline.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    baseline.citation_gold_chunk_ids
                ),
            )
        )

        candidate_recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    candidate.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    candidate.citation_gold_chunk_ids
                ),
            )
        )

        if (
            candidate_precision
            < baseline_precision
            or candidate_recall
            < baseline_recall
        ):
            regression_count += 1

            print()

            print(
                query_id,
                "| Citation Regression"
            )

            print(
                "Query:",
                baseline.query,
            )

            print(
                "Precision:",
                f"{baseline_precision:.4f}",
                "→",
                f"{candidate_precision:.4f}",
            )

            print(
                "Recall:",
                f"{baseline_recall:.4f}",
                "→",
                f"{candidate_recall:.4f}",
            )

            print(
                "Prompt v1:",
                baseline.cited_chunk_ids,
            )

            print(
                "Prompt v2:",
                candidate.cited_chunk_ids,
            )

    if regression_count == 0:
        print(
            "✅ No Prompt v2 regressions detected."
        )


def main() -> None:
    """离线比较 Prompt v1 / Prompt v2 Snapshot。"""

    print("=" * 100)
    print(
        "Prompt v1 vs Prompt v2 "
        "Full-RAG Evaluation Comparison"
    )
    print("=" * 100)

    baseline_results = (
        read_answer_eval_results_jsonl(
            BASELINE_PATH
        )
    )

    candidate_results = (
        read_answer_eval_results_jsonl(
            CANDIDATE_PATH
        )
    )

    if (
        len(baseline_results)
        != len(candidate_results)
    ):
        raise ValueError(
            "两个 Snapshot Case 数量不一致"
        )

    print(
        "Prompt v1:",
        BASELINE_PATH,
    )

    print(
        "Prompt v2:",
        CANDIDATE_PATH,
    )

    print(
        "Cases:",
        len(baseline_results),
    )

    print_summary_comparison(
        baseline_results,
        candidate_results,
    )

    print_case_level_changes(
        baseline_results,
        candidate_results,
    )

    print_prompt_v2_regressions(
        baseline_results,
        candidate_results,
    )

    print()
    print("=" * 100)

    print(
        "✅ Snapshot comparison completed."
    )

    print(
        "✅ No GPU / Qdrant / LLM "
        "inference was required."
    )


if __name__ == "__main__":
    main()