"""运行 Prompt v2 完整 RAG Answer / Refusal Evaluation。"""

from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.answer_result_store import (
    write_answer_eval_results_jsonl,
)
from enterprise_rag.evaluation.answer_runner import (
    run_answer_evaluation,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.runtime.builder import (
    build_rag_runtime,
)


# ==========================================================
# Evaluation Input。
# ==========================================================

EVAL_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# Prompt v2 Full-RAG Snapshot。
#
# run_001：
#     Prompt v1 历史基线。
#
# run_002：
#     Minimal Sufficient Evidence +
#     Scope Matching Prompt。
#
# 不允许覆盖 run_001。
# ==========================================================

OUTPUT_PATH = Path(
    "data/eval/results/"
    "answer_eval_v1_run_002.jsonl"
)


def print_decision_failures(
    case_results,
) -> None:
    """
    打印 Answer / Refusal Decision Failure。

    也就是：

        Gold Answerable
        !=
        System Answerable
    """

    failures = [
        result
        for result in case_results
        if (
            result.expected_answerable
            != result.actual_answerable
        )
    ]

    print()
    print("=" * 100)
    print(
        "Answer / Refusal Failure Cases"
    )
    print("=" * 100)

    if not failures:
        print(
            "No decision failures."
        )
        return

    for result in failures:
        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Category:",
            result.category.value,
        )

        print(
            "Expected answerable:",
            result.expected_answerable,
        )

        print(
            "Actual answerable:",
            result.actual_answerable,
        )

        print(
            "Gate reason:",
            result.gate_reason,
        )

        print(
            "Retrieval Gold:",
            result.gold_chunk_ids,
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
            "Reason:",
            result.reason,
        )


def print_all_citation_mismatches(
    case_results,
) -> None:
    """
    打印所有 Citation Mismatch。

    注意：

    这里必须使用：

        citation_gold_chunk_ids

    而不能重新使用：

        gold_chunk_ids

    因为 Retrieval Gold 和 Citation Gold
    已经正式解耦。
    """

    print()
    print("=" * 100)
    print(
        "All Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in case_results:
        # Citation Metrics 只分析：
        #
        # Gold 可回答
        # 且
        # 系统真正回答
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

        print(
            "Reason:",
            result.reason,
        )

    if mismatch_count == 0:
        print(
            "No citation mismatches."
        )


def print_strict_citation_mismatches(
    case_results,
) -> None:
    """
    只打印 Strict Citation Failure。

    这一组才是最值得继续进行
    Prompt Failure Analysis 的 Case。
    """

    print()
    print("=" * 100)
    print(
        "Strict Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in case_results:
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
            "Answer:",
            result.answer,
        )

        print(
            "Reason:",
            result.reason,
        )

    if mismatch_count == 0:
        print(
            "No strict citation mismatches."
        )


def print_evidence_id_leaks(
    case_results,
) -> None:
    """
    检查最终 answer 正文是否泄露内部 Evidence ID。

    Prompt v2 已明确规定：

        E1 / E2 / E3

    只能出现在 citations 数组中，
    不应出现在给用户看的 answer 正文。

    这里先做一个简单 deterministic 检查。
    """

    print()
    print("=" * 100)
    print(
        "Answer Evidence-ID Leak Check"
    )
    print("=" * 100)

    leak_count = 0

    # 当前 Query 最多只有少量 Evidence。
    # 检查 E1 ~ E20 已足够覆盖当前系统。
    evidence_tokens = tuple(
        f"E{index}"
        for index in range(
            1,
            21,
        )
    )

    for result in case_results:
        if not result.answer:
            continue

        leaked_ids = tuple(
            evidence_id
            for evidence_id
            in evidence_tokens
            if evidence_id
            in result.answer
        )

        if not leaked_ids:
            continue

        leak_count += 1

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Leaked Evidence IDs:",
            leaked_ids,
        )

        print(
            "Answer:",
            result.answer,
        )

    if leak_count == 0:
        print(
            "No Evidence-ID leaks."
        )


def print_summary(
    result,
) -> None:
    """
    打印 Prompt v2 Full-RAG Evaluation Summary。
    """

    metrics = result.metrics

    print()
    print("=" * 100)
    print(
        "Prompt v2 Full-RAG Evaluation Summary"
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
    # All-case Citation。
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
    # Strict Citation。
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
        (
            f"{metrics.strict_citation_precision:.4f}"
        ),
    )

    print(
        "Strict Citation Recall:",
        (
            f"{metrics.strict_citation_recall:.4f}"
        ),
    )

    print(
        "Strict Citation Hit Rate:",
        (
            f"{metrics.strict_citation_hit_rate:.4f}"
        ),
    )

    print()

    print(
        "Total Eval Time:",
        (
            f"{result.total_latency_ms / 1000.0:.2f}s"
        ),
    )

    print(
        "Mean End-to-End Time:",
        (
            f"{result.mean_latency_ms:.2f}"
            "ms/query"
        ),
    )


def main() -> None:
    """
    运行 Prompt v2 完整 Full-RAG Evaluation。

    本次实验会真实调用：

        BGE-M3
        Qdrant
        BM25
        RRF
        bge-reranker-v2-m3
        Evidence Gate
        SiliconFlow LLM

    生成的 Raw Result 保存为：

        answer_eval_v1_run_002.jsonl
    """

    print("=" * 100)
    print(
        "Prompt v2 Full RAG Evaluation"
    )
    print("=" * 100)

    cases = (
        read_retrieval_eval_jsonl(
            EVAL_PATH
        )
    )

    print(
        "Eval cases:",
        len(cases),
    )

    print(
        "Snapshot output:",
        OUTPUT_PATH,
    )

    print()

    print(
        "Initializing RAG runtime..."
    )

    runtime = build_rag_runtime(
        chunks_path=(
            CHUNKS_PATH
        )
    )

    print(
        "RAG runtime initialized."
    )

    print()
    print(
        "Running Prompt v2 "
        "Full-RAG evaluation..."
    )

    result = run_answer_evaluation(
        cases=cases,
        query_service=(
            runtime.query_service
        ),
    )

    # ======================================================
    # 昂贵推理结束后第一时间保存。
    # ======================================================

    write_answer_eval_results_jsonl(
        results=(
            result.case_results
        ),
        output_path=(
            OUTPUT_PATH
        ),
    )

    print()
    print(
        "Prompt v2 snapshot saved:"
    )

    print(
        OUTPUT_PATH
    )

    # ======================================================
    # Report。
    # ======================================================

    print_summary(
        result
    )

    print_decision_failures(
        result.case_results
    )

    print_all_citation_mismatches(
        result.case_results
    )

    print_strict_citation_mismatches(
        result.case_results
    )

    print_evidence_id_leaks(
        result.case_results
    )

    print()
    print("=" * 100)

    print(
        "⚠ 当前为 20 条 Seed Dataset "
        "上的 Prompt v2 Full-RAG Evaluation。"
    )

    print(
        "⚠ run_002 不覆盖 run_001，"
        "用于 Prompt v1 / v2 离线对比。"
    )

    print(
        "⚠ End-to-End Time 包含 Retrieval、"
        "Rerank、LLM 网络及生成，"
        "不作为正式 API 性能 Benchmark。"
    )


if __name__ == "__main__":
    main()