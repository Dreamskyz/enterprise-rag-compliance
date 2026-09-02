"""运行 Citation Prompt Targeted Regression。"""

from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.runtime.builder import (
    build_rag_runtime,
)


EVAL_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# Targeted Regression Cases。
#
# R001：
#     当前真实 over-citation failure。
#
# R006 / R008：
#     合理多 Citation Control Cases。
#
# 如果 Prompt 优化导致它们被错误压成单 Citation，
# 说明优化发生 Regression。
# ==========================================================

TARGET_QUERY_IDS = (
    "R001",
    "R006",
    "R008",
)


def main() -> None:
    """
    只运行少量 Citation Regression Case。

    注意：

    本脚本会真实调用：

        Retrieval
        Reranker
        SiliconFlow LLM

    但只跑 3 条 Query，
    不需要重新执行完整 20 条 Full-RAG Eval。
    """

    print("=" * 100)
    print(
        "Targeted Citation Regression"
    )
    print("=" * 100)

    cases = (
        read_retrieval_eval_jsonl(
            EVAL_PATH
        )
    )

    case_by_id = {
        case.query_id: case
        for case in cases
    }

    selected_cases = []

    for query_id in TARGET_QUERY_IDS:
        case = case_by_id.get(
            query_id
        )

        if case is None:
            raise ValueError(
                "Dataset 中找不到 "
                f"Target Case：{query_id}"
            )

        selected_cases.append(
            case
        )

    print(
        "Target cases:",
        ", ".join(
            TARGET_QUERY_IDS
        ),
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

    all_passed = True

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):
        print()
        print("=" * 100)

        print(
            f"[{index}/{len(selected_cases)}] "
            f"{case.query_id}"
        )

        print(
            "Query:",
            case.query,
        )

        print(
            "Citation Gold:",
            case.citation_gold_chunk_ids,
        )

        result = (
            runtime.query_service.ask(
                query=case.query,
            )
        )

        cited_chunk_ids = tuple(
            citation.chunk_id
            for citation
            in result.citations
        )

        print(
            "Answerable:",
            result.answerable,
        )

        print(
            "Answer:",
            result.answer,
        )

        print(
            "Actual Citations:",
            cited_chunk_ids,
        )

        if not result.answerable:
            print(
                "❌ FAIL: "
                "Target Case 被错误拒答。"
            )

            all_passed = False

            continue

        precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    cited_chunk_ids
                ),
                gold_chunk_ids=(
                    case.citation_gold_chunk_ids
                ),
            )
        )

        recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    cited_chunk_ids
                ),
                gold_chunk_ids=(
                    case.citation_gold_chunk_ids
                ),
            )
        )

        print(
            "Citation Precision:",
            f"{precision:.4f}",
        )

        print(
            "Citation Recall:",
            f"{recall:.4f}",
        )

        passed = (
            precision == 1.0
            and recall == 1.0
        )

        if passed:
            print(
                "✅ PASS"
            )

        else:
            print(
                "❌ FAIL"
            )

            extra = tuple(
                sorted(
                    set(
                        cited_chunk_ids
                    )
                    - set(
                        case.citation_gold_chunk_ids
                    )
                )
            )

            missing = tuple(
                sorted(
                    set(
                        case.citation_gold_chunk_ids
                    )
                    - set(
                        cited_chunk_ids
                    )
                )
            )

            print(
                "Extra:",
                extra,
            )

            print(
                "Missing:",
                missing,
            )

            all_passed = False

    print()
    print("=" * 100)

    if all_passed:
        print(
            "✅ Targeted Citation Regression PASSED."
        )
    else:
        print(
            "❌ Targeted Citation Regression FAILED."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()