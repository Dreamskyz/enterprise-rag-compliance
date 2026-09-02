"""使用最新 Citation Gold Annotation 离线升级旧 Snapshot。"""

from pathlib import Path

from enterprise_rag.evaluation.answer_result_store import (
    read_answer_eval_results_jsonl,
    write_answer_eval_results_jsonl,
)
from enterprise_rag.evaluation.answer_snapshot_enrichment import (
    enrich_answer_eval_results,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)


DATASET_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

INPUT_SNAPSHOT = Path(
    "data/eval/results/"
    "answer_eval_v1_run_001.jsonl"
)

OUTPUT_SNAPSHOT = Path(
    "data/eval/results/"
    "answer_eval_v1_run_001_audited.jsonl"
)


def main() -> None:
    """
    使用新的 Citation Annotation
    升级旧 Full-RAG Snapshot。

    注意：

    本脚本不会：

        - 初始化 BGE-M3；
        - 初始化 Reranker；
        - 查询 Qdrant；
        - 调用 SiliconFlow。

    只做纯离线 Annotation Merge。
    """

    print("=" * 100)
    print(
        "Answer Evaluation Snapshot Enrichment"
    )
    print("=" * 100)

    # ======================================================
    # 1. 读取 Citation-aware Dataset。
    # ======================================================

    cases = (
        read_retrieval_eval_jsonl(
            DATASET_PATH
        )
    )

    print(
        "Dataset cases:",
        len(cases),
    )

    # ======================================================
    # 2. 读取旧 Snapshot。
    # ======================================================

    results = (
        read_answer_eval_results_jsonl(
            INPUT_SNAPSHOT
        )
    )

    print(
        "Snapshot cases:",
        len(results),
    )

    # ======================================================
    # 3. 注入新的人工 Annotation。
    # ======================================================

    enriched_results = (
        enrich_answer_eval_results(
            results=results,
            cases=cases,
        )
    )

    # ======================================================
    # 4. 保存新 Snapshot。
    #
    # 不覆盖旧 run_001，
    # 方便追溯实验历史。
    # ======================================================

    write_answer_eval_results_jsonl(
        results=tuple(
            enriched_results
        ),
        output_path=(
            OUTPUT_SNAPSHOT
        ),
    )

    print()
    print(
        "Audited snapshot created:"
    )

    print(
        OUTPUT_SNAPSHOT
    )

    print(
        "Cases:",
        len(enriched_results),
    )

    print()

    print(
        "✅ No LLM inference was used."
    )


if __name__ == "__main__":
    main()