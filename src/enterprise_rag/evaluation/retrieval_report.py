"""Retrieval Evaluation 结果打印。"""

from collections.abc import (
    Sequence,
)

from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethodEvalResult,
)


def print_retrieval_summary(
    results: Sequence[
        RetrievalMethodEvalResult
    ],
    evaluation_ks: Sequence[int],
) -> None:
    """
    以简单文本表格形式打印 Retrieval Ablation。

    当前先用于命令行检查。
    Part 4 再负责持久化正式结果。
    """

    normalized_ks = sorted(
        set(
            evaluation_ks
        )
    )

    header_parts = [
        "Method",
    ]

    for k in normalized_ks:
        header_parts.append(
            f"Recall@{k}"
        )

    header_parts.extend([
        f"MRR@{max(normalized_ks)}",
        "Mean Latency(ms)",
    ])

    print()

    print(
        " | ".join(
            header_parts
        )
    )

    print(
        " | ".join(
            "---"
            for _ in header_parts
        )
    )

    for result in results:
        row = [
            result.method.value
        ]

        for k in normalized_ks:
            metrics = (
                result.aggregate_by_k[k]
            )

            row.append(
                f"{metrics.mean_recall:.4f}"
            )

        mrr_k = max(
            normalized_ks
        )

        row.append(
            f"{result.aggregate_by_k[mrr_k].mrr:.4f}"
        )

        row.append(
            f"{result.mean_latency_ms:.2f}"
        )

        print(
            " | ".join(
                row
            )
        )