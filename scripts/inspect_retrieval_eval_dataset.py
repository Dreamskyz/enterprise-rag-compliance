"""检查 Retrieval Evaluation Dataset。"""

from collections import Counter
from pathlib import Path

from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)


EVAL_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


def main() -> None:
    """
    检查：

    1. Dataset 能正常读取；
    2. Query ID 唯一；
    3. Answerable / Unanswerable 数量；
    4. Category 分布；
    5. 所有 Gold Chunk ID 真实存在。
    """

    print("=" * 100)
    print(
        "Retrieval Evaluation Dataset Check"
    )
    print("=" * 100)

    cases = (
        read_retrieval_eval_jsonl(
            EVAL_PATH
        )
    )

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    answerable_count = sum(
        1
        for case in cases
        if case.answerable
    )

    unanswerable_count = (
        len(cases)
        - answerable_count
    )

    category_counts = Counter(
        case.category.value
        for case in cases
    )

    missing_gold_ids: list[
        tuple[str, str]
    ] = []

    for case in cases:
        for gold_chunk_id in (
            case.gold_chunk_ids
        ):
            if (
                gold_chunk_id
                not in chunk_ids
            ):
                missing_gold_ids.append(
                    (
                        case.query_id,
                        gold_chunk_id,
                    )
                )

    print(
        "Eval case count:",
        len(cases),
    )

    print(
        "Answerable:",
        answerable_count,
    )

    print(
        "Unanswerable:",
        unanswerable_count,
    )

    print()

    print(
        "Category distribution:"
    )

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()

    print(
        "Knowledge chunk count:",
        len(chunks),
    )

    print(
        "Missing gold chunk IDs:",
        len(missing_gold_ids),
    )

    if missing_gold_ids:
        print()

        for (
            query_id,
            chunk_id,
        ) in missing_gold_ids:
            print(
                f"  {query_id}: "
                f"{chunk_id}"
            )

        raise AssertionError(
            "Evaluation Dataset "
            "存在无效 Gold Chunk ID"
        )

    print()

    print(
        "✅ Retrieval Evaluation Dataset "
        "基础校验通过"
    )


if __name__ == "__main__":
    main()