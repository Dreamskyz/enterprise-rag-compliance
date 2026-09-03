"""检查 Corpus Chunk 的整体质量与分布。"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ==========================================================
# Corpus Inspection 配置
# ==========================================================

# 正式 Chunk 文件。
CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)

# Generic Section Chunker 当前默认上限。
#
# 这里显式写出来，
# 是为了检查是否出现大量刚好顶到 Hard Split 上限的 Chunk。
GENERIC_MAX_CHARS = 1200

# 当前只是经验性的“短 Chunk”观察阈值。
#
# 注意：
#
# 80 并不是经过 Retrieval Eval
# 证明的最优阈值。
#
# 它只用于 inspection：
#
#     找出值得人工看的异常短 Chunk。
SHORT_CHUNK_THRESHOLD = 80


def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    """
    从 JSONL 文件加载 Chunk。

    当前 inspection script
    故意直接读取 JSON dict，
    而不是重新构造 KnowledgeChunk。

    原因：

    1. Inspection 的对象就是最终持久化结果；
    2. 可以直接检查磁盘中的真实数据；
    3. 不额外依赖 dataclass 构造逻辑；
    4. 以后 Schema 新增字段时更容易兼容。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk 文件不存在：{path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            # 理论上 chunks.jsonl 不应该有空行。
            #
            # 这里忽略空行，
            # 避免 inspection 因人为编辑文件而直接崩溃。
            if not line:
                continue

            try:
                row = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "无法解析 JSONL："
                    f"line={line_number}"
                ) from exc

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Chunk 文件为空"
        )

    return rows


def percentile(
    values: list[int],
    ratio: float,
) -> float:
    """
    计算简单线性插值百分位数。

    为什么不引入 numpy？

    因为这个 inspection script
    只做很轻量的统计，

    使用标准库即可完成，
    没必要增加额外运行依赖。

    ratio 示例：

        0.50 -> P50
        0.95 -> P95
    """

    if not values:
        raise ValueError(
            "percentile values 不能为空"
        )

    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "ratio 必须位于 [0, 1]"
        )

    sorted_values = sorted(
        values
    )

    # 只有一个元素时，
    # 所有百分位都只能是它自己。
    if len(sorted_values) == 1:
        return float(
            sorted_values[0]
        )

    # 将百分位位置映射到：
    #
    # [0, len(values) - 1]
    position = (
        len(sorted_values) - 1
    ) * ratio

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = (
        position - lower_index
    )

    lower_value = sorted_values[
        lower_index
    ]

    upper_value = sorted_values[
        upper_index
    ]

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def print_counter(
    title: str,
    counter: Counter[Any],
) -> None:
    """
    统一打印 Counter 统计结果。
    """

    print()
    print(
        title
    )
    print(
        "-" * 80
    )

    for key, count in counter.items():
        print(
            f"{key}: {count}"
        )


def inspect_length_distribution(
    rows: list[dict[str, Any]],
) -> None:
    """
    检查 Chunk 正文长度分布。

    当前统一使用：

        len(content)

    即 Python 字符数。

    这与 Generic Section Chunker
    当前 max_chars 的定义一致。
    """

    lengths = [
        len(
            row["content"]
        )
        for row in rows
    ]

    print()
    print(
        "Chunk Length Distribution"
    )
    print(
        "-" * 80
    )

    print(
        f"min  : {min(lengths)}"
    )

    print(
        "mean : "
        f"{statistics.mean(lengths):.2f}"
    )

    print(
        "p50  : "
        f"{percentile(lengths, 0.50):.2f}"
    )

    print(
        "p95  : "
        f"{percentile(lengths, 0.95):.2f}"
    )

    print(
        f"max  : {max(lengths)}"
    )


def inspect_hard_split_candidates(
    rows: list[dict[str, Any]],
) -> None:
    """
    找出正文长度刚好等于 max_chars 的 Chunk。

    对 Generic Section 文档来说，
    这通常值得人工检查：

    - 是否是单段落过长；
    - 是否代码块过长；
    - 是否上游 paragraph boundary 丢失；
    - 是否确实属于合理 hard split。

    注意：

    regulation 使用自己的 Chunker，
    所以这里只观察 Generic Section 类型。
    """

    candidates = [
        row
        for row in rows
        if (
            row["document_type"]
            in {
                "security_guideline",
                "technical_documentation",
            }
            and len(
                row["content"]
            )
            == GENERIC_MAX_CHARS
        )
    ]

    print()
    print(
        "Exactly Max-Char Chunks"
    )
    print(
        "-" * 80
    )

    print(
        "count: "
        f"{len(candidates)}"
    )

    for row in candidates:
        print()
        print(
            f"chunk_id: {row['chunk_id']}"
        )
        print(
            "document_id: "
            f"{row['document_id']}"
        )
        print(
            "section_path: "
            f"{row.get('section_path')}"
        )
        print(
            "content_length: "
            f"{len(row['content'])}"
        )

        # 只打印前 300 字符，
        # 避免 inspection 输出过长。
        preview = (
            row["content"][:300]
            .replace(
                "\n",
                "\\n",
            )
        )

        print(
            f"preview: {preview}"
        )


def inspect_short_chunks(
    rows: list[dict[str, Any]],
) -> None:
    """
    找出异常短 Chunk。

    这里只标记，不自动删除。

    原因：

    一个很短的 Chunk
    可能是：

    - 噪声；
    - Section 尾部碎片；

    也可能是：

    - 独立定义；
    - 简短但关键的规则；
    - 很短的安全建议。

    所以最终是否需要处理，
    必须结合 section_path 和 content 人工判断。
    """

    candidates = [
        row
        for row in rows
        if len(
            row["content"]
        ) < SHORT_CHUNK_THRESHOLD
    ]

    print()
    print(
        "Short Chunks"
    )
    print(
        "-" * 80
    )

    print(
        "threshold: "
        f"< {SHORT_CHUNK_THRESHOLD}"
    )

    print(
        "count: "
        f"{len(candidates)}"
    )

    for row in candidates:
        print()
        print(
            f"chunk_id: {row['chunk_id']}"
        )
        print(
            "document_id: "
            f"{row['document_id']}"
        )
        print(
            "section_path: "
            f"{row.get('section_path')}"
        )
        print(
            "length: "
            f"{len(row['content'])}"
        )

        preview = (
            row["content"][:200]
            .replace(
                "\n",
                "\\n",
            )
        )

        print(
            f"content: {preview}"
        )


def inspect_duplicate_content(
    rows: list[dict[str, Any]],
) -> None:
    """
    按 content_hash 检查完全相同正文。

    注意：

    duplicate content
    不等于 duplicate chunk。

    例如技术文档可能在不同 Section
    重复展示同一个完整代码示例。

    这种情况下：

        content 一样
        section_path 不一样

    仍然可能有检索语义。

    所以这里只报告，
    不自动去重。
    """

    rows_by_hash: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for row in rows:
        rows_by_hash[
            row["content_hash"]
        ].append(
            row
        )

    duplicate_groups = [
        group
        for group in rows_by_hash.values()
        if len(group) > 1
    ]

    print()
    print(
        "Duplicate Content Groups"
    )
    print(
        "-" * 80
    )

    print(
        "groups: "
        f"{len(duplicate_groups)}"
    )

    for group_index, group in enumerate(
        duplicate_groups,
        start=1,
    ):
        print()
        print(
            f"[Group {group_index}]"
        )

        print(
            "content_hash: "
            f"{group[0]['content_hash']}"
        )

        for row in group:
            print(
                "  - "
                f"{row['chunk_id']}"
            )

            print(
                "    section_path: "
                f"{row.get('section_path')}"
            )

        preview = (
            group[0]["content"][:250]
            .replace(
                "\n",
                "\\n",
            )
        )

        print(
            f"  preview: {preview}"
        )


def inspect_document_lengths(
    rows: list[dict[str, Any]],
) -> None:
    """
    按文档分别查看 Chunk 长度统计。

    全局统计只能告诉我们整体情况。

    但 Corpus V2 是异构语料：

        regulation
        security_guideline
        technical_documentation

    不同文档可能存在完全不同的长度分布。

    因此需要继续按 document_id 拆开观察。
    """

    lengths_by_document: dict[
        str,
        list[int],
    ] = defaultdict(
        list
    )

    for row in rows:
        lengths_by_document[
            row["document_id"]
        ].append(
            len(
                row["content"]
            )
        )

    print()
    print(
        "Length Distribution By Document"
    )
    print(
        "-" * 80
    )

    for (
        document_id,
        lengths,
    ) in lengths_by_document.items():

        print(
            f"{document_id}:"
        )

        print(
            "  count="
            f"{len(lengths)}, "
            "min="
            f"{min(lengths)}, "
            "mean="
            f"{statistics.mean(lengths):.2f}, "
            "p50="
            f"{percentile(lengths, 0.50):.2f}, "
            "p95="
            f"{percentile(lengths, 0.95):.2f}, "
            "max="
            f"{max(lengths)}"
        )


def main() -> None:
    """
    执行完整 Corpus Inspection。
    """

    rows = load_chunks(
        CHUNKS_PATH
    )

    print(
        "=" * 80
    )
    print(
        "Corpus Inspection"
    )
    print(
        "=" * 80
    )

    print(
        f"Chunk file: {CHUNKS_PATH}"
    )

    print(
        f"Total chunks: {len(rows)}"
    )

    # --------------------------------------------------
    # 1. 文档分布
    # --------------------------------------------------

    document_counter = Counter(
        row["document_id"]
        for row in rows
    )

    print_counter(
        title="Chunks By Document",
        counter=document_counter,
    )

    # --------------------------------------------------
    # 2. 文档类型分布
    # --------------------------------------------------

    document_type_counter = Counter(
        row["document_type"]
        for row in rows
    )

    print_counter(
        title="Chunks By Document Type",
        counter=document_type_counter,
    )

    # --------------------------------------------------
    # 3. ACL 分布
    # --------------------------------------------------

    acl_counter = Counter(
        row["access_level"]
        for row in rows
    )

    print_counter(
        title="Chunks By ACL",
        counter=acl_counter,
    )

    # --------------------------------------------------
    # 4. 全局长度分布
    # --------------------------------------------------

    inspect_length_distribution(
        rows
    )

    # --------------------------------------------------
    # 5. 每篇文档长度分布
    # --------------------------------------------------

    inspect_document_lengths(
        rows
    )

    # --------------------------------------------------
    # 6. Hard Split 候选
    # --------------------------------------------------

    inspect_hard_split_candidates(
        rows
    )

    # --------------------------------------------------
    # 7. 极短 Chunk
    # --------------------------------------------------

    inspect_short_chunks(
        rows
    )

    # --------------------------------------------------
    # 8. 完全重复正文
    # --------------------------------------------------

    inspect_duplicate_content(
        rows
    )


if __name__ == "__main__":
    main()