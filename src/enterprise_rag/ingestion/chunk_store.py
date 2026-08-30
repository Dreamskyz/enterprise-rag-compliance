"""KnowledgeChunk 的 JSONL 持久化。"""

import json
from dataclasses import asdict
from pathlib import Path

from enterprise_rag.ingestion.models import KnowledgeChunk


def write_chunks_jsonl(
    chunks: list[KnowledgeChunk],
    output_path: Path,
) -> None:
    """
    将 KnowledgeChunk 列表保存为 JSONL。

    每一行对应一个独立 Chunk。
    """

    # 如果 processed 目录不存在，则自动创建。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",            #如果文件不存在，就创建；如果文件已经存在，就清空并重新写
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            # dataclass -> dict
            data = asdict(chunk)

            line = json.dumps(
                data,
                ensure_ascii=False,
            )

            file.write(line)
            file.write("\n")


def read_chunks_jsonl(
    input_path: Path,
) -> list[KnowledgeChunk]:
    """
    从 JSONL 文件读取 KnowledgeChunk。

    每一行必须对应一个完整的 Chunk JSON 对象。
    """

    chunks: list[KnowledgeChunk] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            # 防止意外空行影响解析。
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 第 {line_number} 行不是合法 JSON"
                ) from exc

            chunks.append(
                KnowledgeChunk(**data)
            )

    return chunks