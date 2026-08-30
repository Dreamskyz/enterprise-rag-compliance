"""验证生成后的 chunks.jsonl 数据质量。"""

from pathlib import Path

from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)
from enterprise_rag.ingestion.validator import (
    validate_chunks,
)


def main() -> None:
    path = Path(
        "data/processed/chunks.jsonl"
    )

    chunks = read_chunks_jsonl(path)

    errors = validate_chunks(chunks)

    print("=" * 80)
    print(f"Chunk 总数：{len(chunks)}")
    print("=" * 80)

    if not errors:
        print("✅ 数据质量校验通过")
        return

    print(
        f"❌ 数据质量校验失败，共发现 {len(errors)} 个问题"
    )

    for error in errors:
        print("-", error)

    raise SystemExit(1)


if __name__ == "__main__":
    main()