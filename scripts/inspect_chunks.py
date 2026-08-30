"""检查法规 Chunk 构建结果。"""

from pathlib import Path

from enterprise_rag.ingestion.chunker import (
    build_regulation_chunks,
)
from enterprise_rag.ingestion.document_builder import (
    build_normalized_document,
)
from enterprise_rag.ingestion.manifest import (
    load_manifest,
)
from enterprise_rag.ingestion.regulation_parser import (
    parse_regulation,
)


def main() -> None:
    manifests = load_manifest(
        Path("data/manifest/documents.yaml")
    )

    #manifest = manifests[0]
    manifest = manifests[1]

    document = build_normalized_document(
        manifest
    )

    chapters = parse_regulation(
        document.text
    )

    chunks = build_regulation_chunks(
        document=document,
        chapters=chapters,
    )

    print("Chunk 数量：", len(chunks))

    for chunk in chunks:
        print(
            chunk.chunk_id,
            "|",
            chunk.chapter_number,
            chunk.article_number,
            "|",
            len(chunk.content),
        )

    lengths = [
        len(chunk.content)
        for chunk in chunks
    ]

    print()
    print("=" * 80)
    print("Chunk 长度统计")
    print("=" * 80)

    print("最短：", min(lengths))
    print("最长：", max(lengths))
    print(
        "平均：",
        round(sum(lengths) / len(lengths), 2),
    )


if __name__ == "__main__":
    main()