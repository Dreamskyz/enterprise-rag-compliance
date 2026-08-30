"""构建标准化法规 Chunk 数据集。"""

from pathlib import Path

from enterprise_rag.ingestion.chunk_store import (
    write_chunks_jsonl,
)
from enterprise_rag.ingestion.chunker import (
    build_regulation_chunks,
)
from enterprise_rag.ingestion.document_builder import (
    build_normalized_document,
)
from enterprise_rag.ingestion.manifest import (
    load_manifest,
)
from enterprise_rag.ingestion.models import KnowledgeChunk
from enterprise_rag.ingestion.regulation_parser import (
    parse_regulation,
)


def main() -> None:
    manifest_path = Path(
        "data/manifest/documents.yaml"
    )

    output_path = Path(
        "data/processed/chunks.jsonl"
    )

    manifests = load_manifest(
        manifest_path
    )

    all_chunks: list[KnowledgeChunk] = []

    for manifest in manifests:
        # 被禁用的文档不进入知识库。
        if not manifest.enabled:
            continue

        # 当前阶段只处理法规类型。
        if manifest.document_type != "regulation":
            continue

        print(
            f"正在处理：{manifest.title}"
        )

        document = build_normalized_document(   #构建标准化文档
            manifest
        )

        chapters = parse_regulation(            #解析法规结构
            document.text
        )

        chunks = build_regulation_chunks(       #构建 Chunk
            document=document,
            chapters=chapters,
        )

        print(
            f"  生成 Chunk：{len(chunks)}"
        )

        all_chunks.extend(chunks)               #把当前文档 Chunk 加入总列表  append：把整体作为一个元素加入 extend：把列表里的元素一个一个加入

    write_chunks_jsonl(
        chunks=all_chunks,
        output_path=output_path,
    )

    print()
    print("=" * 80)
    print(
        f"Chunk 总数：{len(all_chunks)}"
    )
    print(
        f"输出文件：{output_path}"
    )


if __name__ == "__main__":
    main()