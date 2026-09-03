"""构建标准化 KnowledgeChunk 数据集。"""

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
from enterprise_rag.ingestion.generic_section_chunker import (
    build_generic_section_chunks,
)
from enterprise_rag.ingestion.generic_section_parser import (
    parse_generic_sections,
)
from enterprise_rag.ingestion.manifest import (
    DocumentManifest,
    load_manifest,
)
from enterprise_rag.ingestion.models import KnowledgeChunk
from enterprise_rag.ingestion.regulation_parser import (
    parse_regulation,
)
from enterprise_rag.ingestion.validator import (
    validate_chunks,
)


# ----------------------------------------------------------
# 使用 Generic Section Parser / Chunker 的文档类型
# ----------------------------------------------------------
#
# security_guideline：
#     OWASP 等安全规范。
#
# technical_documentation：
#     FastAPI / Qdrant 等技术文档。
#
# 二者虽然业务语义不同，
# 但经过各自 Loader 后都已经转换成：
#
#     Markdown-like structured text
#
# 所以下游可以复用：
#
#     Generic Section Parser
#     Generic Section Chunker
#
# 这里显式声明这个集合，
# 可以避免为每个 Section 型文档复制一套相同路由。
GENERIC_SECTION_DOCUMENT_TYPES = {
    "security_guideline",
    "technical_documentation",
}


def build_chunks_for_document(
    manifest: DocumentManifest,
) -> list[KnowledgeChunk]:
    """
    根据 document_type
    选择对应的解析和切分流程。

    当前支持：

    1. regulation

       -> CAC HTML Loader
       -> NormalizedDocument
       -> Regulation Parser
       -> Regulation Chunker

    2. security_guideline

       -> OWASP HTML Loader
       -> NormalizedDocument
       -> Generic Section Parser
       -> Generic Section Chunker

    3. technical_documentation

       -> FastAPI HTML Loader
       -> NormalizedDocument
       -> Generic Section Parser
       -> Generic Section Chunker

    这个函数就是当前 ingestion 阶段的：

        Document-Type Router

    需要特别注意：

    document_type 表达的是知识语义类型，
    而不是原始文件格式。

    所以：

        security_guideline
        technical_documentation

    完全可以共享同一种 Section 结构处理流程。
    """

    # --------------------------------------------------
    # 0. 所有文档先进入统一 NormalizedDocument
    # --------------------------------------------------
    #
    # 在这一层之前：
    #
    # - CAC 有自己的正文 DOM；
    # - OWASP 有自己的正文 DOM；
    # - FastAPI 有自己的正文 DOM；
    #
    # 这些异构差异已经由
    # document_builder + loader 解决。
    #
    # 从这里开始，
    # Parser / Chunker 只面对统一的数据对象。
    document = build_normalized_document(
        manifest
    )

    # --------------------------------------------------
    # 1. 法规文档
    # --------------------------------------------------
    if (
        manifest.document_type
        == "regulation"
    ):
        # 法规的核心结构是：
        #
        # Chapter
        #     ↓
        # Article
        #
        # 例如：
        #
        # 第一章 总则
        # 第一条 ...
        #
        # 所以走专用法规 Parser。
        chapters = parse_regulation(
            document.text
        )

        # 当前法规切分策略：
        #
        # 一条 Article 对应一个 KnowledgeChunk。
        return build_regulation_chunks(
            document=document,
            chapters=chapters,
        )

    # --------------------------------------------------
    # 2. Generic Section 类型文档
    # --------------------------------------------------
    if (
        manifest.document_type
        in GENERIC_SECTION_DOCUMENT_TYPES
    ):
        # 当前进入这里的包括：
        #
        # security_guideline
        # technical_documentation
        #
        # 它们的 Loader 已经把 HTML
        # 转成类似：
        #
        # # Root
        #
        # ## Section
        #
        # ### Subsection
        #
        # Paragraph...
        #
        # [CODE]
        # | ...
        #
        # 因此这里可以使用完全相同的
        # Generic Section Parser。
        sections = parse_generic_sections(
            document.text
        )

        # Parser 负责恢复：
        #
        # section_title
        # section_path
        # heading level
        # section content
        #
        # Chunker 再根据：
        #
        # - Section 边界；
        # - Paragraph 边界；
        # - max_chars；
        #
        # 构建最终 KnowledgeChunk。
        return build_generic_section_chunks(
            document=document,
            sections=sections,
        )

    # --------------------------------------------------
    # 3. 暂未支持的文档类型
    # --------------------------------------------------
    #
    # 不设置“默认 Parser”。
    #
    # 因为新的 document_type
    # 如果没有明确选择正确的 Parser / Chunker，
    # 静默继续处理反而可能制造错误 Chunk。
    #
    # 所以继续保持 fail-fast。
    raise ValueError(
        "当前 build pipeline 不支持的 "
        f"document_type：{manifest.document_type}"
    )


def main() -> None:
    """
    从 Manifest 构建整个知识库 Chunk 数据集。

    完整流程：

        Manifest
            ↓
        enabled documents
            ↓
        site-specific Loader
            ↓
        NormalizedDocument
            ↓
        Document-Type Router
            ↓
        Parser
            ↓
        Structure-aware Chunker
            ↓
        KnowledgeChunk[]
            ↓
        Global Validation
            ↓
        chunks.jsonl

    当前一个重要的数据质量原则是：

        validate before persistence

    即所有文档全部构建完成后，
    先统一执行 Validator。

    只有整个数据集完全通过，
    才允许覆盖正式 chunks.jsonl。

    这样可以避免错误 Chunk
    进一步污染后续 Qdrant。
    """

    manifest_path = Path(
        "data/manifest/documents.yaml"
    )

    output_path = Path(
        "data/processed/chunks.jsonl"
    )

    # --------------------------------------------------
    # 1. 加载 Manifest
    # --------------------------------------------------

    manifests = load_manifest(
        manifest_path
    )

    all_chunks: list[KnowledgeChunk] = []

    # --------------------------------------------------
    # 2. 逐文档构建 Chunk
    # --------------------------------------------------

    for manifest in manifests:
        # 被禁用的文档不进入当前知识库。
        if not manifest.enabled:
            continue

        print(
            f"正在处理：{manifest.title}"
        )

        print(
            "  document_type："
            f"{manifest.document_type}"
        )

        print(
            "  access_level："
            f"{manifest.access_level}"
        )

        # 通过统一路由函数，
        # 根据 document_type
        # 选择对应的：
        #
        # Parser + Chunker。
        chunks = build_chunks_for_document(
            manifest
        )

        print(
            f"  生成 Chunk：{len(chunks)}"
        )

        # extend：
        #
        # 当前 chunks 本身是：
        #
        # list[KnowledgeChunk]
        #
        # 所以这里不是把整个 list
        # 当成一个元素 append，
        # 而是把里面的每个 KnowledgeChunk
        # 加入全局数据集。
        all_chunks.extend(
            chunks
        )

    # --------------------------------------------------
    # 3. 写盘前统一做数据质量校验
    # --------------------------------------------------

    errors = validate_chunks(
        all_chunks
    )

    if errors:
        print()
        print(
            "=" * 80
        )

        print(
            "Chunk 数据校验失败："
        )

        for error in errors:
            print(
                f"- {error}"
            )

        # 不允许带着错误数据继续写入 chunks.jsonl。
        #
        # 这里 fail-fast，
        # 可以避免后续 Qdrant
        # 被错误数据污染。
        raise ValueError(
            "KnowledgeChunk 数据校验失败"
        )

    # --------------------------------------------------
    # 4. 校验通过后才真正持久化
    # --------------------------------------------------

    write_chunks_jsonl(
        chunks=all_chunks,
        output_path=output_path,
    )

    # --------------------------------------------------
    # 5. 输出构建摘要
    # --------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        f"Chunk 总数：{len(all_chunks)}"
    )

    print(
        f"输出文件：{output_path}"
    )

    print(
        "数据校验：PASS"
    )


if __name__ == "__main__":
    main()