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
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
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
       -> FastAPI / Qdrant HTML Loader
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
    # - Qdrant 有自己的正文 DOM；
    #
    # 这些异构差异已经由：
    #
    # document_builder + loader
    #
    # 解决。
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
        # 法规的核心内部结构是：
        #
        # Chapter
        #     ↓
        # Article
        #
        # 对存在显式章节的法规，例如：
        #
        # 第一章 总则
        # 第一条 ...
        #
        # Parser 使用真实 Chapter。
        #
        # 对没有显式章节的法规，例如：
        #
        # 第一条 ...
        # 第二条 ...
        #
        # Parser 会创建一个内部 implicit chapter，
        # 保证下游仍然使用统一：
        #
        # Chapter -> Article
        #
        # 数据结构。

        chapters = parse_regulation(
            document.text
        )

        # 当前法规切分策略：
        #
        # 一个 Article
        # 对应一个 KnowledgeChunk。

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


def validate_document_coverage(
    manifests: list[DocumentManifest],
    chunks: list[KnowledgeChunk],
) -> list[str]:
    """
    校验 Manifest 和最终 Corpus 之间的
    document-level coverage。

    解决的问题：

        Chunk Validation PASS

    并不意味着：

        每篇 enabled 文档
        都成功产生了 Chunk。

    例如：

        Manifest 中有 28 篇 enabled 文档，
        其中 1 篇因为 Parser 结构不兼容
        生成了 0 个 Chunk。

    如果只验证已经存在的 Chunk，
    整个数据集仍然可能显示：

        PASS

    但实际上 Corpus 已经缺失了一篇文档。

    因此这里额外验证：

        enabled_document_ids
        ==
        chunk_document_ids

    当前主要检查两种错误：

    1. Missing Document

       Manifest 中 enabled，
       但最终没有任何 Chunk。

    2. Unexpected Document

       Chunk 中出现了一个
       当前 enabled Manifest 中不存在的 document_id。

    返回：

        list[str]

    空列表：
        Coverage PASS。

    非空列表：
        Coverage FAIL。
    """

    errors: list[str] = []

    # --------------------------------------------------
    # 1. Manifest 期望进入 Corpus 的文档
    # --------------------------------------------------

    enabled_document_ids = {
        manifest.document_id
        for manifest in manifests
        if manifest.enabled
    }

    # --------------------------------------------------
    # 2. 实际进入 Chunk Corpus 的文档
    # --------------------------------------------------

    chunk_document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    # --------------------------------------------------
    # 3. Missing Documents
    # --------------------------------------------------
    #
    # Manifest 要求存在，
    # 但是最终一个 Chunk 都没有。
    #
    # 这通常意味着：
    #
    # Loader
    # Parser
    # Chunker
    #
    # 某一层静默地产生了空结果。

    missing_document_ids = sorted(
        enabled_document_ids
        - chunk_document_ids
    )

    for document_id in missing_document_ids:
        errors.append(
            "Enabled Manifest 文档没有生成任何 Chunk："
            f"{document_id}"
        )

    # --------------------------------------------------
    # 4. Unexpected Documents
    # --------------------------------------------------
    #
    # 正常 build pipeline 中理论上不应发生。
    #
    # 但如果未来存在：
    #
    # stale data
    # 外部 Chunk 合并
    # pipeline bug
    #
    # 这里可以第一时间发现。

    unexpected_document_ids = sorted(
        chunk_document_ids
        - enabled_document_ids
    )

    for document_id in unexpected_document_ids:
        errors.append(
            "Chunk Corpus 中存在未启用或不存在于 Manifest "
            f"的 document_id：{document_id}"
        )

    return errors


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
        Document Coverage Validation
            ↓
        Chunk-level Validation
            ↓
        chunks.jsonl

    当前两个重要的数据质量原则：

    1. validate before persistence

       所有数据通过校验之后，
       才允许覆盖正式 chunks.jsonl。

    2. validity != completeness

       每个 Chunk 本身合法，
       不代表 Manifest 中所有文档
       都完整进入了知识库。

    所以当前同时验证：

        Document Coverage
        +
        Chunk Validity
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

        # 注意：
        #
        # 这里暂时不直接因为：
        #
        #     len(chunks) == 0
        #
        # 就立即 raise。
        #
        # 原因是我们希望先把所有文档跑完，
        # 再通过 Document Coverage Validation
        # 一次性打印出所有缺失文档。
        #
        # 如果未来同时有 3 篇文档失败，
        # 用户不需要修一篇、重跑一次，
        # 才发现下一篇。

        all_chunks.extend(
            chunks
        )

    # --------------------------------------------------
    # 3. Document-level Coverage Validation
    # --------------------------------------------------

    coverage_errors = (
        validate_document_coverage(
            manifests=manifests,
            chunks=all_chunks,
        )
    )

    if coverage_errors:
        print()
        print(
            "=" * 80
        )

        print(
            "文档覆盖完整性校验失败："
        )

        for error in coverage_errors:
            print(
                f"- {error}"
            )

        # ------------------------------------------------
        # Fail Fast
        # ------------------------------------------------
        #
        # 即使所有已生成 Chunk
        # 在 schema 层面都是合法的，
        # 只要 Manifest 中存在 enabled 文档
        # 没有成功进入 Corpus，
        #
        # 就不允许覆盖正式 chunks.jsonl。

        raise ValueError(
            "Manifest -> Corpus "
            "文档覆盖完整性校验失败"
        )

    # --------------------------------------------------
    # 4. Chunk-level Validation
    # --------------------------------------------------
    #
    # Coverage Validation 回答：
    #
    #     文档有没有全部进来？
    #
    # validate_chunks 回答：
    #
    #     进来的 Chunk 本身是否合法？
    #
    # 两者职责不同，不能互相替代。

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
    # 5. 校验通过后才真正持久化
    # --------------------------------------------------

    write_chunks_jsonl(
        chunks=all_chunks,
        output_path=output_path,
    )

    # --------------------------------------------------
    # 6. 输出构建摘要
    # --------------------------------------------------

    enabled_document_count = sum(
        1
        for manifest in manifests
        if manifest.enabled
    )

    chunk_document_count = len(
        {
            chunk.document_id
            for chunk in all_chunks
        }
    )

    print()

    print(
        "=" * 80
    )

    print(
        f"Enabled 文档数："
        f"{enabled_document_count}"
    )

    print(
        f"进入 Corpus 的文档数："
        f"{chunk_document_count}"
    )

    print(
        f"Chunk 总数：{len(all_chunks)}"
    )

    print(
        f"输出文件：{output_path}"
    )

    print(
        "文档覆盖校验：PASS"
    )

    print(
        "Chunk 数据校验：PASS"
    )


if __name__ == "__main__":
    main()