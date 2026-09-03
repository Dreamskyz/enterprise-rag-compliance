"""通用 Section 文档 Chunk 构建模块。"""

import re

from enterprise_rag.ingestion.chunker import (
    build_content_hash,
)
from enterprise_rag.ingestion.models import (
    GenericSection,
    KnowledgeChunk,
    NormalizedDocument,
)


# ==========================================================
# Generic Section Chunker 默认字符预算。
#
# 当前 Day 6 V1 使用字符数，而不是 Token 数。
#
# 原因：
#
# 1. 实现简单；
# 2. 行为确定；
# 3. 不绑定具体 tokenizer；
# 4. 中英文都能够工作；
# 5. 方便先建立 Corpus v2 baseline。
#
# 后续会通过 Retrieval Regression
# 判断是否需要升级为 tokenizer-aware splitting。
# ==========================================================

DEFAULT_MAX_CHARS = 1200


def build_section_slug(
    section_path: str,
) -> str:
    """
    将 Section Path 转换成稳定的 Chunk ID 片段。

    示例：

        FastAPI > Dependencies > Classes as Dependencies

    转换为：

        fastapi_dependencies_classes_as_dependencies

    设计目标：

    1. 确定性：

       相同 Section Path 每次得到相同结果；

    2. 可读性：

       Debug 和人工编写 Retrieval Gold 时
       可以大致看出 Chunk 来自哪个 Section；

    3. 降低冲突概率：

       使用完整 section_path，
       而不是只使用 section_title，
       降低不同父级下同名 Section
       发生 ID 冲突的风险。

    注意：

    完整 section_path 并不能绝对保证唯一。

    真实技术文档可能合法存在：

        同一个父级
        +
        同名 Heading

    例如 FastAPI 官方 Dependencies 页面中
    就出现了两个：

        Dependencies > Integrated with OpenAPI

    因此最终唯一性不能只依赖 slug，
    还需要在 Chunk 构建阶段增加
    occurrence disambiguation。

    中文标题同样可以保留。

    例如：

        安全规范 > 输入验证

    会得到类似：

        安全规范_输入验证
    """

    normalized = section_path.strip().lower()

    # 将连续的非“字母 / 数字 / 下划线”字符
    # 统一替换成一个下划线。
    #
    # Python 的 \w 默认支持 Unicode，
    # 因而中文不会被全部删除。
    slug = re.sub(
        r"[^\w]+",
        "_",
        normalized,
    )

    # 删除开头和结尾可能出现的下划线。
    slug = slug.strip("_")

    if not slug:
        raise ValueError(
            "section_path 无法生成有效 slug"
        )

    return slug


def build_disambiguated_section_slug(
    section_slug: str,
    occurrence: int,
) -> str:
    """
    为重复出现的 Section Slug
    构造稳定、可读的 ID 消歧片段。

    第一次出现：

        dependencies_integrated_with_openapi

    第二次出现：

        dependencies_integrated_with_openapi__occ02

    第三次出现：

        dependencies_integrated_with_openapi__occ03

    为什么第一次不追加：

        __occ01

    因为我们希望尽量保持已经存在的
    非重复 Section Chunk ID 不变。

    例如以前：

        owasp_xxx__mitigation_strategies__0001

    不应该因为这次修复 FastAPI 重复标题问题
    而变成：

        owasp_xxx__mitigation_strategies__occ01__0001

    这样可以减少无意义的 Chunk ID 漂移。

    occurrence 必须从 1 开始。
    """

    if occurrence <= 0:
        raise ValueError(
            "section occurrence 必须大于 0"
        )

    # 第一次出现继续沿用历史 ID。
    if occurrence == 1:
        return section_slug

    # 第二次及以后才增加消歧后缀。
    return (
        f"{section_slug}"
        f"__occ{occurrence:02d}"
    )


def split_section_content(
    content: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """
    将一个 Section 正文拆分为若干文本块。

    当前 V1 策略：

        Section Boundary
            >
        Paragraph Boundary
            >
        Hard Character Boundary

    具体规则：

    1. 空正文返回空列表；

    2. 正文不超过 max_chars：
       直接作为一个 part；

    3. 正文超过 max_chars：
       优先按照空行分隔出的段落进行聚合；

    4. 如果某一个单独段落本身就超过 max_chars：
       才使用字符级 Hard Split。

    注意：

    当前 max_chars 是工程 baseline，
    不是已经通过评测证明的最优参数。
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    normalized_content = content.strip()

    if not normalized_content:
        return []

    # 短 Section 无需拆分。
    if len(normalized_content) <= max_chars:
        return [
            normalized_content
        ]

    # ------------------------------------------------------
    # 1. 根据一个或多个空行切成段落。
    #
    # \n\s*\n 表示：
    #
    #     一个换行
    #     +
    #     中间可能有空白
    #     +
    #     另一个换行
    #
    # 可以处理：
    #
    #     Paragraph A
    #
    #     Paragraph B
    # ------------------------------------------------------

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            normalized_content,
        )
        if paragraph.strip()
    ]

    parts: list[str] = []

    # 当前正在聚合的段落。
    current_paragraphs: list[str] = []

    def flush_current_part() -> None:
        """
        将当前已聚合的段落正式保存为一个 part。
        """

        nonlocal current_paragraphs

        if not current_paragraphs:
            return

        part = "\n\n".join(
            current_paragraphs
        ).strip()

        if part:
            parts.append(
                part
            )

        current_paragraphs = []

    for paragraph in paragraphs:

        # --------------------------------------------------
        # 情况 A：
        #
        # 当前单独一个段落就已经超过 max_chars。
        #
        # 此时无法继续遵守 Paragraph Boundary，
        # 只能启用 Hard Character Split。
        # --------------------------------------------------

        if len(paragraph) > max_chars:

            # 先保存之前已经聚合的正常段落。
            flush_current_part()

            start = 0

            while start < len(paragraph):
                end = start + max_chars

                hard_part = (
                    paragraph[start:end]
                    .strip()
                )

                if hard_part:
                    parts.append(
                        hard_part
                    )

                start = end

            continue

        # --------------------------------------------------
        # 情况 B：
        #
        # 当前 paragraph 本身可以放进 Chunk。
        #
        # 尝试与已经聚合的段落组合。
        # --------------------------------------------------

        if not current_paragraphs:
            current_paragraphs.append(
                paragraph
            )

            continue

        candidate = "\n\n".join(
            [
                *current_paragraphs,
                paragraph,
            ]
        )

        # 加入当前 paragraph 后仍未超过预算，
        # 继续聚合。
        if len(candidate) <= max_chars:
            current_paragraphs.append(
                paragraph
            )

            continue

        # 加入后会超过预算：
        #
        # 先结束已有 part，
        # 当前 paragraph 成为下一个 part 的开始。
        flush_current_part()

        current_paragraphs.append(
            paragraph
        )

    # 文件末尾别忘了保存最后一个 part。
    flush_current_part()

    return parts


def build_generic_retrieval_text(
    document: NormalizedDocument,
    section: GenericSection,
    content: str,
) -> str:
    """
    构建 Generic Section Chunk 的检索文本。

    与法规 Chunker 的设计保持一致：

        content
            = 原始正文

        retrieval_text
            = 加入结构上下文后的检索文本

    当前结构：

        Document Title
        Section Path
        Content

    例如：

        FastAPI Dependencies
        Dependencies > Classes as Dependencies
        A Python class can be used as a dependency.

    注意：

    即使一个文档中出现两个完全相同的
    section_path，

    retrieval_text 仍然保持真实语义路径，
    不会人为加入：

        occ02

    因为 occurrence 是内部 ID 消歧信息，
    不是原始知识内容。

    Dense / BM25 / Reranker
    后续原则上都使用 retrieval_text。
    """

    return "\n".join(
        [
            document.title,
            section.path,
            content,
        ]
    )


def build_generic_section_chunks(
    document: NormalizedDocument,
    sections: list[GenericSection],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[KnowledgeChunk]:
    """
    将 GenericSection 列表转换成统一 KnowledgeChunk。

    适用文档：

        security_guideline
        technical_documentation

    当前 V1：

    1. 空 Section 不生成 Chunk；

    2. 短 Section：

           一个 Section = 一个 Chunk；

    3. 长 Section：

           优先按段落拆成多个 Chunk；

    4. 每个 Chunk 都保留：

           section_title
           section_path

    5. 法规专属字段统一为：

           None

    6. chunk_index：

           表示当前文档范围内的
           全局 Chunk 顺序；

    7. Chunk ID：

           document_id
           + disambiguated section slug
           + Section 内 part number。

    当同一个 slug 在一篇文档中重复出现时：

        第一次：

            section_slug

        第二次：

            section_slug__occ02

        第三次：

            section_slug__occ03

    occurrence 只用于 Chunk ID，
    不修改 section_title / section_path。
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    chunks: list[KnowledgeChunk] = []

    # chunk_index 是整篇文档范围内的顺序，
    # 而不是 Section 内部顺序。
    chunk_index = 0

    # ------------------------------------------------------
    # Section Slug 出现次数。
    # ------------------------------------------------------
    #
    # Key：
    #
    #     build_section_slug(section.path)
    #
    # Value：
    #
    #     当前 slug 在本文档中
    #     已经出现过多少次。
    #
    # 为什么按 slug 而不是原始 section.path 统计？
    #
    # 因为最终参与 chunk_id 的是 slug。
    #
    # 理论上两个不同原始路径也可能经过：
    #
    #     lowercase
    #     punctuation normalization
    #
    # 后得到同一个 slug。
    #
    # 因此必须在真正的 Identifier Namespace
    # 上做唯一性消歧。
    section_slug_occurrences: dict[
        str,
        int,
    ] = {}

    for section in sections:

        # --------------------------------------------------
        # 1. 先生成当前 Section 的基础 slug。
        # --------------------------------------------------
        #
        # occurrence 在 Section 层级统计，
        # 而不是在 Chunk part 层级统计。
        #
        # 同一个 Section 如果因为正文较长拆成：
        #
        #     0001
        #     0002
        #
        # 它们仍然属于同一个 occurrence。
        section_slug = build_section_slug(
            section.path
        )

        # 当前 slug 是第几次出现。
        occurrence = (
            section_slug_occurrences.get(
                section_slug,
                0,
            )
            + 1
        )

        section_slug_occurrences[
            section_slug
        ] = occurrence

        # 构造真正用于 Chunk ID 的
        # 消歧 slug。
        #
        # occurrence == 1：
        #
        #     original_slug
        #
        # occurrence == 2：
        #
        #     original_slug__occ02
        disambiguated_section_slug = (
            build_disambiguated_section_slug(
                section_slug=section_slug,
                occurrence=occurrence,
            )
        )

        # --------------------------------------------------
        # 2. 将 Section 正文拆成一个或多个 part。
        # --------------------------------------------------
        #
        # 空 Section 会得到 []，
        # 因而不会生成无正文 Chunk。
        #
        # 注意 occurrence 仍然已经被记录。
        #
        # 原因是该 Heading 确实存在于文档结构中，
        # 即使当前没有正文，
        # 它仍然是这个 slug 的一次真实出现。
        content_parts = split_section_content(
            content=section.content,
            max_chars=max_chars,
        )

        if not content_parts:
            continue

        # --------------------------------------------------
        # 3. 为当前 Section 的每个 part 构建 Chunk。
        # --------------------------------------------------
        #
        # part_index 是当前 Section 内部的序号。
        #
        # Chunk ID 使用 1-based：
        #
        #   0001
        #   0002
        #
        # 更符合人工阅读习惯。
        #
        # chunk_index 则仍然保持 0-based，
        # 与原法规 Chunker 一致。
        # --------------------------------------------------

        for part_index, content in enumerate(
            content_parts,
            start=1,
        ):
            chunk_id = (
                f"{document.document_id}"
                f"__{disambiguated_section_slug}"
                f"__{part_index:04d}"
            )

            retrieval_text = (
                build_generic_retrieval_text(
                    document=document,
                    section=section,
                    content=content,
                )
            )

            # 与法规 Chunker共用同一个 hash Contract：
            #
            # SHA-256(content)
            #
            # 不把：
            #
            # retrieval_text
            # Section Metadata
            # occurrence
            #
            # 混入正文 hash。
            content_hash = build_content_hash(
                content
            )

            chunk = KnowledgeChunk(
                chunk_id=chunk_id,

                # ==========================================
                # 文档身份。
                # ==========================================

                document_id=document.document_id,
                title=document.title,

                document_type=(
                    document.document_type
                ),
                language=document.language,
                version=document.version,

                # ==========================================
                # Generic 文档没有法规章 / 条结构。
                # 必须保持真正的 None，
                # 不能使用字符串 "None"。
                # ==========================================

                chapter_number=None,
                chapter_title=None,
                article_number=None,

                # ==========================================
                # 文本。
                # ==========================================

                content=content,
                retrieval_text=retrieval_text,

                # ==========================================
                # 来源和 ACL。
                # ==========================================

                source_url=document.source_url,
                access_level=document.access_level,

                # ==========================================
                # Chunk 内部信息。
                # ==========================================

                chunk_index=chunk_index,
                content_hash=content_hash,

                # ==========================================
                # Generic Section Metadata。
                #
                # 注意：
                #
                # 这里仍然保存真实 Section 信息。
                #
                # 不把：
                #
                #     occ02
                #
                # 写进 section_title / section_path。
                # ==========================================

                section_title=section.title,
                section_path=section.path,
            )

            chunks.append(
                chunk
            )

            chunk_index += 1

    return chunks