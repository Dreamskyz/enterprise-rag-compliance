"""KnowledgeChunk 数据质量校验。"""

import hashlib
from collections import Counter, defaultdict

from enterprise_rag.ingestion.models import KnowledgeChunk


ALLOWED_ACCESS_LEVELS = {
    "public",
    "developer",
    "admin",
}


# 使用 Generic Section Parser / Chunker
# 的文档类型。
#
# 这些类型虽然业务语义不同：
#
# security_guideline
#     -> 安全规范
#
# technical_documentation
#     -> 技术文档
#
# 但它们的结构元数据 contract 相同：
#
# section_title
# section_path
#
# 因此 Validator 不应该为每一种文档类型
# 复制完全相同的 Section 校验代码。
GENERIC_SECTION_DOCUMENT_TYPES = {
    "security_guideline",
    "technical_documentation",
}


def validate_chunks(
    chunks: list[KnowledgeChunk],
) -> list[str]:
    """
    校验 KnowledgeChunk 数据质量。

    返回：
        所有发现的问题。

    如果返回空列表：
        表示当前 Chunk 数据通过校验。

    当前 Validator 支持异构文档：

    1. regulation

       使用：
       chapter / article 元数据。

    2. security_guideline

       使用：
       section 元数据。

    3. technical_documentation

       使用：
       section 元数据。

    核心原则：

        统一 KnowledgeChunk Schema

    不代表所有文档类型的
    所有字段都必须非空。

    而是应该根据 document_type
    校验它真正需要的结构字段。
    """

    errors: list[str] = []

    if not chunks:
        errors.append(
            "Chunk 列表为空"
        )
        return errors

    # --------------------------------------------------
    # 1. chunk_id 必须全局唯一
    # --------------------------------------------------

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    counts = Counter(
        chunk_ids
    )

    duplicate_ids = [
        chunk_id
        for chunk_id, count in counts.items()
        if count > 1
    ]

    for chunk_id in duplicate_ids:
        errors.append(
            f"chunk_id 重复：{chunk_id}"
        )

    # --------------------------------------------------
    # 2. 每个 Chunk 的通用核心字段必须存在
    # --------------------------------------------------

    for chunk in chunks:
        if not chunk.chunk_id.strip():
            errors.append(
                "存在空 chunk_id"
            )

        if not chunk.document_id.strip():
            errors.append(
                f"{chunk.chunk_id}: "
                "document_id 为空"
            )

        if not chunk.title.strip():
            errors.append(
                f"{chunk.chunk_id}: "
                "title 为空"
            )

        if not chunk.document_type.strip():
            errors.append(
                f"{chunk.chunk_id}: "
                "document_type 为空"
            )

        if not chunk.content.strip():
            errors.append(
                f"{chunk.chunk_id}: "
                "content 为空"
            )

        if not chunk.retrieval_text.strip():
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 为空"
            )

        # --------------------------------------------------
        # 3. 按 document_type 校验结构化元数据
        # --------------------------------------------------

        if (
            chunk.document_type
            == "regulation"
        ):
            # 法规 Chunk 应该来自：
            #
            # Chapter -> Article
            #
            # 所以 article_number 是法规的
            # 核心结构字段。
            if (
                chunk.article_number is None
                or not chunk.article_number.strip()
            ):
                errors.append(
                    f"{chunk.chunk_id}: "
                    "regulation 缺少 article_number"
                )

        elif (
            chunk.document_type
            in GENERIC_SECTION_DOCUMENT_TYPES
        ):
            # Generic Section 文档包括：
            #
            # security_guideline
            # technical_documentation
            #
            # 它们不要求：
            #
            # chapter_number
            # article_number
            #
            # 而要求：
            #
            # section_title
            # section_path

            if (
                chunk.section_title is None
                or not chunk.section_title.strip()
            ):
                errors.append(
                    f"{chunk.chunk_id}: "
                    f"{chunk.document_type} "
                    "缺少 section_title"
                )

            if (
                chunk.section_path is None
                or not chunk.section_path.strip()
            ):
                errors.append(
                    f"{chunk.chunk_id}: "
                    f"{chunk.document_type} "
                    "缺少 section_path"
                )

        else:
            # Validator 不应默默接受一个自己完全
            # 不认识的 document_type。
            #
            # 否则以后某个新类型忘了增加校验规则，
            # 错误数据可能悄悄进入向量库。
            errors.append(
                f"{chunk.chunk_id}: "
                "Validator 不支持的 "
                f"document_type={chunk.document_type}"
            )

        # --------------------------------------------------
        # 4. ACL 值必须合法
        # --------------------------------------------------

        if (
            chunk.access_level
            not in ALLOWED_ACCESS_LEVELS
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "非法 access_level="
                f"{chunk.access_level}"
            )

        # --------------------------------------------------
        # 5. retrieval_text 至少包含关键上下文
        # --------------------------------------------------

        # 所有类型的 retrieval_text
        # 都必须携带文档标题。
        if (
            chunk.title
            not in chunk.retrieval_text
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 缺少 title"
            )

        # --------------------------------------------------
        # 5.1 Regulation retrieval context
        # --------------------------------------------------

        # 法规：
        #
        # retrieval_text 应包含 article_number。
        if (
            chunk.document_type
            == "regulation"
            and chunk.article_number is not None
            and chunk.article_number
            not in chunk.retrieval_text
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text "
                "缺少 article_number"
            )

        # --------------------------------------------------
        # 5.2 Generic Section retrieval context
        # --------------------------------------------------

        # Generic Section 文档：
        #
        # retrieval_text 应包含完整 section_path。
        #
        # section_path 不只是当前 Heading，
        # 它还携带父级语义上下文，例如：
        #
        # FastAPI Lifespan Events
        # > Lifespan
        # > Lifespan function
        #
        # 对检索很重要。
        if (
            chunk.document_type
            in GENERIC_SECTION_DOCUMENT_TYPES
            and chunk.section_path is not None
            and chunk.section_path
            not in chunk.retrieval_text
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text "
                "缺少 section_path"
            )

        # --------------------------------------------------
        # 5.3 Chunk content
        # --------------------------------------------------

        # 不管哪种文档类型，
        # retrieval_text 都必须真正包含
        # Chunk 正文。
        if (
            chunk.content
            not in chunk.retrieval_text
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 缺少 content"
            )

        # --------------------------------------------------
        # 6. content_hash 校验
        # --------------------------------------------------

        expected_hash = hashlib.sha256(
            chunk.content.encode(
                "utf-8"
            )
        ).hexdigest()

        if (
            chunk.content_hash
            != expected_hash
        ):
            errors.append(
                f"{chunk.chunk_id}: "
                "content_hash 不匹配"
            )

    # --------------------------------------------------
    # 7. 每篇文档内部 chunk_index 必须连续
    # --------------------------------------------------

    chunks_by_document: dict[
        str,
        list[KnowledgeChunk],
    ] = defaultdict(
        list
    )

    for chunk in chunks:
        chunks_by_document[
            chunk.document_id
        ].append(
            chunk
        )

    for (
        document_id,
        document_chunks,
    ) in chunks_by_document.items():
        indices = sorted(
            chunk.chunk_index
            for chunk in document_chunks
        )

        expected_indices = list(
            range(
                len(
                    document_chunks
                )
            )
        )

        if (
            indices
            != expected_indices
        ):
            errors.append(
                f"{document_id}: "
                "chunk_index 不连续，"
                f"实际={indices}，"
                f"期望={expected_indices}"
            )

    return errors