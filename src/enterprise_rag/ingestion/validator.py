"""KnowledgeChunk 数据质量校验。"""

from collections import Counter, defaultdict

from enterprise_rag.ingestion.models import KnowledgeChunk

import hashlib

ALLOWED_ACCESS_LEVELS = {
    "public",
    "developer",
    "admin",
}


def validate_chunks(
    chunks: list[KnowledgeChunk],
) -> list[str]:
    """
    校验 Chunk 数据质量。

    返回：
        所有发现的问题。

    如果返回空列表，表示当前数据通过校验。
    """

    errors: list[str] = []

    if not chunks:
        errors.append("Chunk 列表为空")
        return errors

    # --------------------------------------------------
    # 1. chunk_id 必须唯一
    # --------------------------------------------------

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    counts = Counter(chunk_ids)

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
    # 2. 每个 Chunk 的核心字段必须存在
    # --------------------------------------------------

    for chunk in chunks:
        if not chunk.chunk_id.strip():
            errors.append(
                "存在空 chunk_id"
            )

        if not chunk.document_id.strip():
            errors.append(
                f"{chunk.chunk_id}: document_id 为空"
            )

        if not chunk.title.strip():
            errors.append(
                f"{chunk.chunk_id}: title 为空"
            )

        if not chunk.article_number.strip():
            errors.append(
                f"{chunk.chunk_id}: article_number 为空"
            )

        if not chunk.content.strip():
            errors.append(
                f"{chunk.chunk_id}: content 为空"
            )

        if not chunk.retrieval_text.strip():
            errors.append(
                f"{chunk.chunk_id}: retrieval_text 为空"
            )

        # --------------------------------------------------
        # 3. ACL 值必须合法
        # --------------------------------------------------

        if chunk.access_level not in ALLOWED_ACCESS_LEVELS:
            errors.append(
                f"{chunk.chunk_id}: "
                f"非法 access_level={chunk.access_level}"
            )

        # --------------------------------------------------
        # 4. retrieval_text 至少要包含关键上下文
        # --------------------------------------------------

        if chunk.title not in chunk.retrieval_text:
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 缺少 title"
            )

        if chunk.article_number not in chunk.retrieval_text:
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 缺少 article_number"
            )

        if chunk.content not in chunk.retrieval_text:
            errors.append(
                f"{chunk.chunk_id}: "
                "retrieval_text 缺少 content"
            )

        # --------------------------------------------------
        # 5. content_hash 校验
        # --------------------------------------------------

        expected_hash = hashlib.sha256(
            chunk.content.encode("utf-8")
        ).hexdigest()

        if chunk.content_hash != expected_hash:
            errors.append(
                f"{chunk.chunk_id}: content_hash 不匹配"
            )

    chunks_by_document: dict[str, list[KnowledgeChunk]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    for document_id, document_chunks in chunks_by_document.items():
        indices = sorted(
            chunk.chunk_index
            for chunk in document_chunks
        )

        expected_indices = list(
            range(len(document_chunks))
        )

        if indices != expected_indices:
            errors.append(
                f"{document_id}: "
                "chunk_index 不连续，"
                f"实际={indices}，"
                f"期望={expected_indices}"
            )
            
    return errors