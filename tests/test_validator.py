"""Chunk 数据质量校验测试。"""

from enterprise_rag.ingestion.chunker import (
    build_content_hash,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.ingestion.validator import (
    validate_chunks,
)


def make_chunk(
    chunk_id: str,
    chunk_index: int,
) -> KnowledgeChunk:
    content = "测试正文"

    return KnowledgeChunk(
        chunk_id=chunk_id,

        document_id="test_doc",
        title="测试文档",

        document_type="regulation",
        language="zh-CN",
        version="1",

        chapter_number="第一章",
        chapter_title="总则",
        article_number="第一条",

        content=content,
        retrieval_text=(
            "测试文档\n"
            "第一章 总则\n"
            "第一条\n"
            "测试正文"
        ),

        source_url="https://example.com",
        access_level="public",

        chunk_index=chunk_index,
        content_hash=build_content_hash(
            content
        ),
    )


def test_valid_chunks_have_no_errors() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk_1",
            chunk_index=0,
        )
    ]

    errors = validate_chunks(chunks)

    assert errors == []


def test_duplicate_chunk_id_is_rejected() -> None:
    chunks = [
        make_chunk(
            chunk_id="duplicate",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="duplicate",
            chunk_index=1,
        ),
    ]

    errors = validate_chunks(chunks)

    assert any(
        "chunk_id 重复" in error
        for error in errors
    )


from dataclasses import replace

def test_invalid_access_level_is_rejected() -> None:
    valid = make_chunk(
        chunk_id="chunk_1",
        chunk_index=0,
    )

    invalid = replace(
        valid,
        access_level="superuser",
    )

    errors = validate_chunks(
        [invalid]
    )

    assert any(
        "非法 access_level" in error
        for error in errors
    )