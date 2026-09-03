"""Chunk 数据质量校验测试。"""

from dataclasses import replace

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
    """
    构造一个合法的法规 Chunk。

    这个辅助函数主要服务于原有 Validator 测试，
    保持 regulation 作为基础测试数据。
    """

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

        # 法规 Chunk 不使用通用 Section 元数据。
        section_title=None,
        section_path=None,
    )


def make_security_guideline_chunk(
    chunk_id: str = "owasp_chunk_1",
    chunk_index: int = 0,
) -> KnowledgeChunk:
    """
    构造一个合法的 security_guideline Chunk。

    与 regulation 不同：

    - chapter_number = None
    - chapter_title = None
    - article_number = None

    而是依靠：

    - section_title
    - section_path

    表达文档结构。
    """

    content = (
        "Direct prompt injections occur when "
        "a user's prompt directly alters "
        "the behavior of the model."
    )

    section_title = (
        "Direct Prompt Injections"
    )

    section_path = (
        "LLM01:2025 Prompt Injection"
        " > Types of Prompt Injection Vulnerabilities"
        " > Direct Prompt Injections"
    )

    return KnowledgeChunk(
        chunk_id=chunk_id,

        document_id=(
            "owasp_llm01_prompt_injection_2025"
        ),
        title="LLM01:2025 Prompt Injection",

        document_type="security_guideline",
        language="en",
        version="2025",

        # Generic Section 文档没有法规的
        # chapter / article 结构。
        chapter_number=None,
        chapter_title=None,
        article_number=None,

        content=content,
        retrieval_text=(
            "LLM01:2025 Prompt Injection\n"
            f"{section_path}\n"
            f"{content}"
        ),

        source_url=(
            "https://genai.owasp.org/"
            "llmrisk/llm01-prompt-injection/"
        ),
        access_level="public",

        chunk_index=chunk_index,
        content_hash=build_content_hash(
            content
        ),

        section_title=section_title,
        section_path=section_path,
    )


def test_valid_chunks_have_no_errors() -> None:
    """
    合法法规 Chunk 应通过 Validator。
    """

    chunks = [
        make_chunk(
            chunk_id="chunk_1",
            chunk_index=0,
        )
    ]

    errors = validate_chunks(
        chunks
    )

    assert errors == []


def test_duplicate_chunk_id_is_rejected() -> None:
    """
    chunk_id 在整个知识库中必须唯一。
    """

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

    errors = validate_chunks(
        chunks
    )

    assert any(
        "chunk_id 重复" in error
        for error in errors
    )


def test_invalid_access_level_is_rejected() -> None:
    """
    ACL access_level 必须属于允许集合。
    """

    valid = make_chunk(
        chunk_id="chunk_1",
        chunk_index=0,
    )

    # KnowledgeChunk 是 frozen dataclass，
    # 因此不能直接修改字段。
    #
    # dataclasses.replace() 会基于原对象
    # 创建一个字段发生变化的新对象。
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


def test_security_guideline_chunk_is_valid() -> None:
    """
    合法 security_guideline Chunk 应通过校验。

    这是本次异构文档支持最关键的测试之一：

    article_number=None 本身不是错误，
    因为该文档使用 section 元数据表达结构。
    """

    chunk = (
        make_security_guideline_chunk()
    )

    errors = validate_chunks(
        [chunk]
    )

    assert errors == []


def test_regulation_without_article_number_is_rejected() -> None:
    """
    regulation 必须具有 article_number。

    虽然 KnowledgeChunk Schema 允许
    article_number 为 None，

    但对于 regulation 来说，
    article_number 仍然属于文档类型约束。
    """

    valid = make_chunk(
        chunk_id="chunk_1",
        chunk_index=0,
    )

    invalid = replace(
        valid,
        article_number=None,
    )

    errors = validate_chunks(
        [invalid]
    )

    assert any(
        "regulation 缺少 article_number"
        in error
        for error in errors
    )


def test_security_guideline_without_section_path_is_rejected() -> None:
    """
    security_guideline 必须具有 section_path。

    section_path 用于提供完整父级语义上下文，
    是 Generic Section Chunk 的核心元数据。
    """

    valid = (
        make_security_guideline_chunk()
    )

    invalid = replace(
        valid,
        section_path=None,
    )

    errors = validate_chunks(
        [invalid]
    )

    assert any(
        "security_guideline 缺少 section_path"
        in error
        for error in errors
    )


def test_security_guideline_without_section_title_is_rejected() -> None:
    """
    security_guideline 也必须具有 section_title。
    """

    valid = (
        make_security_guideline_chunk()
    )

    invalid = replace(
        valid,
        section_title=None,
    )

    errors = validate_chunks(
        [invalid]
    )

    assert any(
        "security_guideline 缺少 section_title"
        in error
        for error in errors
    )


def test_chunk_index_must_be_continuous_per_document() -> None:
    """
    同一篇文档中的 chunk_index
    必须从 0 开始连续递增。

    这里故意构造：

        0, 2

    缺少：

        1

    Validator 应拒绝。
    """

    first = (
        make_security_guideline_chunk(
            chunk_id="owasp_chunk_1",
            chunk_index=0,
        )
    )

    second = (
        make_security_guideline_chunk(
            chunk_id="owasp_chunk_2",
            chunk_index=2,
        )
    )

    errors = validate_chunks(
        [
            first,
            second,
        ]
    )

    assert any(
        "chunk_index 不连续"
        in error
        for error in errors
    )