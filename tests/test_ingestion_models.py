"""测试 Ingestion 数据模型。"""

from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)


def test_regulation_chunk_is_backward_compatible() -> None:
    """
    旧法规 Chunk 应继续能够按原有方式构造。

    Day 6 新增的：

        section_title
        section_path

    默认都应该为 None。

    这个测试主要保护 Schema Evolution
    不破坏 Day 1 已经稳定的法规 Pipeline。
    """

    chunk = KnowledgeChunk(
        chunk_id=(
            "cn_genai_interim_2023__第七条"
        ),
        document_id=(
            "cn_genai_interim_2023"
        ),
        title=(
            "生成式人工智能服务管理暂行办法"
        ),
        document_type="regulation",
        language="zh",
        version="2023",

        chapter_number="第二章",
        chapter_title="技术发展与治理",
        article_number="第七条",

        content=(
            "生成式人工智能服务提供者"
            "开展预训练、优化训练等"
            "训练数据处理活动时，应当遵守有关规定。"
        ),

        retrieval_text=(
            "生成式人工智能服务管理暂行办法\n"
            "第二章 技术发展与治理\n"
            "第七条\n"
            "生成式人工智能服务提供者"
            "开展预训练、优化训练等"
            "训练数据处理活动时，应当遵守有关规定。"
        ),

        source_url=(
            "https://example.com/regulation"
        ),

        access_level="public",

        chunk_index=0,

        content_hash="test-hash",
    )

    assert (
        chunk.chapter_number
        == "第二章"
    )

    assert (
        chunk.chapter_title
        == "技术发展与治理"
    )

    assert (
        chunk.article_number
        == "第七条"
    )

    # Day 6 新增字段对旧法规自动保持为空。
    assert (
        chunk.section_title
        is None
    )

    assert (
        chunk.section_path
        is None
    )


def test_technical_document_chunk_supports_section_metadata() -> None:
    """
    技术文档没有法规的 Chapter / Article 结构。

    应允许：

        chapter_number=None
        chapter_title=None
        article_number=None

    并使用：

        section_title
        section_path

    描述技术文档层级。
    """

    chunk = KnowledgeChunk(
        chunk_id=(
            "fastapi_dependencies__"
            "classes_as_dependencies__0001"
        ),

        document_id="fastapi_dependencies",

        title="FastAPI Dependencies",

        document_type=(
            "technical_documentation"
        ),

        language="en",

        version="current",

        # 技术文档没有法规结构。
        chapter_number=None,
        chapter_title=None,
        article_number=None,

        content=(
            "A Python class can be used "
            "as a dependency."
        ),

        retrieval_text=(
            "FastAPI Dependencies\n"
            "Tutorial > Dependencies > "
            "Classes as Dependencies\n"
            "A Python class can be used "
            "as a dependency."
        ),

        source_url=(
            "https://fastapi.tiangolo.com/"
        ),

        # 当前仅用于模拟企业内部技术知识域。
        access_level="developer",

        chunk_index=0,

        content_hash="technical-test-hash",

        section_title=(
            "Classes as Dependencies"
        ),

        section_path=(
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        ),
    )

    # 技术文档没有法规结构。
    assert (
        chunk.chapter_number
        is None
    )

    assert (
        chunk.chapter_title
        is None
    )

    assert (
        chunk.article_number
        is None
    )

    # 技术文档使用新的 Section Metadata。
    assert (
        chunk.section_title
        == "Classes as Dependencies"
    )

    assert (
        chunk.section_path
        == (
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        )
    )


def test_security_guideline_can_use_same_chunk_contract() -> None:
    """
    OWASP 等安全规范也应该复用 KnowledgeChunk，

    而不是再定义：

        OwaspChunk

    这验证统一 Chunk Contract 的设计。
    """

    chunk = KnowledgeChunk(
        chunk_id=(
            "owasp_llm01_prompt_injection__"
            "prevention__0001"
        ),

        document_id=(
            "owasp_llm01_prompt_injection"
        ),

        title=(
            "LLM01: Prompt Injection"
        ),

        document_type=(
            "security_guideline"
        ),

        language="en",

        version="current",

        chapter_number=None,
        chapter_title=None,
        article_number=None,

        content=(
            "Prompt injection vulnerabilities "
            "occur when user prompts alter "
            "the model's intended behavior."
        ),

        retrieval_text=(
            "LLM01: Prompt Injection\n"
            "Prevention and Mitigation Strategies\n"
            "Prompt injection vulnerabilities "
            "occur when user prompts alter "
            "the model's intended behavior."
        ),

        source_url=(
            "https://owasp.org/"
        ),

        access_level="public",

        chunk_index=0,

        content_hash="owasp-test-hash",

        section_title=(
            "Prevention and Mitigation Strategies"
        ),

        section_path=(
            "LLM01: Prompt Injection > "
            "Prevention and Mitigation Strategies"
        ),
    )

    assert (
        chunk.document_type
        == "security_guideline"
    )

    assert (
        chunk.section_title
        == (
            "Prevention and Mitigation Strategies"
        )
    )

    assert (
        chunk.article_number
        is None
    )