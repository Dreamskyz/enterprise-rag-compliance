"""Generic Section Chunker 测试。"""

from enterprise_rag.ingestion.generic_section_chunker import (
    build_generic_section_chunks,
    build_section_slug,
    split_section_content,
)
from enterprise_rag.ingestion.models import (
    GenericSection,
    NormalizedDocument,
)


def _build_test_document() -> NormalizedDocument:
    """
    构造统一测试技术文档。

    ACL 暂时使用 developer，
    用于验证文档级 Metadata
    能否正确进入最终 KnowledgeChunk。
    """

    return NormalizedDocument(
        document_id="fastapi_dependencies",
        title="FastAPI Dependencies",

        document_type=(
            "technical_documentation"
        ),

        language="en",
        version="current",

        text="",

        source_url=(
            "https://fastapi.tiangolo.com/"
        ),

        access_level="developer",
    )


def test_build_single_generic_section_chunk() -> None:
    """
    一个普通非空 Section
    应生成一个标准 KnowledgeChunk。
    """

    document = _build_test_document()

    sections = [
        GenericSection(
            title=(
                "Classes as Dependencies"
            ),
            level=3,
            path=(
                "FastAPI > Dependencies > "
                "Classes as Dependencies"
            ),
            content=(
                "A Python class can be used "
                "as a dependency."
            ),
        )
    ]

    chunks = build_generic_section_chunks(
        document=document,
        sections=sections,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert (
        chunk.document_id
        == "fastapi_dependencies"
    )

    assert (
        chunk.document_type
        == "technical_documentation"
    )

    assert (
        chunk.section_title
        == "Classes as Dependencies"
    )

    assert (
        chunk.section_path
        == (
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        )
    )

    assert chunk.chapter_number is None
    assert chunk.chapter_title is None
    assert chunk.article_number is None

    assert chunk.chunk_index == 0

    assert (
        chunk.access_level
        == "developer"
    )


def test_generic_retrieval_text_contains_document_and_section_context() -> None:
    """
    retrieval_text 应同时包含：

        Document Title
        Section Path
        Content
    """

    document = _build_test_document()

    section = GenericSection(
        title="Classes as Dependencies",
        level=3,
        path=(
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        ),
        content=(
            "A Python class can be used "
            "as a dependency."
        ),
    )

    chunks = build_generic_section_chunks(
        document=document,
        sections=[section],
    )

    retrieval_text = (
        chunks[0].retrieval_text
    )

    assert (
        "FastAPI Dependencies"
        in retrieval_text
    )

    assert (
        (
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        )
        in retrieval_text
    )

    assert (
        (
            "A Python class can be used "
            "as a dependency."
        )
        in retrieval_text
    )


def test_empty_section_does_not_create_chunk() -> None:
    """
    Parser 可以保留空结构节点，

    但 Chunker 不应把空 Section
    写入知识库。
    """

    document = _build_test_document()

    sections = [
        GenericSection(
            title="Dependencies",
            level=2,
            path=(
                "FastAPI > Dependencies"
            ),
            content="",
        ),
        GenericSection(
            title=(
                "Classes as Dependencies"
            ),
            level=3,
            path=(
                "FastAPI > Dependencies > "
                "Classes as Dependencies"
            ),
            content=(
                "Class dependency content."
            ),
        ),
    ]

    chunks = build_generic_section_chunks(
        document=document,
        sections=sections,
    )

    assert len(chunks) == 1

    assert (
        chunks[0].section_title
        == "Classes as Dependencies"
    )

    # 空 Section 被跳过以后，
    # 第一个真实 Chunk 的全局 index
    # 仍然应该从 0 开始。
    assert chunks[0].chunk_index == 0


def test_long_section_is_split_by_paragraph_boundary() -> None:
    """
    长 Section 应优先按照段落边界拆分，
    而不是直接从段落中间截断。
    """

    document = _build_test_document()

    paragraph_a = "A" * 40
    paragraph_b = "B" * 40
    paragraph_c = "C" * 40

    section = GenericSection(
        title="Long Section",
        level=2,
        path=(
            "FastAPI > Long Section"
        ),
        content=(
            f"{paragraph_a}\n\n"
            f"{paragraph_b}\n\n"
            f"{paragraph_c}"
        ),
    )

    # A + 空行 + B：
    #
    # 40 + 2 + 40 = 82
    #
    # 小于 90。
    #
    # 再加入 C：
    #
    # 82 + 2 + 40 = 124
    #
    # 超过 90。
    #
    # 因此应拆成：
    #
    # Part 1 = A + B
    # Part 2 = C
    chunks = build_generic_section_chunks(
        document=document,
        sections=[section],
        max_chars=90,
    )

    assert len(chunks) == 2

    assert (
        chunks[0].content
        == (
            f"{paragraph_a}\n\n"
            f"{paragraph_b}"
        )
    )

    assert (
        chunks[1].content
        == paragraph_c
    )


def test_oversized_single_paragraph_uses_hard_split() -> None:
    """
    如果一个单独段落自己就超过字符预算，
    最后才允许使用字符级 Hard Split。
    """

    document = _build_test_document()

    section = GenericSection(
        title="Large Paragraph",
        level=2,
        path=(
            "FastAPI > Large Paragraph"
        ),
        content="A" * 25,
    )

    chunks = build_generic_section_chunks(
        document=document,
        sections=[section],
        max_chars=10,
    )

    assert len(chunks) == 3

    assert len(chunks[0].content) == 10
    assert len(chunks[1].content) == 10
    assert len(chunks[2].content) == 5


def test_chunk_index_is_document_global_order() -> None:
    """
    chunk_index 应在整篇文档范围连续递增，
    而不是每到一个新 Section 就重新从 0 开始。
    """

    document = _build_test_document()

    sections = [
        GenericSection(
            title="Section A",
            level=2,
            path="FastAPI > Section A",
            content="A" * 15,
        ),
        GenericSection(
            title="Section B",
            level=2,
            path="FastAPI > Section B",
            content="B",
        ),
    ]

    # Section A：
    # 15 字符，用 max_chars=10
    # → 两个 Chunk。
    #
    # Section B：
    # → 一个 Chunk。
    #
    # 全局 chunk_index 应该：
    #
    # 0, 1, 2
    chunks = build_generic_section_chunks(
        document=document,
        sections=sections,
        max_chars=10,
    )

    assert len(chunks) == 3

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [
        0,
        1,
        2,
    ]


def test_chunk_ids_are_stable_for_same_input() -> None:
    """
    相同输入重复构建时：

        chunk_id
        content_hash

    都必须保持稳定。

    这是未来 Retrieval Gold
    可以长期存在的基础。
    """

    document = _build_test_document()

    sections = [
        GenericSection(
            title=(
                "Classes as Dependencies"
            ),
            level=3,
            path=(
                "FastAPI > Dependencies > "
                "Classes as Dependencies"
            ),
            content=(
                "A Python class can be used "
                "as a dependency."
            ),
        )
    ]

    chunks_a = build_generic_section_chunks(
        document=document,
        sections=sections,
    )

    chunks_b = build_generic_section_chunks(
        document=document,
        sections=sections,
    )

    assert (
        chunks_a[0].chunk_id
        == chunks_b[0].chunk_id
    )

    assert (
        chunks_a[0].content_hash
        == chunks_b[0].content_hash
    )


def test_same_section_title_in_different_paths_has_different_chunk_ids() -> None:
    """
    不能只根据 section_title 生成 ID。

    因为真实技术文档中很可能出现：

        Dependencies > Examples

        Security > Examples

    两个同名 Examples Section。
    """

    document = _build_test_document()

    sections = [
        GenericSection(
            title="Examples",
            level=3,
            path=(
                "FastAPI > Dependencies > "
                "Examples"
            ),
            content="Dependency example.",
        ),
        GenericSection(
            title="Examples",
            level=3,
            path=(
                "FastAPI > Security > "
                "Examples"
            ),
            content="Security example.",
        ),
    ]

    chunks = build_generic_section_chunks(
        document=document,
        sections=sections,
    )

    assert len(chunks) == 2

    assert (
        chunks[0].chunk_id
        != chunks[1].chunk_id
    )


def test_section_slug_is_deterministic() -> None:
    """
    Section Path → slug
    应保持稳定和可读。
    """

    path = (
        "FastAPI > Dependencies > "
        "Classes as Dependencies"
    )

    slug_a = build_section_slug(
        path
    )

    slug_b = build_section_slug(
        path
    )

    assert slug_a == slug_b

    assert (
        slug_a
        == (
            "fastapi_dependencies_"
            "classes_as_dependencies"
        )
    )


def test_invalid_max_chars_is_rejected() -> None:
    """
    max_chars <= 0 属于非法配置，
    应明确失败，而不是产生不可预测行为。
    """

    document = _build_test_document()

    section = GenericSection(
        title="Dependencies",
        level=2,
        path=(
            "FastAPI > Dependencies"
        ),
        content="Some content.",
    )

    try:
        build_generic_section_chunks(
            document=document,
            sections=[section],
            max_chars=0,
        )

    except ValueError as exc:
        assert (
            "max_chars"
            in str(exc)
        )

    else:
        raise AssertionError(
            "max_chars=0 应触发 ValueError"
        )