"""法规 Chunker 测试。"""

from enterprise_rag.ingestion.chunker import (
    build_regulation_chunks,
)
from enterprise_rag.ingestion.models import (
    NormalizedDocument,
    RegulationArticle,
    RegulationChapter,
)


def test_build_regulation_chunks() -> None:
    document = NormalizedDocument(
        document_id="test_doc",
        title="测试法规",

        document_type="regulation",
        language="zh-CN",
        version="1",

        text="",

        source_url="https://example.com",
        access_level="public",
    )

    chapters = [
        RegulationChapter(
            chapter_number="第一章",
            title="总则",
            articles=[
                RegulationArticle(
                    article_number="第一条",
                    content="这是第一条。",
                ),
                RegulationArticle(
                    article_number="第二条",
                    content="这是第二条。",
                ),
            ],
        )
    ]

    chunks = build_regulation_chunks(
        document=document,
        chapters=chapters,
    )

    assert len(chunks) == 2

    first = chunks[0]

    assert first.chunk_id == "test_doc__第一条"
    assert first.chunk_index == 0
    assert first.access_level == "public"

    assert "测试法规" in first.retrieval_text
    assert "第一章 总则" in first.retrieval_text
    assert "第一条" in first.retrieval_text
    assert "这是第一条。" in first.retrieval_text


from enterprise_rag.ingestion.chunker import (
    build_content_hash,
)


def test_content_hash_is_stable() -> None:
    content = "同样的正文"

    hash_a = build_content_hash(content)
    hash_b = build_content_hash(content)

    assert hash_a == hash_b


def test_content_hash_changes_with_content() -> None:
    hash_a = build_content_hash("正文 A")
    hash_b = build_content_hash("正文 B")

    assert hash_a != hash_b