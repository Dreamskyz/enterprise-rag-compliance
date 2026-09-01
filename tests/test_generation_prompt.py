"""测试 Generation Prompt Builder。"""

from enterprise_rag.generation.prompt_builder import (
    build_evidence_items,
    build_generation_messages,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
    RetrievalCandidate,
)


def make_result(
    chunk_id: str,
    content: str,
) -> RerankedSearchResult:
    """构造测试 Retrieval Result。"""

    candidate = RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc",
        title="测试法规",
        document_type="regulation",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="测试",
        article_number="第一条",
        content=content,
        retrieval_text=content,
        source_url=(
            "https://example.com"
        ),
        access_level="public",
        chunk_index=0,
        content_hash="hash",
    )

    return RerankedSearchResult(
        candidate=candidate,
        rerank_score=5.0,
        original_rank=1,
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
    )


def test_build_evidence_items_assigns_ids() -> None:
    """Evidence 应按顺序编号为 E1/E2。"""

    items = build_evidence_items(
        [
            make_result(
                "chunk-a",
                "正文 A",
            ),
            make_result(
                "chunk-b",
                "正文 B",
            ),
        ]
    )

    assert [
        item.evidence_id
        for item in items
    ] == [
        "E1",
        "E2",
    ]


def test_build_evidence_items_respects_limit() -> None:
    """Evidence 数量必须受 max_evidence 限制。"""

    items = build_evidence_items(
        [
            make_result(
                "a",
                "A",
            ),
            make_result(
                "b",
                "B",
            ),
        ],
        max_evidence=1,
    )

    assert len(items) == 1


def test_prompt_contains_question_and_evidence() -> None:
    """Prompt 必须包含 Query 和真实 Evidence。"""

    items = build_evidence_items(
        [
            make_result(
                "chunk-a",
                "必须及时处理违法内容。",
            )
        ]
    )

    messages = (
        build_generation_messages(
            query="需要几小时处理？",
            evidence_items=items,
        )
    )

    assert len(messages) == 2

    user_content = (
        messages[1]["content"]
    )

    assert "需要几小时处理？" in (
        user_content
    )

    assert "[E1]" in user_content

    assert (
        "必须及时处理违法内容"
        in user_content
    )