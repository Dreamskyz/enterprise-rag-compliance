"""测试 Grounded Generation JSON Parser。"""

import pytest

from enterprise_rag.generation.models import (
    EvidenceItem,
)
from enterprise_rag.generation.parser import (
    parse_grounded_answer,
)


def build_evidence() -> list[
    EvidenceItem
]:
    """构造两条合法 Evidence。"""

    return [
        EvidenceItem(
            evidence_id="E1",
            chunk_id="chunk-1",
            title="测试法规",
            article_number="第一条",
            content="测试正文一",
            source_url=(
                "https://example.com/1"
            ),
        ),
        EvidenceItem(
            evidence_id="E2",
            chunk_id="chunk-2",
            title="测试法规",
            article_number="第二条",
            content="测试正文二",
            source_url=(
                "https://example.com/2"
            ),
        ),
    ]


def test_parser_accepts_answerable_response() -> None:
    """合法可回答 JSON 应成功解析。"""

    raw = """
    {
      "answerable": true,
      "answer": "这是回答。",
      "reason": "E1 提供了明确依据。",
      "citations": ["E1"]
    }
    """

    result = parse_grounded_answer(
        raw,
        build_evidence(),
    )

    assert result.answerable is True

    assert result.answer == (
        "这是回答。"
    )

    assert len(
        result.citations
    ) == 1

    assert (
        result.citations[0].chunk_id
        == "chunk-1"
    )


def test_parser_accepts_refusal() -> None:
    """证据不足时应支持结构化拒答。"""

    raw = """
    {
      "answerable": false,
      "answer": null,
      "reason": "证据未提供具体时限。",
      "citations": []
    }
    """

    result = parse_grounded_answer(
        raw,
        build_evidence(),
    )

    assert result.answerable is False

    assert result.answer is None

    assert result.citations == ()


def test_parser_rejects_unknown_citation() -> None:
    """LLM 不得引用不存在的 Evidence ID。"""

    raw = """
    {
      "answerable": true,
      "answer": "测试回答",
      "reason": "存在依据",
      "citations": ["E99"]
    }
    """

    with pytest.raises(
        ValueError,
        match="未知 Evidence ID",
    ):
        parse_grounded_answer(
            raw,
            build_evidence(),
        )


def test_parser_rejects_answer_without_citation() -> None:
    """
    answerable=true 时
    至少必须引用一条 Evidence。
    """

    raw = """
    {
      "answerable": true,
      "answer": "测试回答",
      "reason": "存在依据",
      "citations": []
    }
    """

    with pytest.raises(
        ValueError,
        match="citations",
    ):
        parse_grounded_answer(
            raw,
            build_evidence(),
        )


def test_parser_rejects_refusal_with_answer() -> None:
    """
    answerable=false 时
    不允许偷偷返回 Answer。
    """

    raw = """
    {
      "answerable": false,
      "answer": "其实还是回答了",
      "reason": "证据不足",
      "citations": []
    }
    """

    with pytest.raises(
        ValueError,
        match="answer 必须为 null",
    ):
        parse_grounded_answer(
            raw,
            build_evidence(),
        )


def test_parser_strips_json_code_fence() -> None:
    """轻量兼容偶发的 Markdown JSON Fence。"""

    raw = '''```json
{
  "answerable": false,
  "answer": null,
  "reason": "证据不足",
  "citations": []
}
```'''

    result = parse_grounded_answer(
        raw,
        build_evidence(),
    )

    assert result.answerable is False