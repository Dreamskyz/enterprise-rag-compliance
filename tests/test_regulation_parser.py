"""法规解析器测试。"""

from pathlib import Path

from enterprise_rag.ingestion.regulation_parser import (
    parse_regulation,
)


def test_parse_regulation_structure() -> None:
    """
    应正确识别章、条以及条内列表。
    """

    path = Path(
        "tests/fixtures/sample_regulation.txt"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    chapters = parse_regulation(text)

    assert len(chapters) == 2

    article_count = sum(
        len(chapter.articles)
        for chapter in chapters
    )

    assert article_count == 4

    assert chapters[0].chapter_number == "第一章"
    assert chapters[0].title == "总则"

    assert (
        chapters[1].articles[0].article_number
        == "第三条"
    )

    assert "（一）要求一" in (
        chapters[1].articles[0].content
    )

    assert "（二）要求二" in (
        chapters[1].articles[0].content
    )


def test_last_article_is_not_lost() -> None:
    """
    文件结束时，最后一条仍必须被 flush 保存。
    """

    text = """
第一章 总则
第一条
第一条内容。
第二条
这是最后一条。
""".strip()

    chapters = parse_regulation(text)

    articles = chapters[0].articles

    assert len(articles) == 2
    assert articles[-1].article_number == "第二条"
    assert articles[-1].content == "这是最后一条。"


def test_article_like_content_is_not_treated_as_new_article() -> None:
    """
    正文即使以“第X条”开头，
    只要条号后没有分隔空白，
    就不应被误判成新的条款标题。
    """

    text = """
第一章 总则
第一条
第一条内容。
第二条
第二条内容。
""".strip()

    chapters = parse_regulation(text)

    articles = chapters[0].articles

    assert len(articles) == 2

    assert articles[0].article_number == "第一条"
    assert articles[0].content == "第一条内容。"

    assert articles[1].article_number == "第二条"
    assert articles[1].content == "第二条内容。"