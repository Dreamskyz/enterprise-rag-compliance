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

    assert (
        articles[-1].article_number
        == "第二条"
    )

    assert (
        articles[-1].content
        == "这是最后一条。"
    )


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

    assert (
        articles[0].article_number
        == "第一条"
    )

    assert (
        articles[0].content
        == "第一条内容。"
    )

    assert (
        articles[1].article_number
        == "第二条"
    )

    assert (
        articles[1].content
        == "第二条内容。"
    )


def test_parse_regulation_without_explicit_chapter() -> None:
    """
    没有“第一章 / 第二章”这类显式章节的法规，
    也必须能够正常解析 Article。

    例如《人工智能生成合成内容标识办法》
    的正文结构就是：

        第一条
        ...
        第二条
        ...
        第三条
        ...

    而不是：

        第一章
        第一条
        ...

    对于这种法规，Parser 会创建一个
    仅供内部统一数据结构使用的 implicit chapter。

    需要注意：

        chapter_number == ""
        chapter_title == ""

    表示原文没有真实章节，

    而不是伪造一个：

        第一章
    """

    text = """
关于印发某办法的通知
某某部门
2025年3月7日

某办法

第一条
第一条正文。

第二条
第二条正文。

第三条
第三条正文。
""".strip()

    chapters = parse_regulation(text)

    # --------------------------------------------------
    # 无显式 Chapter 的法规，
    # 仍然应该产生一个内部 Chapter 容器。
    # --------------------------------------------------

    assert len(chapters) == 1

    chapter = chapters[0]

    # --------------------------------------------------
    # 空字符串表示：
    #
    # 原文没有真实 Chapter。
    #
    # 这里不能伪造成“第一章”，
    # 否则会污染 Citation / Metadata。
    # --------------------------------------------------

    assert chapter.chapter_number == ""
    assert chapter.title == ""

    # --------------------------------------------------
    # 三条法规都必须被完整保留下来。
    # --------------------------------------------------

    assert len(chapter.articles) == 3

    assert (
        chapter.articles[0].article_number
        == "第一条"
    )

    assert (
        chapter.articles[0].content
        == "第一条正文。"
    )

    assert (
        chapter.articles[1].article_number
        == "第二条"
    )

    assert (
        chapter.articles[1].content
        == "第二条正文。"
    )

    assert (
        chapter.articles[2].article_number
        == "第三条"
    )

    assert (
        chapter.articles[2].content
        == "第三条正文。"
    )