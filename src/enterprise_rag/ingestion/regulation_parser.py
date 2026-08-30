"""中国法规章节和条款结构解析。"""

import re   #Python 内置的正则表达式模块

from enterprise_rag.ingestion.models import (
    RegulationArticle,
    RegulationChapter,
)


# 匹配：
# 第一章
# 第二章
# 第五章
CHAPTER_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百]+章)\s*(.*)$"
)


# 匹配：
# 第一条
# 第二十四条
ARTICLE_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百]+条)\s*(.*)$"
)


def parse_regulation(
    text: str,
) -> list[RegulationChapter]:
    """
    将法规纯文本解析为：
        Chapter -> Article

    当前假设：
    1. 章节标题单独占一行；
    2. 条款编号单独占一行；
    3. 同一条后续行都属于该条，
       直到遇到下一条或下一章。
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chapters: list[RegulationChapter] = []

    current_chapter_number: str | None = None
    current_chapter_title: str = ""

    current_articles: list[RegulationArticle] = []

    current_article_number: str | None = None
    current_article_lines: list[str] = []

    def flush_article() -> None:
        """
        将当前正在收集的条款写入 current_articles。
        """

        nonlocal current_article_number
        nonlocal current_article_lines

        if current_article_number is None:
            return

        content = "\n".join(current_article_lines).strip()

        current_articles.append(
            RegulationArticle(
                article_number=current_article_number,
                content=content,
            )
        )

        current_article_number = None
        current_article_lines = []

    def flush_chapter() -> None:
        """
        将当前章节写入 chapters。
        """

        nonlocal current_chapter_number
        nonlocal current_chapter_title
        nonlocal current_articles

        if current_chapter_number is None:
            return

        chapters.append(
            RegulationChapter(
                chapter_number=current_chapter_number,
                title=current_chapter_title,
                articles=current_articles,
            )
        )

        current_chapter_number = None
        current_chapter_title = ""
        current_articles = []

    for line in lines:
        chapter_match = CHAPTER_PATTERN.match(line)

        if chapter_match:
            # 新章开始前，先结束上一条。
            flush_article()

            # 再结束上一章。
            flush_chapter()

            current_chapter_number = chapter_match.group(1)
            current_chapter_title = chapter_match.group(2)

            continue

        article_match = ARTICLE_PATTERN.match(line)

        if article_match:
            # 新条开始前，先把上一条保存。
            flush_article()

            current_article_number = article_match.group(1)

            # 有些文档可能是：
            # 第一条 为了……
            # 因此需要保留同一行后面的正文。
            first_line_content = article_match.group(2)

            current_article_lines = []

            if first_line_content:
                current_article_lines.append(
                    first_line_content
                )

            continue

        # 当前已经进入某一条后，
        # 普通正文继续累加到当前条款。
        if current_article_number is not None:
            current_article_lines.append(line)

    # 文件结束时别忘了把最后一条和最后一章写进去。
    flush_article()
    flush_chapter()

    return chapters