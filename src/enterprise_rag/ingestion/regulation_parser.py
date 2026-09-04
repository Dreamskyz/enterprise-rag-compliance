"""中国法规章节和条款结构解析。"""

import re  # Python 内置的正则表达式模块

from enterprise_rag.ingestion.models import (
    RegulationArticle,
    RegulationChapter,
)


# ==========================================================
# Chapter Pattern
# ==========================================================
#
# 匹配：
#
# 第一章
# 第一章 总则
# 第二章 数据处理规则
#
# group(1):
#     第一章
#
# group(2):
#     总则
#
# 如果章节编号后没有标题，
# group(2) 会得到空字符串。
# ==========================================================

CHAPTER_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百]+章)\s*(.*)$"
)


# ==========================================================
# Article Pattern
# ==========================================================
#
# 支持两种常见格式：
#
# 第一条
#
# 第一条 为了……
#
# group(1):
#     第一条
#
# group(2):
#     同一行中的正文；
#     如果条款编号独占一行，则为 None。
# ==========================================================

ARTICLE_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百]+条)(?:\s+(.*))?$"
)


def parse_regulation(
    text: str,
) -> list[RegulationChapter]:
    """
    将法规纯文本解析为：

        Chapter
            ↓
        Article

    支持两种法规结构：

    1. 有显式章节：

        第一章 总则
        第一条
        ...
        第二条
        ...

        第二章 ...
        第三条
        ...

    2. 没有显式章节：

        第一条
        ...
        第二条
        ...
        第三条
        ...

       对于第二种情况，Parser 会建立一个：

           implicit chapter

       作为内部结构容器。

       注意：
       implicit chapter 的：

           chapter_number = ""
           chapter_title = ""

       它只是为了让下游仍然使用统一的：

           Chapter -> Article

       数据结构，并不表示原始法规中真的存在一个章节。

    当前假设：

    1. 章节标题单独占一行；
    2. 条款编号通常单独占一行；
    3. 同时兼容：

           第一条 为了……

       这种“条号 + 正文同一行”的形式；
    4. 同一条后续普通文本都属于该条，
       直到遇到下一条或下一章。
    """

    # ======================================================
    # 1. 文本预处理
    # ======================================================
    #
    # splitlines():
    #     将全文按行拆开。
    #
    # strip():
    #     清除每一行首尾空白。
    #
    # if line.strip():
    #     删除空行。
    # ======================================================

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # 最终返回结果。
    chapters: list[RegulationChapter] = []

    # ======================================================
    # 2. 当前 Chapter 状态
    # ======================================================

    # None：
    #     当前还没有进入任何 Chapter。
    #
    # ""：
    #     当前处于“无显式章节法规”的 implicit chapter。
    #
    # "第一章"：
    #     当前处于真实章节。
    current_chapter_number: str | None = None

    current_chapter_title: str = ""

    # 当前 Chapter 中已经完整解析完成的 Article。
    current_articles: list[RegulationArticle] = []

    # ======================================================
    # 3. 当前 Article 状态
    # ======================================================

    current_article_number: str | None = None

    current_article_lines: list[str] = []

    # ======================================================
    # 4. Flush Article
    # ======================================================

    def flush_article() -> None:
        """
        将当前正在收集的 Article
        写入 current_articles。
        """

        nonlocal current_article_number
        nonlocal current_article_lines

        # 当前没有正在解析的 Article，
        # 不需要做任何事情。
        if current_article_number is None:
            return

        # 将当前 Article 的多行正文重新拼接。
        content = "\n".join(
            current_article_lines
        ).strip()

        current_articles.append(
            RegulationArticle(
                article_number=current_article_number,
                content=content,
            )
        )

        # 当前 Article 已经结束，
        # 清空 Article 状态。
        current_article_number = None
        current_article_lines = []

    # ======================================================
    # 5. Flush Chapter
    # ======================================================

    def flush_chapter() -> None:
        """
        将当前 Chapter 写入最终 chapters。

        同时支持：

        1. 真实 Chapter；
        2. implicit chapter。
        """

        nonlocal current_chapter_number
        nonlocal current_chapter_title
        nonlocal current_articles

        # None 表示：
        #
        # 当前从未进入任何真实 / implicit Chapter。
        #
        # 例如全文中完全没有解析到 Article，
        # 此时不能产生空 Chapter。
        if current_chapter_number is None:
            return

        # 一个 Chapter 连 Article 都没有，
        # 没有保留价值。
        #
        # 这里也可以避免某些异常格式生成空结构。
        if not current_articles:
            current_chapter_number = None
            current_chapter_title = ""
            current_articles = []
            return

        chapters.append(
            RegulationChapter(
                chapter_number=current_chapter_number,
                title=current_chapter_title,
                articles=current_articles,
            )
        )

        # 当前 Chapter 已经结束，
        # 重置状态。
        current_chapter_number = None
        current_chapter_title = ""
        current_articles = []

    # ======================================================
    # 6. 主扫描循环
    # ======================================================

    for line in lines:

        # --------------------------------------------------
        # 6.1 尝试识别 Chapter
        # --------------------------------------------------

        chapter_match = CHAPTER_PATTERN.match(
            line
        )

        if chapter_match:

            # 开启新 Chapter 之前，
            # 先结束上一条。
            flush_article()

            # 再结束上一章。
            #
            # 如果前面是 implicit chapter，
            # 它也会在这里正常保存。
            flush_chapter()

            current_chapter_number = (
                chapter_match.group(1)
            )

            current_chapter_title = (
                chapter_match.group(2)
            )

            continue

        # --------------------------------------------------
        # 6.2 尝试识别 Article
        # --------------------------------------------------

        article_match = ARTICLE_PATTERN.match(
            line
        )

        if article_match:

            # 新 Article 开始前，
            # 先结束上一 Article。
            flush_article()

            # ------------------------------------------------
            # 关键修复：
            #
            # 如果已经识别到 Article，
            # 但是之前从未遇到 Chapter，
            # 说明当前法规属于：
            #
            #     无显式章节法规
            #
            # 例如：
            #
            #     人工智能生成合成内容标识办法
            #
            # 原文结构直接是：
            #
            #     第一条
            #     第二条
            #     ...
            #
            # 此时创建一个内部 implicit chapter。
            #
            # 使用空字符串非常重要：
            #
            #     ""
            #
            # 而不是伪造：
            #
            #     "第一章"
            #
            # 因为原文并不存在第一章。
            # ------------------------------------------------

            if current_chapter_number is None:
                current_chapter_number = ""
                current_chapter_title = ""

            current_article_number = (
                article_match.group(1)
            )

            # 某些法规可能写成：
            #
            #     第一条 为了……
            #
            # 因此需要保留条号后
            # 同一行中的正文。
            first_line_content = (
                article_match.group(2)
            )

            # 新 Article 开始，
            # 清空正文缓冲区。
            current_article_lines = []

            if first_line_content:
                current_article_lines.append(
                    first_line_content
                )

            continue

        # --------------------------------------------------
        # 6.3 普通正文
        # --------------------------------------------------
        #
        # 只有已经进入 Article 后，
        # 普通文本才属于当前 Article。
        #
        # Article 之前的通知、发文机关、
        # 发文日期等前言信息不会被错误塞入第一条。
        # --------------------------------------------------

        if current_article_number is not None:
            current_article_lines.append(
                line
            )

    # ======================================================
    # 7. EOF Flush
    # ======================================================
    #
    # 文件结束时，
    # 必须主动保存最后一条和最后一章。
    #
    # 对无显式 Chapter 的法规：
    #
    # current_chapter_number == ""
    #
    # 因此这里也会正常保存 implicit chapter。
    # ======================================================

    flush_article()
    flush_chapter()

    return chapters