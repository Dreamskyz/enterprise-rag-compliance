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
CHAPTER_PATTERN = re.compile(                                 #章节正则,^表示必须从这一行开头开始匹配,+表示前面的字符出现至少 1 次,\s 表示空白字符,*表示0 次或多次
    r"^(第[一二三四五六七八九十百]+章)\s*(.*)$"                    #(.*)，. 表示任意字符    因此这里表示把“第一章”后面的内容全部拿出来
)                                                             #$表示匹配到这一行结尾


# 匹配：
# 第一条
# 第二十四条
#ARTICLE_PATTERN = re.compile(                                 #Article 正则
    #r"^(第[一二三四五六七八九十百]+条)\s*(.*)$"
#)
ARTICLE_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百]+条)(?:\s+(.*))?$"                 #(?:...)表示一个分组，但是我不需要把这个分组本身单独保存下来 ， \s+表示至少 1 个空白字符，末尾的 ? 表示这一整部分可以不存在
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

    lines = [                                   #文本预处理，这是一个列表推导式，作用是删除空行，同时清除每行首尾空白
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chapters: list[RegulationChapter] = []      #最终结果容器

    current_chapter_number: str | None = None   #当前正在解析的章节编号
    current_chapter_title: str = ""             #保存当前章节标题

    current_articles: list[RegulationArticle] = []   #用于暂时保存当前这一章已经解析完成的 Article

    current_article_number: str | None = None        #当前正在解析的条
    current_article_lines: list[str] = []            #当前正在解析的条

    def flush_article() -> None:                    #把“当前正在收集的 Article”正式保存进 current_articles
        """
        将当前正在收集的条款写入 current_articles。
        """

        nonlocal current_article_number             #nonlocal，不要创建新的局部变量，操作外层 parse_regulation() 里面那个变量
        nonlocal current_article_lines

        if current_article_number is None:          #判断有没有正在解析的 Article
            return

        content = "\n".join(current_article_lines).strip()      #合并正文

        current_articles.append(                        #创建 Article
            RegulationArticle(                          #先创建RegulationArticle(...)
                article_number=current_article_number,
                content=content,
            )
        )

        current_article_number = None                   #然后加入当前章节
        current_article_lines = []                      #清空 Article 状态，因为上一条已经结束了

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

    for line in lines:                                      #开始逐行扫描
        chapter_match = CHAPTER_PATTERN.match(line)         #先判断是不是 Chapter

        if chapter_match:
            # 新章开始前，先结束上一条。
            flush_article()

            # 再结束上一章。
            flush_chapter()

            current_chapter_number = chapter_match.group(1)    #保存新章节编号
            current_chapter_title = chapter_match.group(2)     #保存章节标题

            continue

        article_match = ARTICLE_PATTERN.match(line)             #判断 Article，如果当前行不是 Chapter，就继续看看是不是 Article

        if article_match:                                       #如果发现新 Article，进入 Article 处理逻辑
            # 新条开始前，先把上一条保存。
            flush_article()

            current_article_number = article_match.group(1)

            # 有些文档可能是：
            # 第一条 为了……
            # 因此需要保留同一行后面的正文。
            first_line_content = article_match.group(2)

            current_article_lines = []                          #新 Article 开始，所以清空正文列表

            if first_line_content:                              #判断同一行有没有正文
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