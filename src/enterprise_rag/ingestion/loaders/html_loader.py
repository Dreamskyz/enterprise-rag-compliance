"""HTML 原始文件加载器。"""

import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


def load_html(path: Path) -> str:
    """
    读取完整 HTML 页面中的可见文本。

    主要用于调试。

    正式 ingestion 不应直接使用这个结果，
    因为完整网页通常包含：

    - 导航栏；
    - 页脚；
    - 推荐内容；
    - 分享按钮；
    - 其他页面噪声。

    因此正式建库应该优先使用
    针对正文区域的抽取函数。
    """

    # 从本地读取完整 HTML。
    html = path.read_text(
        encoding="utf-8"
    )

    # 把 HTML 字符串解析成 BeautifulSoup DOM 树。
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # script / style 不属于用户可见知识正文，
    # 提取文本前先彻底删除。
    for tag in soup(
        ["script", "style"]
    ):
        tag.decompose()

    # get_text() 会去掉所有 HTML 标签。
    #
    # 这里只用于调试整个页面的可见文本。
    return soup.get_text(
        separator="\n",
        strip=True,
    )


def extract_cac_article(
    path: Path,
) -> str:
    """
    从中国网信网页面提取正式文章正文。

    国家网信办当前文章页面的正文位于：

        <div id="BodyLabel">...</div>

    因此这里直接基于 DOM 容器抽取，
    避免把页头、页尾、导航栏、二维码等内容
    混入知识库。

    返回：

        不带 Markdown Heading 的普通正文文本。

    这正好适合后续 Regulation Parser。
    """

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # 在整棵 HTML DOM 树中寻找
    # 国家网信办正式文章正文容器。
    article = soup.find(
        id="BodyLabel"
    )

    if article is None:
        raise ValueError(
            "未找到国家网信办正文容器 "
            f"BodyLabel：{path}"
        )

    # 即使正文区域中意外出现 script / style，
    # 也不应该进入知识库。
    for tag in article(
        ["script", "style"]
    ):
        tag.decompose()

    # 法规 Parser 本身会识别：
    #
    # 第一章
    # 第一条
    #
    # 因此法规链路只需要纯文本，
    # 不需要人为生成 Markdown Heading。
    return article.get_text(
        separator="\n",
        strip=True,
    )


def _clean_heading_text(
    text: str,
) -> str:
    """
    清理 HTML 标题文本中的轻量格式噪声。

    当前只做非常保守的处理：

    1. 清理首尾空白；
    2. 压缩标题内部连续空白；
    3. 删除标题末尾残留的反斜杠。

    例如 OWASP 页面中存在：

        Conduct adversarial testing\\

    这种 Markdown / HTML 转换残留。

    注意：

        这里只处理格式噪声，
        不改写原始知识内容。
    """

    cleaned = text.strip()

    # 把连续普通空格 / Tab 压缩成一个空格。
    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    # 删除标题末尾可能出现的一个或多个反斜杠。
    #
    # 这里只处理末尾，
    # 不会删除正文内部本来有意义的反斜杠。
    cleaned = re.sub(
        r"\\+\s*$",
        "",
        cleaned,
    )

    return cleaned.strip()


def _extract_list_item_text(
    item: Tag,
) -> str:
    """
    提取单个 <li> 自己直接拥有的文本。

    为什么不能简单使用：

        item.get_text()

    因为一个 li 可能包含嵌套列表，例如：

        <li>
            Parent
            <ul>
                <li>Child</li>
            </ul>
        </li>

    如果直接 item.get_text()，
    父列表项会被提取成：

        Parent Child

    后面子 li 又会再次得到：

        Child

    从而造成重复。

    因此这里遍历真正的 DOM 文本节点，
    只保留“最近的 li 父节点就是当前 item”的文本。
    """

    parts: list[str] = []

    # descendants 会返回 DOM 后代节点，
    # 包括 Tag 和 NavigableString。
    #
    # 与 stripped_strings 不同，
    # NavigableString 仍然知道自己位于 DOM 的什么位置。
    for node in item.descendants:

        # 我们只处理真正的文本节点。
        if not isinstance(
            node,
            NavigableString,
        ):
            continue

        text = str(node).strip()

        if not text:
            continue

        # 找到该文本节点最近的 <li> 父节点。
        nearest_li = node.find_parent(
            "li"
        )

        # 只有最近的 li 就是当前 item 时，
        # 这段文本才真正属于当前列表项。
        #
        # 如果最近的 li 是另一个子 li，
        # 说明它属于嵌套列表，应当跳过。
        if nearest_li is item:
            parts.append(
                text
            )

    # 一个列表项内部可能由多个
    # span / strong / a 等标签共同组成，
    # 所以最后使用空格拼接。
    return " ".join(
        parts
    ).strip()


def _extract_code_block_text(
    element: Tag,
) -> str:
    """
    将 HTML <pre> 代码块转换成
    对 Generic Section Parser 安全的文本表示。

    技术文档中的代码本身也是知识，
    因此 FastAPI / Qdrant ingestion
    不能简单把 <pre><code>...</code></pre> 丢掉。

    但是直接保留代码又存在一个问题。

    Python 代码可能包含：

        # initialize model

    而 Generic Section Parser 会把：

        # ...

    当作 Markdown Heading。

    因此这里为每一行代码增加：

        |

    前缀。

    例如：

        [CODE]
        | from fastapi import FastAPI
        | app = FastAPI()

    这样既保留代码内容，
    又不会让代码注释与 Markdown Heading 冲突。

    注意：

    这不是为了生成漂亮 Markdown，
    而是为了构造适合后续 RAG ingestion 的
    结构化中间表示。
    """

    # 使用 get_text() 获取 <pre> 内部代码。
    #
    # 不使用 strip=True，
    # 因为代码内部换行具有实际意义。
    raw_code = element.get_text()

    # 统一 Windows / Unix 换行。
    raw_code = raw_code.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # 去掉整个代码块外层的空行，
    # 但不改变内部代码行顺序。
    raw_code = raw_code.strip("\n")

    if not raw_code.strip():
        return ""

    safe_lines: list[str] = []

    for line in raw_code.splitlines():

        # rstrip() 只删除行尾空白，
        # 避免网页 HTML 带来的无意义尾空格。
        #
        # 左侧缩进暂时保留在 | 后面，
        # 这样函数体层级仍可阅读。
        cleaned_line = line.rstrip()

        safe_lines.append(
            f"| {cleaned_line}"
        )

    return "\n".join(
        [
            "[CODE]",
            *safe_lines,
        ]
    )


def _html_content_to_markdown_like(
    content: Tag,
) -> str:
    """
    将已经定位好的 HTML 正文容器
    转换为 Markdown-like 文本。

    目的不是生成完整 Markdown 文档，
    而是只保留下游 Generic Section Parser
    和 Generic Section Chunker
    真正需要的知识结构。

    当前支持：

        h1 -> #
        h2 -> ##
        h3 -> ###
        ...
        h6 -> ######

        p   -> 普通正文
        li  -> 列表正文
        pre -> 技术代码块

    这样下游既能够恢复 Section 层级，
    又能够保留技术文档中的代码示例。

    这里还有一个重要设计：

        不同语义 block 之间使用一个空行分隔。

    例如：

        paragraph A

        paragraph B

    这样后续 normalize_structured_text()
    能够保留自然段边界，

    Generic Section Chunker 才能优先
    按自然段切分，
    而不是退化成固定字符硬切。
    """

    blocks: list[str] = []

    # 按 DOM 中真实出现的顺序
    # 遍历知识相关元素。
    #
    # 不直接遍历 div / span，
    # 因为它们主要属于网页布局结构。
    #
    # FastAPI 等技术文档中的 <pre>
    # 保存代码示例，因此现在也属于
    # retrieval-relevant 内容。
    for element in content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "pre",
        ]
    ):

        # --------------------------------------------------
        # 1. Heading
        # --------------------------------------------------
        if element.name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            heading_text = _clean_heading_text(
                element.get_text(
                    " ",
                    strip=True,
                )
            )

            # 空标题不进入结果。
            if not heading_text:
                continue

            # h3 -> ###
            # h4 -> ####
            heading_level = int(
                element.name[1]
            )

            blocks.append(
                f"{'#' * heading_level} "
                f"{heading_text}"
            )

            continue

        # --------------------------------------------------
        # 2. Paragraph
        # --------------------------------------------------
        if element.name == "p":

            # 如果 p 位于 li 内部，
            # 后面处理 li 时会提取同一份内容。
            #
            # 因此这里跳过，防止正文重复。
            if (
                element.find_parent("li")
                is not None
            ):
                continue

            # 如果未来某个 HTML 页面把 p 放进 pre，
            # 代码块会由 <pre> 独立处理，
            # 同样避免重复。
            if (
                element.find_parent("pre")
                is not None
            ):
                continue

            paragraph = element.get_text(
                " ",
                strip=True,
            )

            if paragraph:
                blocks.append(
                    paragraph
                )

            continue

        # --------------------------------------------------
        # 3. List Item
        # --------------------------------------------------
        if element.name == "li":
            list_text = (
                _extract_list_item_text(
                    element
                )
            )

            if list_text:

                # 统一转换成简单 Markdown bullet。
                #
                # 当前主要关心：
                #
                # 1. 列表文字不能丢；
                # 2. 列表项之间仍有边界；
                #
                # 不需要完整复刻 HTML 样式。
                blocks.append(
                    f"- {list_text}"
                )

            continue

        # --------------------------------------------------
        # 4. Code Block
        # --------------------------------------------------
        if element.name == "pre":
            code_text = (
                _extract_code_block_text(
                    element
                )
            )

            if code_text:
                blocks.append(
                    code_text
                )

    # 注意这里不是：
    #
    #     "\n".join(blocks)
    #
    # 而是：
    #
    #     "\n\n".join(blocks)
    #
    # 目的是在不同 HTML 语义块之间
    # 显式保留自然段边界。
    return "\n\n".join(
        blocks
    )


def extract_owasp_article(
    path: Path,
    title: str,
) -> str:
    """
    从 OWASP GenAI LLM Risk 页面
    提取结构化正文。

    OWASP 当前 LLM Risk 页面使用 Elementor，
    文章正文位于：

        .elementor-widget-xpro-post-content
            .xpro-elementor-content

    与 CAC 法规不同：

    OWASP 的 H3 / H4 Heading
    本身携带重要语义层级，
    因此不能简单使用：

        get_text()

    否则 Heading 层级会丢失。

    本函数最终返回 Markdown-like 文本，例如：

        ## LLM01:2025 Prompt Injection

        Introduction...

        ### Types of Prompt Injection Vulnerabilities

        #### Direct Prompt Injections

        ...

    这样可以直接交给
    Generic Section Parser。
    """

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # OWASP 当前文章正文的语义容器。
    #
    # 不直接使用 <main id="main">，
    # 因为 main 中还包含：
    #
    # - 页面标题；
    # - 推荐的其他 LLM Top 10 风险；
    # - 页面布局；
    # - 其他非正文内容。
    content = soup.select_one(
        ".elementor-widget-xpro-post-content "
        ".xpro-elementor-content"
    )

    if content is None:
        raise ValueError(
            "未找到 OWASP 正文容器 "
            ".elementor-widget-xpro-post-content "
            f".xpro-elementor-content：{path}"
        )

    # 删除正文容器内部可能存在的
    # script / style 噪声。
    for tag in content(
        ["script", "style"]
    ):
        tag.decompose()

    # WordPress / Jetpack 页面功能区域
    # 不属于知识正文。
    #
    # 例如：
    #
    # Share this:
    # Print
    # Email
    # X
    #
    # 这些都不应该进入向量库。
    for selector in (
        ".sharedaddy",
        ".sd-sharing",
        ".robots-nocontent",
        "#jp-relatedposts",
        ".gsp_post_data",
    ):
        for tag in content.select(
            selector
        ):
            tag.decompose()

    # Manifest 中的 title 是我们自己管理的
    # canonical 文档标题。
    #
    # OWASP 页面主标题和正文位于两个独立 widget，
    # 因此这里主动补一个 H2 根标题。
    #
    # 这样正文最前面的 introduction 就不会成为
    # Generic Parser 所说的 heading 前 preamble，
    # 从而避免丢失。
    root_title = _clean_heading_text(
        title
    )

    if not root_title:
        raise ValueError(
            "OWASP 文档标题不能为空"
        )

    body_text = (
        _html_content_to_markdown_like(
            content
        )
    )

    if not body_text.strip():
        raise ValueError(
            f"OWASP 正文为空：{path}"
        )

    # 根标题和正文之间同样保留一个空行。
    #
    # 这样 structured normalizer
    # 不会丢掉根 Section 中 introduction
    # 的第一个自然段边界。
    return (
        f"## {root_title}\n\n"
        f"{body_text}"
    )


def extract_fastapi_article(
    path: Path,
) -> str:
    """
    从 FastAPI 官方文档页面中
    提取结构化技术正文。

    当前 FastAPI 官方文档正文位于：

        <article class="md-content__inner md-typeset">

    页面其他区域还包含：

    - 顶部 Header；
    - 左侧全站导航；
    - 右侧 On this page；
    - Sponsor / Banner；
    - Footer；
    - 页面路径导航。

    这些内容都不应该进入知识库。

    FastAPI 正文自身已经包含：

        h1
        h2
        h3
        p
        ul / li
        pre / code

    因此不需要像 OWASP 一样
    人工补充根标题。

    本函数只负责：

    1. 准确定位正文 DOM；
    2. 删除正文内部无关功能元素；
    3. 将 HTML 语义结构转换为
       Markdown-like 中间表示。

    后续统一交给：

        normalize_structured_text()
        parse_generic_sections()
        build_generic_section_chunks()

    处理。
    """

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # FastAPI 当前官方文档使用
    # Material / Zensical 风格页面结构。
    #
    # 真正文章正文位于：
    #
    # <article class="md-content__inner md-typeset">
    #
    # 不能选择整个 <main class="md-main">，
    # 因为 main 里面同时还存在左右侧导航栏。
    content = soup.select_one(
        "article.md-content__inner.md-typeset"
    )

    if content is None:
        raise ValueError(
            "未找到 FastAPI 正文容器 "
            "article.md-content__inner.md-typeset："
            f"{path}"
        )

    # script / style 绝不进入知识库。
    for tag in content(
        ["script", "style"]
    ):
        tag.decompose()

    # FastAPI Heading 内通常存在：
    #
    # <a class="headerlink" ...>¶</a>
    #
    # 这是用于网页锚点跳转的 UI 元素，
    # 不属于真正标题内容。
    #
    # 如果不删除，
    # Heading 可能会变成：
    #
    # Dependencies ¶
    #
    # 因此必须在 Heading 文本提取前删除。
    for tag in content.select(
        "a.headerlink"
    ):
        tag.decompose()

    # details 通常包含类似：
    #
    # "Other versions and variants"
    #
    # FastAPI 页面会在里面重复展示
    # 旧 Python 版本或其他等价写法。
    #
    # 当前 V1 的目标不是建立完整 API reference，
    # 而是构建干净、低重复的技术规范语料。
    #
    # 因此先移除这些版本展开区，
    # 避免同一知识因为多个 Python 写法
    # 被重复注入多个 Chunk。
    for tag in content.find_all(
        "details"
    ):
        tag.decompose()

    body_text = (
        _html_content_to_markdown_like(
            content
        )
    )

    if not body_text.strip():
        raise ValueError(
            f"FastAPI 正文为空：{path}"
        )

    return body_text