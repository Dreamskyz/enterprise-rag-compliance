"""HTML 原始文件加载器。"""

import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


# ============================================================
# Qdrant V1 保留的代码示例语言
# ============================================================
#
# Qdrant 官方文档经常针对同一个知识点同时提供：
#
# JSON
# HTTP
# Python
# TypeScript
# Rust
# Java
# C#
# Go
#
# 对于当前这个 Python RAG 项目，
# 如果全部保留，会导致：
#
# 1. Corpus 明显膨胀；
# 2. 同一个语义出现大量重复 Chunk；
# 3. Retrieval 中产生无意义的候选竞争。
#
# 因此 V1 有意只保留：
#
# JSON   -> 数据结构
# HTTP   -> Qdrant 原始 API 语义
# Python -> 当前项目真实使用的客户端语言
#
# 没有标记语言的代码块暂时也保留，
# 避免误删普通文本型示例。
QDRANT_RETAINED_CODE_LANGUAGES = {
    "json",
    "http",
    "python",
}


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


def _extract_owasp_pseudo_heading(
    text: str,
) -> str | None:
    """
    识别 OWASP 页面中的特殊伪 Heading。

    当前部分 OWASP GenAI 页面源码中存在：

        <p>###@ Sanitization:</p>
        <p>###@ Access Controls:</p>

    也就是说：

        页面原始 Markdown 中本应具有标题语义的内容，

    在最终 HTML 中并没有被渲染成：

        <h3>...</h3>

    而是错误地保留成了普通：

        <p>...</p>

    如果直接按照普通 Paragraph 处理，
    Generic Section Parser 最终会把：

        ###@ Sanitization:

    当成正文，
    从而丢失原始 Section 层级。

    因此这里只针对 OWASP 这一类
    source-specific anomaly 做兼容。

    支持格式：

        #@ Title
        ##@ Title
        ###@ Title
        ...
        ######@ Title

    返回：

        合法 Markdown-like Heading，
        例如：

            ### Sanitization:

    如果不是该模式，则返回 None。

    注意：

    这里不做：

        text.replace("###@", "### ")

    这种全局字符串替换。

    只有当整个 Paragraph 本身符合
    pseudo-heading 模式时，
    才把它恢复为 Heading，
    避免误伤普通正文。
    """

    cleaned = text.strip()

    match = re.fullmatch(
        r"(#{1,6})@\s*(.+)",
        cleaned,
    )

    if match is None:
        return None

    hashes = match.group(1)

    heading_text = _clean_heading_text(
        match.group(2)
    )

    if not heading_text:
        return None

    return (
        f"{hashes} "
        f"{heading_text}"
    )


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
    *,
    recognize_owasp_pseudo_headings: bool = False,
    include_asides: bool = False,
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

    当：

        include_asides=True

    时，还会把：

        <aside>...</aside>

    作为知识正文保留下来。

    这主要用于 Qdrant 官方技术文档，
    因为它会把：

        性能建议；
        索引建议；
        API 使用注意事项；

    放入 aside 中。

    如果直接忽略 aside，
    会静默丢失具有实际技术价值的知识。

    当：

        recognize_owasp_pseudo_headings=True

    时，还会识别 OWASP 页面中的：

        <p>###@ Sanitization:</p>

    并恢复成：

        ### Sanitization:

    这种合法 Markdown-like Heading。

    默认两个兼容选项都关闭，
    避免某一个数据源的特殊规则
    意外改变其他数据源的历史 ingestion 行为。

    这样下游既能够恢复 Section 层级，
    又能够保留技术文档中的代码示例。

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

    # --------------------------------------------------
    # 根据调用方能力决定需要遍历哪些 HTML 元素。
    #
    # aside 并不是所有数据源都需要，
    # 所以默认不加入。
    # --------------------------------------------------
    element_names = [
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

    if include_asides:
        element_names.append(
            "aside"
        )

    # 按 DOM 中真实出现的顺序
    # 遍历知识相关元素。
    #
    # 不直接遍历 div / span，
    # 因为它们主要属于网页布局结构。
    for element in content.find_all(
        element_names
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

            # 如果 p 位于 pre 内部，
            # 代码块会由 <pre> 独立处理，
            # 同样避免重复。
            if (
                element.find_parent("pre")
                is not None
            ):
                continue

            # 当 include_asides=True 时，
            # aside 会作为一个整体语义块处理。
            #
            # 如果某些页面的 aside 内部又嵌套了 p，
            # 这里必须跳过内部 p，
            # 否则同一段文字会出现两次。
            if (
                include_asides
                and element.find_parent("aside")
                is not None
            ):
                continue

            paragraph = element.get_text(
                " ",
                strip=True,
            )

            if not paragraph:
                continue

            # --------------------------------------------------
            # OWASP 特殊兼容：
            #
            # 某些 OWASP 页面源码存在：
            #
            # <p>###@ Sanitization:</p>
            #
            # 这种 paragraph 实际承担 Heading 语义。
            #
            # 只有 OWASP 调用方显式开启该功能时，
            # 才进行识别。
            # --------------------------------------------------
            if recognize_owasp_pseudo_headings:
                pseudo_heading = (
                    _extract_owasp_pseudo_heading(
                        paragraph
                    )
                )

                if pseudo_heading is not None:
                    blocks.append(
                        pseudo_heading
                    )

                    continue

            blocks.append(
                paragraph
            )

            continue

        # --------------------------------------------------
        # 3. List Item
        # --------------------------------------------------
        if element.name == "li":

            # 如果某个 aside 内部包含列表，
            # 当整个 aside 已经被保留时，
            # 不再重复提取内部 li。
            if (
                include_asides
                and element.find_parent("aside")
                is not None
            ):
                continue

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

            # 如果未来 aside 中存在 pre，
            # 整个 aside 已经被作为一个语义块保留，
            # 因此不重复注入代码。
            if (
                include_asides
                and element.find_parent("aside")
                is not None
            ):
                continue

            code_text = (
                _extract_code_block_text(
                    element
                )
            )

            if code_text:
                blocks.append(
                    code_text
                )

            continue

        # --------------------------------------------------
        # 5. Aside
        # --------------------------------------------------
        if element.name == "aside":

            # Qdrant 会使用 aside 保存一些
            # 具有实际知识价值的 Note / Tip。
            #
            # 例如：
            #
            # For performant filtering,
            # create payload indexes...
            #
            # 这些内容不是网页 UI，
            # 因此必须进入知识库。
            aside_text = element.get_text(
                " ",
                strip=True,
            )

            if aside_text:
                # 添加一个轻量 [NOTE] 标记，
                # 让 Chunk 中仍然能够看出
                # 这是官方文档中特别强调的说明。
                #
                # 它不会干扰 Generic Section Parser，
                # 因为它不是 Markdown Heading。
                blocks.append(
                    f"[NOTE] {aside_text}"
                )

            continue

    # 不同 HTML 语义块之间
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

    另外，部分 OWASP 页面源码存在：

        <p>###@ Sanitization:</p>

    这类没有被 HTML 正确渲染成 Heading 的
    Markdown-like 残留。

    当前 Extractor 会在 OWASP 链路中
    将其恢复为合法 Heading：

        ### Sanitization:

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
    root_title = _clean_heading_text(
        title
    )

    if not root_title:
        raise ValueError(
            "OWASP 文档标题不能为空"
        )

    body_text = (
        _html_content_to_markdown_like(
            content,
            recognize_owasp_pseudo_headings=True,
        )
    )

    if not body_text.strip():
        raise ValueError(
            f"OWASP 正文为空：{path}"
        )

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
    """

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # FastAPI 当前官方文档真正正文区域。
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

    # 删除 Heading 锚点中的 ¶。
    for tag in content.select(
        "a.headerlink"
    ):
        tag.decompose()

    # 删除 FastAPI 不同 Python 版本
    # 重复展示的 details 区域。
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


def _get_qdrant_code_language(
    pre: Tag,
) -> str | None:
    """
    获取 Qdrant <pre> 代码块的语言。

    Qdrant 当前 HTML 大致为：

        <pre class="chroma">
            <code
                class="language-python"
                data-lang="python"
            >
                ...
            </code>
        </pre>

    因此优先读取：

        data-lang

    如果 data-lang 不存在，
    再尝试从：

        class="language-python"

    中恢复语言。

    最终如果完全无法识别，
    返回 None。
    """

    code = pre.find(
        "code"
    )

    if code is None:
        return None

    # --------------------------------------------------
    # 1. 优先使用 data-lang。
    # --------------------------------------------------
    data_lang = code.get(
        "data-lang"
    )

    if isinstance(
        data_lang,
        str,
    ):
        cleaned = (
            data_lang
            .strip()
            .lower()
        )

        if cleaned:
            return cleaned

    # --------------------------------------------------
    # 2. fallback：
    #    class="language-python"
    # --------------------------------------------------
    classes = code.get(
        "class",
        [],
    )

    for class_name in classes:
        if not isinstance(
            class_name,
            str,
        ):
            continue

        if class_name.startswith(
            "language-"
        ):
            language = (
                class_name[
                    len("language-"):
                ]
                .strip()
                .lower()
            )

            if language:
                return language

    return None


def _remove_redundant_qdrant_code_blocks(
    content: Tag,
) -> None:
    """
    删除 Qdrant 中当前项目不需要的重复 SDK 代码示例。

    Qdrant 官方文档常常对一个相同功能同时展示：

        HTTP
        Python
        TypeScript
        Rust
        Java
        C#
        Go

    如果全部进入 RAG Corpus：

    - 会增加大量高度相似 Chunk；
    - 会放大技术文档在检索候选中的占比；
    - 会让同一知识点产生重复竞争；
    - 还会增加 Embedding / Rerank 成本。

    当前项目技术栈是 Python，
    因此 V1 只保留：

        JSON
        HTTP
        Python

    没有语言标签的代码块暂时保留，
    防止误删普通示例。

    注意：

    这是 Corpus Curating，
    不是声称其他语言示例“不正确”。

    我们只是为当前企业 RAG 助手
    选择最低重复、最相关的技术证据。
    """

    for pre in list(
        content.find_all("pre")
    ):
        language = (
            _get_qdrant_code_language(
                pre
            )
        )

        # 无语言标记的代码块保守保留。
        if language is None:
            continue

        if (
            language
            not in QDRANT_RETAINED_CODE_LANGUAGES
        ):
            # Qdrant 代码块通常位于：
            #
            # <div class="highlight">
            #     <pre>...</pre>
            # </div>
            #
            # 如果只删除 pre，
            # 会留下一个没有意义的空 div。
            #
            # 因此如果父元素正好是
            # highlight wrapper，
            # 就连 wrapper 一起删除。
            parent = pre.parent

            if (
                isinstance(parent, Tag)
                and parent.name == "div"
                and "highlight"
                in parent.get(
                    "class",
                    [],
                )
            ):
                parent.decompose()
            else:
                pre.decompose()


def extract_qdrant_article(
    path: Path,
) -> str:
    """
    从 Qdrant 官方文档页面中
    提取结构化技术正文。

    当前 Qdrant 官方文档正文位于：

        <article class="documentation-article">

    页面外部同时还包含：

    - 顶部导航；
    - 左侧 Documentation Sidebar；
    - Breadcrumb；
    - 搜索 UI；
    - Footer；
    - 其他页面布局。

    因此不能选择整个：

        <main>

    而必须只选择：

        article.documentation-article

    Qdrant 还有两个与 FastAPI 不同的特点。

    第一：

        部分重要技术提示位于：

            <aside>

        例如 Filtering 文档中，
        payload index 的性能建议就在 aside 中。

        因此 Qdrant ingestion 必须保留 aside。

    第二：

        同一个 API 示例往往会提供多个 SDK 版本。

        当前 V1 只保留：

            JSON
            HTTP
            Python

        来减少高度重复的技术 Chunk。

    最终输出仍然是统一的：

        Markdown-like structured text

    然后继续交给：

        normalize_structured_text()
        parse_generic_sections()
        build_generic_section_chunks()

    因此 Qdrant 的加入不会改变
    Parser / Chunker 的统一架构。
    """

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # --------------------------------------------------
    # 1. 精确定位 Qdrant 正文。
    # --------------------------------------------------
    content = soup.select_one(
        "article.documentation-article"
    )

    if content is None:
        raise ValueError(
            "未找到 Qdrant 正文容器 "
            "article.documentation-article："
            f"{path}"
        )

    # --------------------------------------------------
    # 2. 删除绝对不属于知识正文的脚本/样式。
    # --------------------------------------------------
    for tag in content(
        ["script", "style"]
    ):
        tag.decompose()

    # --------------------------------------------------
    # 3. 去掉高度重复的多语言 SDK 示例。
    #
    # 必须在 HTML -> Markdown-like 转换之前做。
    #
    # 因为一旦转换成普通文本，
    # 就很难再稳定知道某个 Code Block
    # 原本到底属于 Python、Rust 还是 Java。
    # --------------------------------------------------
    _remove_redundant_qdrant_code_blocks(
        content
    )

    # --------------------------------------------------
    # 4. 转换成统一的 Structured Text。
    #
    # 与 FastAPI 的主要区别：
    #
    # Qdrant 显式开启 include_asides=True，
    # 保留官方技术提示。
    # --------------------------------------------------
    body_text = (
        _html_content_to_markdown_like(
            content,
            include_asides=True,
        )
    )

    if not body_text.strip():
        raise ValueError(
            f"Qdrant 正文为空：{path}"
        )

    return body_text