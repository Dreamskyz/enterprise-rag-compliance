"""文档文本标准化模块。"""

import re
# re 是 Python 内置的正则表达式模块。
# 这个文件里主要用它处理：
#
#     re.sub(r"[ \t]+", " ", line)
#
# 即：
# 把连续多个普通空格或 Tab 压缩成一个空格。


def _normalize_common_whitespace(
    text: str,
) -> str:
    """
    执行普通文本和结构化文本都需要的基础空白标准化。

    这里不决定“是否保留空行”。

    原因是：
    - 普通法规文本希望删除空行；
    - Markdown-like 结构化文本需要保留段落边界。

    所以把两种 Normalizer 共有的逻辑抽到这里。
    """

    # Windows 换行：
    # \r\n
    #
    # 旧式 Mac 换行：
    # \r
    #
    # 全部统一为：
    # \n
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # 全角空格统一成普通空格。
    text = text.replace(
        "\u3000",
        " ",
    )

    # 不换行空格（NBSP）统一成普通空格。
    text = text.replace(
        "\xa0",
        " ",
    )

    return text


def _normalize_line(
    raw_line: str,
) -> str:
    """
    标准化单独一行文本。

    当前规则：
    1. 删除行首行尾空白；
    2. 连续普通空格 / Tab 压缩成一个空格。
    """

    # 删除字符串开头和结尾的空白字符。
    line = raw_line.strip()

    # re.sub(
    #     要找什么,
    #     替换成什么,
    #     在哪个字符串中找,
    # )
    #
    # [ \t]+：
    # 一个或多个连续的普通空格 / Tab。
    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    )

    return line


def normalize_text(
    text: str,
) -> str:
    """
    对已经抽取出的普通正文文本进行基础标准化。

    只处理“格式问题”，
    不负责判断网页哪些区域属于正文。

    当前规则：
    1. 统一换行符；
    2. 清理行首行尾空白；
    3. 删除连续空行；
    4. 统一常见特殊空格。

    注意：
    这个函数保持原来的语义，
    当前主要服务 regulation 文档。

    对于 Markdown-like 结构化文档，
    应使用 normalize_structured_text()。
    """

    # 先执行两种 Normalizer
    # 都需要的基础标准化。
    text = _normalize_common_whitespace(
        text
    )

    # 创建一个空列表，
    # 用来保存清洗后的有效文本行。
    lines: list[str] = []

    # 按换行符逐行处理文本。
    for raw_line in text.splitlines():
        line = _normalize_line(
            raw_line
        )

        # 普通文本 Normalizer：
        #
        # 只有非空行才保存。
        #
        # 因此所有空行最终都会被删除，
        # 这与原来的 normalize_text()
        # 行为保持一致。
        if line:
            lines.append(
                line
            )

    # 把所有有效行重新拼起来。
    return "\n".join(
        lines
    )


def normalize_structured_text(
    text: str,
) -> str:
    """
    对 Markdown-like 结构化文本进行标准化。

    与 normalize_text() 最大的区别：

        保留“一个空行”。

    为什么？

    对于 OWASP / FastAPI / Qdrant
    这类结构化文档：

        paragraph A

        paragraph B

    中间的空行不是无意义噪声，
    而是在表达：

        “这里是两个不同自然段。”

    Generic Section Chunker 会利用这个信息：

        re.split(r"\\n\\s*\\n", content)

    优先按自然段聚合 Chunk。

    因此这里必须保留段落边界。

    规则：
    1. 统一换行符；
    2. 统一特殊空格；
    3. 清理每行两端空白；
    4. 压缩行内连续空格 / Tab；
    5. 连续多个空行最多保留一个；
    6. 不保留文本末尾多余空行。
    """

    text = _normalize_common_whitespace(
        text
    )

    lines: list[str] = []

    # 记录上一行最终是否已经是空行。
    #
    # 用它避免：
    #
    # paragraph A
    #
    #
    #
    # paragraph B
    #
    # 保留三个空行。
    #
    # 最终只保留：
    #
    # paragraph A
    #
    # paragraph B
    previous_blank = False

    for raw_line in text.splitlines():
        line = _normalize_line(
            raw_line
        )

        if line:
            # 当前是有效文本行。
            lines.append(
                line
            )

            previous_blank = False

        elif (
            lines
            and not previous_blank
        ):
            # 当前是空行。
            #
            # 只有：
            # 1. 前面已经有正文；
            # 2. 上一个保存的不是空行；
            #
            # 才保留一个空字符串，
            # 用来表达段落边界。
            lines.append(
                ""
            )

            previous_blank = True

    # 如果原文本末尾存在空行：
    #
    # paragraph
    #
    #
    #
    #
    # 前面的逻辑可能会留下一个 ""。
    #
    # 末尾空行没有结构价值，
    # 所以删除。
    while (
        lines
        and lines[-1] == ""
    ):
        lines.pop()

    return "\n".join(
        lines
    )