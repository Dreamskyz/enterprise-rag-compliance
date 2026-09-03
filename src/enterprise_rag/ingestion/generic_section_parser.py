"""Markdown 风格通用 Section 层级解析器。"""

import re

from enterprise_rag.ingestion.models import (
    GenericSection,
)


# ==========================================================
# Markdown ATX Heading 正则。
#
# 支持：
#
#   # H1
#   ## H2
#   ### H3
#   #### H4
#   ##### H5
#   ###### H6
#
# group(1):
#     # 符号本身，用其长度得到 heading level。
#
# group(2):
#     标题正文。
#
# 当前 V1 要求：
#
#     # 与标题正文之间至少存在一个空白字符。
#
# 因此：
#
#     ## Dependencies
#
# 会被识别，
#
# 但：
#
#     ##Dependencies
#
# 暂时不会识别。
#
# 这样更接近标准 Markdown ATX Heading，
# 同时降低普通正文中 # 字符造成的误识别概率。
# ==========================================================

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def parse_generic_sections(
    text: str,
) -> list[GenericSection]:
    """
    将 Markdown 风格文本解析成一组 GenericSection。

    当前 V1 负责：

    1. 识别 H1 ~ H6 Markdown Heading；
    2. 恢复父子标题层级；
    3. 为每个 Section 构造完整 section path；
    4. 收集只属于当前 Section 的正文；
    5. 保留空 Section；
    6. 忽略第一个 Heading 之前的前导正文。

    示例输入：

        # FastAPI

        Framework documentation.

        ## Dependencies

        Dependency system.

        ### Classes as Dependencies

        A class can be a dependency.

    输出结构：

        FastAPI
            level=1
            path="FastAPI"

        Dependencies
            level=2
            path="FastAPI > Dependencies"

        Classes as Dependencies
            level=3
            path=(
                "FastAPI > Dependencies > "
                "Classes as Dependencies"
            )

    注意：

    Parser 只负责“恢复文档结构”。

    它不负责：

        Chunk 大小控制
        content_hash
        chunk_id
        retrieval_text
        ACL
        文档级 Metadata

    这些属于后续 Chunker 的职责。
    """

    # ------------------------------------------------------
    # 最终解析结果。
    # ------------------------------------------------------

    sections: list[GenericSection] = []

    # ------------------------------------------------------
    # 当前正在收集的 Section。
    #
    # 尚未遇到第一个 Heading 时为 None。
    # ------------------------------------------------------

    current_title: str | None = None

    current_level: int | None = None

    current_path: str | None = None

    current_content_lines: list[str] = []

    # ------------------------------------------------------
    # 标题层级栈。
    #
    # 每个元素保存：
    #
    #     (heading_level, heading_title)
    #
    # 例如当前位于：
    #
    #     # FastAPI
    #     ## Dependencies
    #     ### Classes
    #
    # 栈为：
    #
    #     [
    #         (1, "FastAPI"),
    #         (2, "Dependencies"),
    #         (3, "Classes"),
    #     ]
    #
    # 当之后出现新的 H2：
    #
    #     ## Security
    #
    # H3 和旧 H2 都要先弹出，
    # 然后再压入新的 H2。
    # ------------------------------------------------------

    heading_stack: list[
        tuple[int, str]
    ] = []

    def flush_section() -> None:
        """
        将当前正在收集的 Section
        写入最终 sections。

        即使正文为空也仍然保留 Section。

        原因：

        一个父级标题可能只是结构节点，例如：

            ## Dependencies
            ### Classes
            ### Sub-dependencies

        Dependencies 本身可能没有正文，
        但它依然参与 section_path 的构造，
        因而属于有效文档结构。
        """

        nonlocal current_title
        nonlocal current_level
        nonlocal current_path
        nonlocal current_content_lines

        # 尚未进入任何 Heading，
        # 不存在可以保存的 Section。
        if current_title is None:
            return

        # 理论上 title 存在时，
        # level 和 path 必然也已经存在。
        #
        # 这里显式检查，
        # 防止未来修改 Parser 时出现非法状态。
        if current_level is None:
            raise RuntimeError(
                "Section 存在 title，"
                "但缺少 heading level"
            )

        if current_path is None:
            raise RuntimeError(
                "Section 存在 title，"
                "但缺少 section path"
            )

        # 去掉 Section 正文整体首尾空行，
        # 但保留正文内部的换行结构。
        content = "\n".join(
            current_content_lines
        ).strip()

        sections.append(
            GenericSection(
                title=current_title,
                level=current_level,
                path=current_path,
                content=content,
            )
        )

        # 当前 Section 已经正式保存，
        # 清空状态，等待下一个 Heading。
        current_title = None
        current_level = None
        current_path = None
        current_content_lines = []

    # ------------------------------------------------------
    # 开始逐行扫描。
    #
    # 与法规 Parser 不同：
    #
    # 这里不能一开始删除所有空行。
    #
    # 因为通用技术文档中的段落换行
    # 可能具有可读性意义。
    #
    # 我们只在最终 flush 时去除首尾空白。
    # ------------------------------------------------------

    for raw_line in text.splitlines():
        line = raw_line.strip()

        heading_match = HEADING_PATTERN.match(
            line
        )

        # --------------------------------------------------
        # 当前行是新的 Markdown Heading。
        # --------------------------------------------------

        if heading_match:
            # 新 Section 开始前，
            # 先保存上一 Section。
            flush_section()

            heading_marks = (
                heading_match.group(1)
            )

            title = (
                heading_match.group(2).strip()
            )

            level = len(
                heading_marks
            )

            # ----------------------------------------------
            # 恢复 Heading Hierarchy。
            #
            # 新 Heading 到来时，
            # 所有 level >= 当前 level 的旧 Heading
            # 都已经不再是它的父节点。
            #
            # 示例：
            #
            #   H1 FastAPI
            #   H2 Dependencies
            #   H3 Classes
            #   H3 Sub-dependencies
            #
            # 遇到第二个 H3 时：
            #
            #   Classes
            #
            # 必须先从 stack 中弹出。
            #
            # 再例如：
            #
            #   H1 FastAPI
            #   H2 Dependencies
            #   H3 Classes
            #   H2 Security
            #
            # 新 H2 到来时，
            # H3 Classes 和旧 H2 Dependencies
            # 都需要弹出。
            # ----------------------------------------------

            while (
                heading_stack
                and heading_stack[-1][0]
                >= level
            ):
                heading_stack.pop()

            # 当前 Heading 加入层级栈。
            heading_stack.append(
                (
                    level,
                    title,
                )
            )

            # section_path 只包含真实存在的标题。
            #
            # 即使 Heading Level 跳跃：
            #
            #   # FastAPI
            #   ### Dependencies
            #
            # 也不会虚构一个不存在的 H2。
            #
            # 最终仍然得到：
            #
            #   FastAPI > Dependencies
            current_path = " > ".join(
                heading_title
                for (
                    _,
                    heading_title,
                ) in heading_stack
            )

            current_title = title
            current_level = level
            current_content_lines = []

            continue

        # --------------------------------------------------
        # 普通正文。
        # --------------------------------------------------

        if current_title is not None:
            # 已经进入某个 Section，
            # 当前普通行属于这个 Section。
            current_content_lines.append(
                raw_line.strip()
            )

        # 如果尚未遇到第一个 Heading，
        # 当前属于文档前导正文。
        #
        # V1 明确选择：
        #
        #     忽略。
        #
        # 暂时不制造：
        #
        #     Untitled Section
        #
        # 避免生成不稳定的 Retrieval Metadata。

    # ------------------------------------------------------
    # 文件结束后，
    # 最后一个 Section 不会再遇到新 Heading，
    # 因此需要手动 flush。
    # ------------------------------------------------------

    flush_section()

    return sections