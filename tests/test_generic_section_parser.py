"""通用 Section Parser 测试。"""

from enterprise_rag.ingestion.generic_section_parser import (
    parse_generic_sections,
)


def test_parse_basic_heading_hierarchy() -> None:
    """
    应正确解析：

        H1
        H2
        H3

    并生成完整 section path。
    """

    text = """
# FastAPI

FastAPI framework documentation.

## Dependencies

FastAPI has a powerful dependency injection system.

### Classes as Dependencies

A Python class can be used as a dependency.
""".strip()

    sections = parse_generic_sections(
        text
    )

    assert len(sections) == 3

    assert sections[0].title == "FastAPI"
    assert sections[0].level == 1
    assert sections[0].path == "FastAPI"

    assert (
        sections[0].content
        == "FastAPI framework documentation."
    )

    assert (
        sections[1].title
        == "Dependencies"
    )

    assert sections[1].level == 2

    assert (
        sections[1].path
        == "FastAPI > Dependencies"
    )

    assert (
        sections[1].content
        == (
            "FastAPI has a powerful "
            "dependency injection system."
        )
    )

    assert (
        sections[2].title
        == "Classes as Dependencies"
    )

    assert sections[2].level == 3

    assert (
        sections[2].path
        == (
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        )
    )

    assert (
        sections[2].content
        == (
            "A Python class can be used "
            "as a dependency."
        )
    )


def test_same_level_headings_are_siblings() -> None:
    """
    两个相同 level 的 Heading
    应互为兄弟节点，而不是父子节点。
    """

    text = """
# FastAPI

## Dependencies

### Classes as Dependencies

Class dependency content.

### Sub-dependencies

Sub-dependency content.
""".strip()

    sections = parse_generic_sections(
        text
    )

    assert len(sections) == 4

    classes_section = sections[2]

    sub_dependencies_section = (
        sections[3]
    )

    assert (
        classes_section.path
        == (
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        )
    )

    assert (
        sub_dependencies_section.path
        == (
            "FastAPI > Dependencies > "
            "Sub-dependencies"
        )
    )

    # 错误情况会变成：
    #
    # FastAPI > Dependencies >
    # Classes as Dependencies >
    # Sub-dependencies
    #
    # 这个断言保护同级 Heading 的 sibling 语义。
    assert (
        "Classes as Dependencies"
        not in sub_dependencies_section.path
    )


def test_heading_can_return_from_deep_level_to_shallow_level() -> None:
    """
    从 H3 回到 H2 时，
    旧的 H3 和旧 H2 都应该正确退出层级栈。
    """

    text = """
# FastAPI

## Dependencies

### Classes as Dependencies

Class dependency content.

## Security

Security content.
""".strip()

    sections = parse_generic_sections(
        text
    )

    security_section = sections[-1]

    assert (
        security_section.title
        == "Security"
    )

    assert (
        security_section.level
        == 2
    )

    assert (
        security_section.path
        == "FastAPI > Security"
    )

    assert (
        security_section.content
        == "Security content."
    )


def test_heading_level_jump_does_not_create_fake_parent() -> None:
    """
    Markdown Heading Level 可以发生跳跃。

    例如：

        H1
        H3

    Parser 不应该虚构不存在的 H2。
    """

    text = """
# FastAPI

### Dependencies

Dependency content.
""".strip()

    sections = parse_generic_sections(
        text
    )

    assert len(sections) == 2

    dependencies = sections[1]

    # 保留真实原始 level。
    assert dependencies.level == 3

    # 但 path 只包含真实存在的标题。
    assert (
        dependencies.path
        == "FastAPI > Dependencies"
    )


def test_empty_parent_section_is_preserved() -> None:
    """
    父级 Heading 即使没有自己的正文，
    也应该作为结构节点保留下来。

    因为它仍然参与子 Section 的 path 构造。
    """

    text = """
# FastAPI

## Dependencies

### Classes as Dependencies

Class dependency content.
""".strip()

    sections = parse_generic_sections(
        text
    )

    dependencies = sections[1]

    assert (
        dependencies.title
        == "Dependencies"
    )

    assert dependencies.content == ""

    assert (
        sections[2].path
        == (
            "FastAPI > Dependencies > "
            "Classes as Dependencies"
        )
    )


def test_preamble_before_first_heading_is_ignored() -> None:
    """
    第一个 Heading 之前的正文，
    V1 暂时不生成匿名 Section。

    这可以避免：

        Untitled
        Preamble
        Root

    等人工虚构标题污染 Retrieval Metadata。
    """

    text = """
This text appears before the first heading.

It should not become an anonymous section.

# FastAPI

Framework documentation.
""".strip()

    sections = parse_generic_sections(
        text
    )

    assert len(sections) == 1

    assert sections[0].title == "FastAPI"

    assert (
        sections[0].content
        == "Framework documentation."
    )

    assert (
        "This text appears"
        not in sections[0].content
    )


def test_empty_text_returns_empty_sections() -> None:
    """
    空文本不应该报错，
    应直接得到空 Section 列表。
    """

    sections = parse_generic_sections(
        ""
    )

    assert sections == []


def test_text_without_heading_returns_empty_sections() -> None:
    """
    没有任何 Markdown Heading 的纯正文，
    在 V1 中没有稳定结构边界。

    因而不创建匿名 Section。
    """

    text = """
This document contains plain text only.

There is no heading.
""".strip()

    sections = parse_generic_sections(
        text
    )

    assert sections == []