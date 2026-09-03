"""FastAPI 官方技术文档 HTML Loader 测试。"""

from pathlib import Path

import pytest

from enterprise_rag.ingestion.loaders.html_loader import (
    extract_fastapi_article,
)


def _write_html(
    tmp_path: Path,
    html: str,
) -> Path:
    """
    把测试 HTML 写入临时文件。

    Loader 的正式接口接收 Path，
    因此测试也尽量按照真实调用方式执行，
    而不是绕过文件读取过程。
    """

    path = tmp_path / "fastapi_test.html"

    path.write_text(
        html,
        encoding="utf-8",
    )

    return path


def test_extract_fastapi_article_preserves_heading_hierarchy(
    tmp_path: Path,
) -> None:
    """
    FastAPI 技术文档中的 Heading
    必须转换成 Markdown-like Heading。

    这是后续 Generic Section Parser
    恢复 Section 层级的前提。
    """

    path = _write_html(
        tmp_path,
        """
        <html>
            <body>
                <article class="md-content__inner md-typeset">
                    <h1>
                        Dependencies
                        <a class="headerlink">¶</a>
                    </h1>

                    <p>Dependency introduction.</p>

                    <h2>
                        First Steps
                        <a class="headerlink">¶</a>
                    </h2>

                    <p>First steps content.</p>

                    <h3>
                        Import Depends
                        <a class="headerlink">¶</a>
                    </h3>

                    <p>Import content.</p>
                </article>
            </body>
        </html>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert "# Dependencies" in text
    assert "## First Steps" in text
    assert "### Import Depends" in text


def test_extract_fastapi_article_removes_headerlink_noise(
    tmp_path: Path,
) -> None:
    """
    FastAPI Heading 内的 ¶
    只是网页锚点 UI，

    不应该进入：
    - section_title；
    - section_path；
    - retrieval_text；
    - embedding。
    """

    path = _write_html(
        tmp_path,
        """
        <article class="md-content__inner md-typeset">
            <h1>
                Dependencies
                <a class="headerlink"
                   href="#dependencies">
                    ¶
                </a>
            </h1>

            <p>Content.</p>
        </article>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert "# Dependencies" in text
    assert "¶" not in text


def test_extract_fastapi_article_preserves_paragraphs_and_lists(
    tmp_path: Path,
) -> None:
    """
    技术文档不仅有 Heading，
    普通说明和列表也是重要检索证据。

    同时不同 block 之间应该保留空行，
    供 structured normalizer 和
    Generic Section Chunker 使用。
    """

    path = _write_html(
        tmp_path,
        """
        <article class="md-content__inner md-typeset">
            <h1>Dependencies</h1>

            <p>Dependency injection is useful.</p>

            <ul>
                <li>Share database connections.</li>
                <li>Enforce security requirements.</li>
            </ul>
        </article>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert (
        "Dependency injection is useful."
        in text
    )

    assert (
        "- Share database connections."
        in text
    )

    assert (
        "- Enforce security requirements."
        in text
    )

    # block 之间应该存在自然段边界。
    assert "\n\n" in text


def test_extract_fastapi_article_preserves_code_block(
    tmp_path: Path,
) -> None:
    """
    FastAPI 是技术文档。

    如果 Loader 丢掉代码，
    RAG 就只能回答概念，
    无法给出有证据支持的技术写法。

    因此 <pre> 中代码必须保留。
    """

    path = _write_html(
        tmp_path,
        """
        <article class="md-content__inner md-typeset">
            <h1>Dependencies</h1>

            <p>Example:</p>

            <pre><code>from fastapi import Depends, FastAPI

app = FastAPI()</code></pre>
        </article>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert "[CODE]" in text

    assert (
        "| from fastapi import Depends, FastAPI"
        in text
    )

    assert (
        "| app = FastAPI()"
        in text
    )


def test_code_comment_does_not_become_markdown_heading(
    tmp_path: Path,
) -> None:
    """
    代码里的：

        # Load the model

    不能直接出现在行首，

    否则 Generic Section Parser
    会把它错误识别为 H1。

    当前 contract 是：
    所有代码行都以 '| ' 开头。
    """

    path = _write_html(
        tmp_path,
        """
        <article class="md-content__inner md-typeset">
            <h1>Lifespan Events</h1>

            <pre><code>async def lifespan():
    # Load the model
    yield
    # Clean up the model</code></pre>
        </article>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert (
        "|     # Load the model"
        in text
    )

    assert (
        "|     # Clean up the model"
        in text
    )

    # 真正危险的是代码注释独立出现在行首。
    assert (
        "\n# Load the model\n"
        not in text
    )

    assert (
        "\n# Clean up the model\n"
        not in text
    )


def test_extract_fastapi_article_removes_details_variants(
    tmp_path: Path,
) -> None:
    """
    FastAPI 官方文档中的 details
    经常包含：

        Other versions and variants

    以及不同 Python 版本的重复代码。

    当前 Corpus V2 baseline
    只保留正文推荐版本，
    避免人为制造大量重复语料。
    """

    path = _write_html(
        tmp_path,
        """
        <article class="md-content__inner md-typeset">
            <h1>Dependencies</h1>

            <p>Canonical explanation.</p>

            <details>
                <summary>
                    Other versions and variants
                </summary>

                <p>
                    Legacy Python variant that
                    should not enter the corpus.
                </p>

                <pre><code>legacy_example()</code></pre>
            </details>
        </article>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert (
        "Canonical explanation."
        in text
    )

    assert (
        "Legacy Python variant"
        not in text
    )

    assert (
        "legacy_example()"
        not in text
    )


def test_extract_fastapi_article_ignores_page_navigation(
    tmp_path: Path,
) -> None:
    """
    FastAPI 页面正文之外存在大量导航。

    Loader 只能读取：
        article.md-content__inner.md-typeset

    不能把整个 main / body 的导航文字
    混入知识库。
    """

    path = _write_html(
        tmp_path,
        """
        <html>
            <body>
                <header>
                    FastAPI Cloud
                </header>

                <nav>
                    Deployment
                    Security
                    Release Notes
                </nav>

                <main>
                    <article class="md-content__inner md-typeset">
                        <h1>Dependencies</h1>
                        <p>Real article content.</p>
                    </article>
                </main>

                <footer>
                    Newsletter
                </footer>
            </body>
        </html>
        """,
    )

    text = extract_fastapi_article(
        path
    )

    assert (
        "Real article content."
        in text
    )

    assert "FastAPI Cloud" not in text
    assert "Deployment" not in text
    assert "Release Notes" not in text
    assert "Newsletter" not in text


def test_extract_fastapi_article_missing_container_raises(
    tmp_path: Path,
) -> None:
    """
    如果 FastAPI 官网未来修改 DOM，
    Loader 不应该静默返回整页垃圾文本。

    应该 fail-fast，
    让 ingestion 构建阶段直接暴露问题。
    """

    path = _write_html(
        tmp_path,
        """
        <html>
            <body>
                <main>
                    <p>
                        This page no longer has
                        the expected article container.
                    </p>
                </main>
            </body>
        </html>
        """,
    )

    with pytest.raises(
        ValueError,
        match="未找到 FastAPI 正文容器",
    ):
        extract_fastapi_article(
            path
        )