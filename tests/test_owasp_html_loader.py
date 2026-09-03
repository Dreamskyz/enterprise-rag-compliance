"""OWASP HTML 结构化正文抽取测试。"""

from pathlib import Path

from enterprise_rag.ingestion.document_builder import (
    build_normalized_document,
)
from enterprise_rag.ingestion.loaders.html_loader import (
    extract_owasp_article,
)
from enterprise_rag.ingestion.manifest import (
    DocumentManifest,
)


def _write_test_html(
    path: Path,
) -> None:
    """
    写入一个最小 OWASP 风格 HTML。

    测试不直接依赖真实网络页面，
    这样 pytest 才是可重复、离线可运行的。
    """

    path.write_text(
        """
        <html>
        <body>

            <main id="main">

                <div class="elementor-widget-xpro-post-title">
                    <div class="elementor-widget-container">
                        <h2 class="xpro-post-title">
                            LLM01:2025 Prompt Injection
                        </h2>
                    </div>
                </div>

                <div class="elementor-widget-xpro-post-content">
                    <div class="elementor-widget-container">

                        <div class="xpro-elementor-content">

                            <div class="markdown-heading">
                                <p>
                                    A Prompt Injection Vulnerability occurs
                                    when prompts alter model behavior.
                                </p>
                            </div>

                            <div class="markdown-heading">
                                <h3>
                                    Types of Prompt Injection Vulnerabilities
                                </h3>
                            </div>

                            <div class="markdown-heading">
                                <h4>
                                    Direct Prompt Injections
                                </h4>
                            </div>

                            <p>
                                Direct prompt injections alter
                                model behavior.
                            </p>

                            <div class="markdown-heading">
                                <h3>
                                    Prevention and Mitigation Strategies
                                </h3>
                            </div>

                            <div class="markdown-heading">
                                <h4>
                                    7. Conduct adversarial testing\\
                                </h4>
                            </div>

                            <ul>
                                <li>
                                    Perform adversarial testing.
                                </li>
                                <li>
                                    Apply least privilege.
                                </li>
                            </ul>

                            <div class="sharedaddy">
                                <h3>Share this:</h3>
                                <p>
                                    This text must not enter
                                    the knowledge base.
                                </p>
                            </div>

                        </div>
                    </div>
                </div>

                <h2>LLM Top 10</h2>
                <h2>
                    LLM02:2025 Sensitive Information Disclosure
                </h2>

            </main>

        </body>
        </html>
        """,
        encoding="utf-8",
    )


def test_extract_owasp_article_preserves_headings(
    tmp_path: Path,
) -> None:
    """OWASP Loader 应保留 H2/H3/H4 层级。"""

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    text = extract_owasp_article(
        path=html_path,
        title="LLM01:2025 Prompt Injection",
    )

    assert (
        "## LLM01:2025 Prompt Injection"
        in text
    )

    assert (
        "### Types of Prompt Injection Vulnerabilities"
        in text
    )

    assert (
        "#### Direct Prompt Injections"
        in text
    )


def test_extract_owasp_article_preserves_body(
    tmp_path: Path,
) -> None:
    """正文段落不能因为结构转换而丢失。"""

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    text = extract_owasp_article(
        path=html_path,
        title="LLM01:2025 Prompt Injection",
    )

    assert (
        "A Prompt Injection Vulnerability occurs"
        in text
    )

    assert (
        "Direct prompt injections alter"
        in text
    )


def test_extract_owasp_article_preserves_lists(
    tmp_path: Path,
) -> None:
    """列表内容应该进入结构化正文。"""

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    text = extract_owasp_article(
        path=html_path,
        title="LLM01:2025 Prompt Injection",
    )

    assert (
        "- Perform adversarial testing."
        in text
    )

    assert (
        "- Apply least privilege."
        in text
    )


def test_extract_owasp_article_removes_page_noise(
    tmp_path: Path,
) -> None:
    """
    社交分享和其他 LLM Top 10 页面区域
    不应该进入正文。
    """

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    text = extract_owasp_article(
        path=html_path,
        title="LLM01:2025 Prompt Injection",
    )

    assert "Share this:" not in text

    assert (
        "This text must not enter"
        not in text
    )

    assert "LLM Top 10" not in text

    assert (
        "LLM02:2025 Sensitive Information Disclosure"
        not in text
    )


def test_extract_owasp_article_cleans_heading_backslash(
    tmp_path: Path,
) -> None:
    """标题末尾 Markdown 残留反斜杠应被清理。"""

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    text = extract_owasp_article(
        path=html_path,
        title="LLM01:2025 Prompt Injection",
    )

    assert (
        "#### 7. Conduct adversarial testing"
        in text
    )

    assert (
        "#### 7. Conduct adversarial testing\\"
        not in text
    )


def test_extract_owasp_article_rejects_missing_container(
    tmp_path: Path,
) -> None:
    """
    如果 OWASP 修改页面模板，
    Loader 应明确失败，而不是悄悄生成空知识库。
    """

    html_path = (
        tmp_path / "broken.html"
    )

    html_path.write_text(
        "<html><body><p>No article.</p></body></html>",
        encoding="utf-8",
    )

    try:
        extract_owasp_article(
            path=html_path,
            title="LLM01:2025 Prompt Injection",
        )
    except ValueError as exc:
        assert (
            "未找到 OWASP 正文容器"
            in str(exc)
        )
    else:
        raise AssertionError(
            "缺少 OWASP 正文容器时应抛出 ValueError"
        )


def test_document_builder_supports_security_guideline(
    tmp_path: Path,
) -> None:
    """
    Document Builder 应能把 security_guideline
    路由到 OWASP Loader。
    """

    html_path = (
        tmp_path / "owasp_llm01.html"
    )

    _write_test_html(
        html_path
    )

    manifest = DocumentManifest(
        document_id=(
            "owasp_llm01_prompt_injection_2025"
        ),
        title="LLM01:2025 Prompt Injection",
        source_url=(
            "https://genai.owasp.org/"
            "llmrisk/llm01-prompt-injection/"
        ),
        source_type="official",
        document_type="security_guideline",
        language="en",
        version="2025",
        published_at=None,
        effective_at=None,
        access_level="public",
        local_path=str(html_path),
        enabled=True,
    )

    document = build_normalized_document(
        manifest
    )

    assert (
        document.document_id
        == "owasp_llm01_prompt_injection_2025"
    )

    assert (
        document.document_type
        == "security_guideline"
    )

    assert document.language == "en"

    assert (
        "## LLM01:2025 Prompt Injection"
        in document.text
    )

    assert (
        "### Types of Prompt Injection Vulnerabilities"
        in document.text
    )

    assert (
        "#### Direct Prompt Injections"
        in document.text
    )