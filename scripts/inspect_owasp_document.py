"""检查单篇 OWASP 文档的 HTML 抽取与 Section 解析结果。"""

import argparse
from pathlib import Path

from enterprise_rag.ingestion.generic_section_parser import (
    parse_generic_sections,
)
from enterprise_rag.ingestion.loaders.html_loader import (
    extract_owasp_article,
)
from enterprise_rag.ingestion.manifest import (
    DocumentManifest,
    load_manifest,
)
from enterprise_rag.ingestion.normalizer import (
    normalize_structured_text,
)


MANIFEST_PATH = Path(
    "data/manifest/documents.yaml"
)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    使用方式：

        python scripts/inspect_owasp_document.py \
            --document-id owasp_llm02_sensitive_information_disclosure_2025
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect OWASP HTML extraction "
            "and generic section parsing."
        )
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="要检查的 OWASP document_id。",
    )

    return parser.parse_args()


def find_manifest(
    *,
    manifests: list[DocumentManifest],
    document_id: str,
) -> DocumentManifest:
    """
    根据 document_id 找到对应 Manifest。

    为什么这里 fail-fast：

    如果 document_id 写错，
    不应该默默检查错误文档或者什么都不做。
    """

    for manifest in manifests:
        if manifest.document_id == document_id:
            return manifest

    raise ValueError(
        "Manifest 中不存在 document_id："
        f"{document_id}"
    )


def validate_owasp_manifest(
    manifest: DocumentManifest,
) -> None:
    """
    检查当前 Manifest 是否真的是 OWASP Security Guideline。

    这个脚本只负责 OWASP 文档检查，
    不应该被误用于法规或 FastAPI 文档。
    """

    if (
        manifest.document_type
        != "security_guideline"
    ):
        raise ValueError(
            "当前文档不是 security_guideline："
            f"{manifest.document_id} "
            f"({manifest.document_type})"
        )

    if not manifest.document_id.startswith(
        "owasp_"
    ):
        raise ValueError(
            "当前脚本只用于 OWASP 文档："
            f"{manifest.document_id}"
        )


def print_text_preview(
    text: str,
    *,
    max_chars: int = 2500,
) -> None:
    """
    打印结构化正文前若干字符。

    目的不是输出整个网页，
    而是快速人工检查：

    1. 根标题有没有；
    2. Heading 有没有保留；
    3. 正文有没有；
    4. 是否混入导航栏 / 分享按钮等噪声。
    """

    print()
    print("=" * 100)
    print("Structured Text Preview")
    print("=" * 100)

    if len(text) <= max_chars:
        print(text)
        return

    print(text[:max_chars])

    print()
    print(
        f"... preview truncated, "
        f"total chars = {len(text)}"
    )


def print_section_summary(
    sections,
) -> None:
    """
    输出 Generic Section Parser 的结果。

    我们重点检查：

        title
        level
        path
        content length

    而不是一次把全部 Section 正文打印出来。
    """

    print()
    print("=" * 100)
    print("Parsed Sections")
    print("=" * 100)

    print(
        "Section count:",
        len(sections),
    )

    print()

    for index, section in enumerate(
        sections,
        start=1,
    ):
        print(
            f"[{index:02d}] "
            f"level={section.level}"
        )

        print(
            "     title:",
            section.title,
        )

        print(
            "     path :",
            section.path,
        )

        print(
            "     chars:",
            len(section.content),
        )

        # --------------------------------------------------
        # 只打印非常短的正文预览。
        #
        # 我们只是人工判断 Section 内容有没有正常进入，
        # 不需要把整个 OWASP 页面刷满终端。
        # --------------------------------------------------

        preview = (
            section.content
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 180:
            preview = (
                preview[:180]
                + "..."
            )

        print(
            "     text :",
            preview,
        )

        print()


def main() -> None:
    """
    OWASP Compatibility Audit 主入口。

    完整链路：

        Manifest
        ↓
        Raw HTML
        ↓
        extract_owasp_article()
        ↓
        normalize_structured_text()
        ↓
        parse_generic_sections()
        ↓
        Human-readable Audit
    """

    args = parse_args()

    # ======================================================
    # 1. 找到目标 Manifest。
    # ======================================================

    manifests = load_manifest(
        MANIFEST_PATH
    )

    manifest = find_manifest(
        manifests=manifests,
        document_id=args.document_id,
    )

    validate_owasp_manifest(
        manifest
    )

    raw_path = Path(
        manifest.local_path
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw HTML 不存在，请先下载："
            f"{raw_path}"
        )

    # ======================================================
    # 2. 使用正式 OWASP Extractor。
    # ======================================================

    extracted_text = (
        extract_owasp_article(
            raw_path,
            title=manifest.title,
        )
    )

    # ======================================================
    # 3. 使用正式 Structured Normalizer。
    #
    # 这里非常重要：
    #
    # 我们不是只测试 HTML Selector，
    # 而是测试真正 Ingestion 会走过的链路。
    # ======================================================

    normalized_text = (
        normalize_structured_text(
            extracted_text
        )
    )

    # ======================================================
    # 4. 使用 Generic Section Parser。
    # ======================================================

    sections = (
        parse_generic_sections(
            normalized_text
        )
    )

    # ======================================================
    # 5. Fail-fast 基础检查。
    # ======================================================

    if not normalized_text.strip():
        raise ValueError(
            "Structured text 为空。"
        )

    if not sections:
        raise ValueError(
            "Generic Section Parser "
            "没有解析出任何 Section。"
        )

    # ======================================================
    # 6. 输出人工 Audit 信息。
    # ======================================================

    print("=" * 100)
    print("OWASP Document Compatibility Audit")
    print("=" * 100)

    print(
        "Document ID :",
        manifest.document_id,
    )

    print(
        "Title       :",
        manifest.title,
    )

    print(
        "Raw path    :",
        raw_path,
    )

    print(
        "Raw bytes   :",
        raw_path.stat().st_size,
    )

    print(
        "Text chars  :",
        len(normalized_text),
    )

    print_text_preview(
        normalized_text
    )

    print_section_summary(
        sections
    )


if __name__ == "__main__":
    main()