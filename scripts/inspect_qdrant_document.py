"""检查 Qdrant 官方文档的 ingestion 兼容性。"""

import argparse
from pathlib import Path

from enterprise_rag.ingestion.document_builder import (
    build_normalized_document,
)
from enterprise_rag.ingestion.generic_section_parser import (
    parse_generic_sections,
)
from enterprise_rag.ingestion.manifest import (
    DocumentManifest,
    load_manifest,
)


MANIFEST_PATH = Path(
    "data/manifest/documents.yaml"
)

PREVIEW_CHAR_LIMIT = 5000

SECTION_TEXT_PREVIEW_LIMIT = 180


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    使用方式：

        python scripts/inspect_qdrant_document.py \
            --document-id qdrant_payload
    """

    parser = argparse.ArgumentParser(
        description=(
            "检查一篇 Qdrant 官方技术文档 "
            "从 Raw HTML 到 Generic Section "
            "的完整 ingestion 结果。"
        )
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help=(
            "Manifest 中的 Qdrant document_id，"
            "例如 qdrant_payload"
        ),
    )

    return parser.parse_args()


def find_manifest(
    manifests: list[DocumentManifest],
    document_id: str,
) -> DocumentManifest:
    """
    根据 document_id 找到对应 Manifest。

    如果不存在，则直接 fail-fast。

    这样可以避免因为 ID 拼写错误，
    最后得到难以理解的 FileNotFoundError。
    """

    for manifest in manifests:
        if manifest.document_id == document_id:
            return manifest

    raise ValueError(
        "Manifest 中不存在 document_id："
        f"{document_id}"
    )


def validate_qdrant_manifest(
    manifest: DocumentManifest,
) -> None:
    """
    检查用户选择的文档是否确实属于
    当前 Qdrant ingestion 范围。

    当前 V1 使用 document_id 前缀：

        qdrant_*

    路由到 Qdrant Loader。

    所以 Inspector 也沿用同一约定。
    """

    if not manifest.document_id.startswith(
        "qdrant_"
    ):
        raise ValueError(
            "该 Inspector 只用于 Qdrant 文档："
            f"{manifest.document_id}"
        )

    if (
        manifest.document_type
        != "technical_documentation"
    ):
        raise ValueError(
            "Qdrant 文档应使用 "
            "document_type="
            "technical_documentation，"
            f"当前为：{manifest.document_type}"
        )


def shorten_text(
    text: str,
    limit: int,
) -> str:
    """
    把过长正文压缩成终端友好的 Preview。

    只用于检查输出，
    不会修改真实 Corpus。
    """

    cleaned = " ".join(
        text.split()
    )

    if len(cleaned) <= limit:
        return cleaned

    return (
        cleaned[:limit]
        + "..."
    )


def print_structured_text_preview(
    text: str,
) -> None:
    """
    打印 NormalizedDocument 中
    Markdown-like Structured Text 的前一部分。

    这里主要人工观察：

    - # / ## / ### 层级是否正常；
    - [NOTE] 是否被保留；
    - [CODE] 是否存在；
    - 是否混入导航栏等网页噪声。
    """

    print()
    print("=" * 100)
    print("Structured Text Preview")
    print("=" * 100)
    print()

    if len(text) <= PREVIEW_CHAR_LIMIT:
        print(text)
        return

    print(
        text[:PREVIEW_CHAR_LIMIT]
    )

    print()
    print(
        "... preview truncated, "
        f"total chars = {len(text)}"
    )


def print_sections(
    text: str,
) -> None:
    """
    使用生产环境相同的 Generic Section Parser
    解析文档，并打印 Section 结构。

    这一步非常关键。

    Loader 输出“看起来正常”并不代表
    Parser 一定能正确恢复结构。

    因此检查链路必须是：

        Raw HTML
            ↓
        Site-specific Loader
            ↓
        Structured Normalizer
            ↓
        Generic Section Parser
    """

    sections = parse_generic_sections(
        text
    )

    print()
    print("=" * 100)
    print("Parsed Sections")
    print("=" * 100)
    print()

    print(
        f"Section count: {len(sections)}"
    )

    for index, section in enumerate(
        sections,
        start=1,
    ):
        print()

        print(
            f"[{index:02d}] "
            f"level={section.level}"
        )

        print(
            f"title: {section.title}"
        )

        print(
            f"path : {section.path}"
        )

        print(
            f"chars: {len(section.content)}"
        )

        print(
            "text : "
            + shorten_text(
                section.content,
                SECTION_TEXT_PREVIEW_LIMIT,
            )
        )


def print_qdrant_signals(
    text: str,
) -> None:
    """
    打印一些 Qdrant-specific 信号。

    这些不是正式自动测试，
    而是帮助我们快速发现 Corpus 清洗异常。

    重点检查：

    1. [NOTE]：
       aside 是否成功进入正文。

    2. Python：
       Python SDK 示例是否仍然存在。

    3. TypeScript / Rust / Java：
       是否存在明显的被过滤语言残留。

    注意：

    一个页面没有 [NOTE] 本身不是错误。

    因为不是所有 Qdrant 页面都一定包含 aside。
    """

    checks = {
        "NOTE blocks": (
            text.count("[NOTE]")
        ),
        "CODE blocks": (
            text.count("[CODE]")
        ),
        "Python QdrantClient": (
            "QdrantClient" in text
        ),
        "TypeScript marker": (
            "@qdrant/js-client-rest"
            in text
        ),
        "Rust marker": (
            "qdrant_client::"
            in text
        ),
        "Java marker": (
            ".newBuilder("
            in text
        ),
    }

    print()
    print("=" * 100)
    print("Qdrant-specific Signals")
    print("=" * 100)
    print()

    for name, value in checks.items():
        print(
            f"{name:<24}: {value}"
        )


def print_noise_signals(
    text: str,
) -> None:
    """
    粗粒度检查明显网页 UI 噪声。

    这些字符串来自 Qdrant 页面外围 UI，
    正常情况下不应该进入：

        article.documentation-article

    如果出现，
    说明正文 selector 可能失效，
    或网站 DOM 结构已经改变。

    这里只做辅助观察，
    不能把它当成完整 HTML 清洗测试。
    """

    noise_markers = [
        "Start Free",
        "Log in",
        "Getting Started",
        "User Manual",
        "Qdrant Tools",
        "Tutorials",
        "Support",
    ]

    detected = [
        marker
        for marker in noise_markers
        if marker in text
    ]

    print()
    print("=" * 100)
    print("Potential Page Noise")
    print("=" * 100)
    print()

    if not detected:
        print(
            "No obvious navigation noise detected."
        )
        return

    print(
        "Detected markers:"
    )

    for marker in detected:
        print(
            f"- {marker}"
        )


def main() -> None:
    """
    执行 Qdrant 文档兼容性检查。
    """

    args = parse_args()

    # --------------------------------------------------
    # 1. 从唯一事实源 Manifest 加载文档配置。
    # --------------------------------------------------
    manifests = load_manifest(
        MANIFEST_PATH
    )

    manifest = find_manifest(
        manifests=manifests,
        document_id=args.document_id,
    )

    validate_qdrant_manifest(
        manifest
    )

    raw_path = Path(
        manifest.local_path
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            "Qdrant Raw HTML 不存在："
            f"{raw_path}"
        )

    # --------------------------------------------------
    # 2. 直接调用生产环境的 Document Builder。
    #
    # Inspector 不重新实现 ingestion 逻辑，
    # 而是复用真正的生产链路：
    #
    # Manifest
    #    ↓
    # Qdrant Loader
    #    ↓
    # Structured Normalizer
    #    ↓
    # NormalizedDocument
    #
    # 这样 Inspector 检查的就是
    # “系统真正会入库的结果”。
    # --------------------------------------------------
    document = build_normalized_document(
        manifest
    )

    raw_bytes = raw_path.stat().st_size

    print(
        "# Qdrant Document "
        "Compatibility Audit"
    )

    print()

    print(
        f"Document ID : "
        f"{manifest.document_id}"
    )

    print(
        f"Title       : "
        f"{manifest.title}"
    )

    print(
        f"Raw path    : "
        f"{raw_path}"
    )

    print(
        f"Raw bytes   : "
        f"{raw_bytes}"
    )

    print(
        f"Text chars  : "
        f"{len(document.text)}"
    )

    # --------------------------------------------------
    # 3. Structured Text Preview。
    # --------------------------------------------------
    print_structured_text_preview(
        document.text
    )

    # --------------------------------------------------
    # 4. Generic Parser 结果。
    # --------------------------------------------------
    print_sections(
        document.text
    )

    # --------------------------------------------------
    # 5. Qdrant-specific 清洗检查。
    # --------------------------------------------------
    print_qdrant_signals(
        document.text
    )

    # --------------------------------------------------
    # 6. 粗粒度网页噪声检查。
    # --------------------------------------------------
    print_noise_signals(
        document.text
    )


if __name__ == "__main__":
    main()