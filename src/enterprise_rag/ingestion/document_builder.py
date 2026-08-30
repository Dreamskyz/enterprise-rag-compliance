"""构建标准化文档对象。"""

from pathlib import Path

from enterprise_rag.ingestion.loaders.html_loader import (
    extract_cac_article,
)
from enterprise_rag.ingestion.manifest import DocumentManifest
from enterprise_rag.ingestion.models import NormalizedDocument
from enterprise_rag.ingestion.normalizer import normalize_text


def build_normalized_document(
    manifest: DocumentManifest,
) -> NormalizedDocument:
    """
    根据 Manifest 加载原始文档，
    并构建标准化文档对象。

    当前 V1 先支持国家网信办 HTML。
    """

    path = Path(manifest.local_path)

    article_text = extract_cac_article(path)

    normalized_text = normalize_text(
        article_text
    )

    return NormalizedDocument(
        document_id=manifest.document_id,
        title=manifest.title,
        text=normalized_text,
        source_url=manifest.source_url,
        access_level=manifest.access_level,
    )