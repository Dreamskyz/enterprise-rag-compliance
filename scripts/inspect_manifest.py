"""检查 Document Manifest 是否能够正常加载。"""

from pathlib import Path

from enterprise_rag.ingestion.manifest import load_manifest


def main() -> None:
    manifest_path = Path(
        "data/manifest/documents.yaml"
    )

    documents = load_manifest(manifest_path)

    print(f"文档数量：{len(documents)}")

    for document in documents:
        print("-" * 80)
        print("document_id:", document.document_id)
        print("title:", document.title)
        print("access_level:", document.access_level)
        print("local_path:", document.local_path)


if __name__ == "__main__":
    main()