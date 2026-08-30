"""检查 HTML 文档抽取和标准化结果。"""

from pathlib import Path

from enterprise_rag.ingestion.loaders.html_loader import extract_cac_article
from enterprise_rag.ingestion.normalizer import normalize_text


def main() -> None:
    path = Path(
        #"data/raw/cn_genai_interim_2023.html"
        "data/raw/cn_deep_synthesis_2022.html"
    )

    article_text = extract_cac_article(path)

    normalized_text = normalize_text(article_text)

    print("=" * 80)
    print("标准化后文本长度：", len(normalized_text))
    print("=" * 80)

    print(normalized_text[:5000])


if __name__ == "__main__":
    main()