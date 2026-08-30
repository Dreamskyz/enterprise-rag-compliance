"""检查法规章节和条款解析结果。"""

from pathlib import Path

from enterprise_rag.ingestion.loaders.html_loader import (
    extract_cac_article,
)
from enterprise_rag.ingestion.normalizer import normalize_text
from enterprise_rag.ingestion.regulation_parser import (
    parse_regulation,
)


def main() -> None:
    path = Path(
        #"data/raw/cn_genai_interim_2023.html"
        "data/raw/cn_deep_synthesis_2022.html"
    )

    article_text = extract_cac_article(path)

    normalized_text = normalize_text(article_text)

    chapters = parse_regulation(normalized_text)

    article_count = sum(
        len(chapter.articles)
        for chapter in chapters
    )

    print("=" * 80)
    print("章节数量：", len(chapters))
    print("条款数量：", article_count)
    print("=" * 80)

    for chapter in chapters:
        print()
        print(
            chapter.chapter_number,
            chapter.title,
        )

        for article in chapter.articles:
            print(
                "  ",
                article.article_number,
            )

    #print()
    #print("=" * 80)
    #print("检查第七条")
    #print("=" * 80)

    #for chapter in chapters:
        #for article in chapter.articles:
            #if article.article_number == "第七条":
                #print(article.content)

if __name__ == "__main__":
    main()

