"""法规文档 Chunk 构建模块。"""

import hashlib          #hashlib 是 Python 标准库，专门用来做哈希计算  SHA-256 不是加密正文，而是在给正文生成“指纹”

from enterprise_rag.ingestion.models import (           #导入数据模型
    KnowledgeChunk,
    NormalizedDocument,
    RegulationChapter,
)


def build_retrieval_text(           #负责给一个法规条文构造“用于检索的完整文本”
    document: NormalizedDocument,
    chapter: RegulationChapter,
    article_number: str,
    content: str,
) -> str:
    """
    构建统一的检索文本。

    后续 Dense / BM25 / Reranker
    原则上都使用这一份文本表示。
    """

    return "\n".join(           #使用"\n".join(...)把多个字符串用换行符连接起来
        [
            document.title,
            f"{chapter.chapter_number} {chapter.title}".strip(),
            article_number,
            content,
        ]
    )


def build_content_hash(content: str) -> str:
    """
    为 Chunk 正文计算 SHA-256。

    作用：
    后续可判断同一 chunk_id 的正文是否发生变化。
    """

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def build_regulation_chunks(
    document: NormalizedDocument,
    chapters: list[RegulationChapter],
) -> list[KnowledgeChunk]:
    """
    将法规结构转换为标准 KnowledgeChunk。

    当前 V1：
        一条法规 = 一个 Chunk。
    """

    chunks: list[KnowledgeChunk] = []       #初始化结果列表

    chunk_index = 0

    for chapter in chapters:
        for article in chapter.articles:
            chunk_id = (                        #构建 chunk_id
                f"{document.document_id}"
                f"__{article.article_number}"
            )

            retrieval_text = build_retrieval_text(
                document=document,
                chapter=chapter,
                article_number=article.article_number,
                content=article.content,
            )

            content_hash = build_content_hash(
                article.content
            )

            chunk = KnowledgeChunk(
                chunk_id=chunk_id,

                document_id=document.document_id,
                title=document.title,

                document_type=document.document_type,
                language=document.language,
                version=document.version,

                chapter_number=chapter.chapter_number,
                chapter_title=chapter.title,
                article_number=article.article_number,

                content=article.content,
                retrieval_text=retrieval_text,

                source_url=document.source_url,
                access_level=document.access_level,

                chunk_index=chunk_index,
                content_hash=content_hash,
            )

            chunks.append(chunk)

            chunk_index += 1

    return chunks