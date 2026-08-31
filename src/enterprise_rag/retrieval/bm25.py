"""基于 Jieba + BM25 的中文 Sparse Retriever。"""

from collections.abc import Sequence

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.retrieval.models import (
    BM25SearchResult,
    RetrievalCandidate,
)


class BM25Retriever:
    """
    中文 BM25 Sparse Retriever。
    """

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "BM25 Retriever 至少需要一个 Chunk"
            )

        self.chunks = list(chunks)

        self.tokenized_corpus = [
            self._tokenize(
                chunk.retrieval_text
            )
            for chunk in self.chunks
        ]

        if not any(
            self.tokenized_corpus
        ):
            raise RuntimeError(
                "BM25 语料分词结果为空"
            )

        self._bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        使用 Jieba 精确模式分词。
        """

        if not text.strip():
            return []

        tokens = jieba.lcut(
            text,
            cut_all=False,
        )

        return [                    #去除分词结果中的空白 Token
            token.strip()
            for token in tokens
            if token.strip()
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25SearchResult]:
        """
        执行 BM25 Sparse Retrieval。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(
            query_tokens
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        top_indices = scores.argsort()[
            ::-1
        ][:top_k]

        results: list[
            BM25SearchResult
        ] = []

        for index in top_indices:
            chunk_index = int(index)

            chunk = self.chunks[
                chunk_index
            ]

            candidate = RetrievalCandidate(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                document_type=chunk.document_type,
                language=chunk.language,
                version=chunk.version,
                chapter_number=chunk.chapter_number,
                chapter_title=chunk.chapter_title,
                article_number=chunk.article_number,
                content=chunk.content,
                retrieval_text=chunk.retrieval_text,
                source_url=chunk.source_url,
                access_level=chunk.access_level,
                chunk_index=chunk.chunk_index,
                content_hash=chunk.content_hash,
            )

            results.append(
                BM25SearchResult(
                    candidate=candidate,
                    score=float(
                        scores[chunk_index]
                    ),
                )
            )

        return results