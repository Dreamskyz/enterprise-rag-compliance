"""基于 Jieba + BM25 的 ACL-aware 中文 Sparse Retriever。"""

from collections.abc import Sequence
from dataclasses import dataclass

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.acl.policy import (
    can_access,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.retrieval.models import (
    BM25SearchResult,
    RetrievalCandidate,
)


@dataclass
class _RoleBM25Index:
    """
    单个角色对应的 BM25 检索索引。

    chunks:
        当前角色真正有权访问的 Chunk。

    bm25:
        基于这些授权 Chunk 构建的 BM25Okapi 索引。

    该类型只在 bm25.py 内部使用，
    不属于对外业务接口。
    """

    chunks: list[KnowledgeChunk]
    bm25: BM25Okapi


class BM25Retriever:
    """
    ACL-aware 中文 BM25 Sparse Retriever。

    当前职责：

    1. 根据角色提前划分授权语料；
    2. 为 guest / developer / admin
       分别建立 BM25 Index；
    3. Query 只进入当前角色对应的授权索引；
    4. 返回 BM25 Top-K。

    ACL 原则：

        unauthorized Chunk
            ↓
        不进入当前角色 BM25 corpus
            ↓
        不参与 BM25 candidate generation
    """

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
    ) -> None:
        """
        初始化 BM25 Retriever。

        参数：
            chunks:
                全部 KnowledgeChunk。

        初始化时会根据 ACL Policy
        为每种 UserRole 建立独立 BM25 索引。
        """

        if not chunks:
            raise ValueError(
                "BM25 Retriever 至少需要一个 Chunk"
            )

        # 固定全量 Chunk 输入。
        self.chunks = list(chunks)

        # --------------------------------------------------
        # 为每种角色构建独立的授权 BM25 Index。
        # --------------------------------------------------

        self._role_indexes: dict[
            UserRole,
            _RoleBM25Index,
        ] = {}

        for role in UserRole:
            authorized_chunks = [
                chunk
                for chunk in self.chunks
                if can_access(
                    role,
                    chunk.access_level,
                )
            ]

            # 某些未来角色理论上可能没有任何可访问数据。
            #
            # 当前 guest 至少应该能看到 public，
            # 但这里仍然安全处理空语料。
            if not authorized_chunks:
                continue

            tokenized_corpus = [
                self._tokenize(
                    chunk.retrieval_text
                )
                for chunk in authorized_chunks
            ]

            if not any(
                tokenized_corpus
            ):
                raise RuntimeError(
                    "BM25 授权语料分词结果为空："
                    f"role={role.value}"
                )

            self._role_indexes[
                role
            ] = _RoleBM25Index(
                chunks=authorized_chunks,
                bm25=BM25Okapi(
                    tokenized_corpus
                ),
            )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        使用 Jieba 精确模式对中文文本分词。

        当前 V1：

        - 不使用复杂停用词表；
        - 不使用自定义词典；
        - 不修改 BM25 k1 / b；

        保留可解释 baseline。
        """

        if not text.strip():
            return []

        tokens = jieba.lcut(
            text,
            cut_all=False,
        )

        return [
            token.strip()
            for token in tokens
            if token.strip()
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        role: UserRole = UserRole.GUEST,
    ) -> list[BM25SearchResult]:
        """
        执行 ACL-aware BM25 Retrieval。

        参数：
            query:
                用户自然语言问题。

            top_k:
                最多返回的 Candidate 数量。

            role:
                当前用户角色。

                默认：
                    guest

                即默认使用最小权限：
                    public only

        返回：
            仅来自当前角色授权语料的
            BM25SearchResult。
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

        # --------------------------------------------------
        # 1. 选择当前角色专属的授权 BM25 Index。
        # --------------------------------------------------

        role_index = self._role_indexes.get(
            role
        )

        # 当前角色没有任何可访问语料时，
        # 直接返回空结果。
        if role_index is None:
            return []

        # --------------------------------------------------
        # 2. 只在授权 corpus 内进行 BM25 打分。
        # --------------------------------------------------

        scores = role_index.bm25.get_scores(
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

        # --------------------------------------------------
        # 3. 授权 corpus index → Candidate
        # --------------------------------------------------

        for index in top_indices:
            authorized_index = int(
                index
            )

            chunk = (
                role_index.chunks[
                    authorized_index
                ]
            )

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
                        scores[
                            authorized_index
                        ]
                    ),
                )
            )

        return results