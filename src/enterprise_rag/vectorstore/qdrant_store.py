"""Qdrant 向量存储封装。"""

from collections.abc import Sequence
from uuid import UUID, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)


COLLECTION_NAME = "compliance_chunks_v1"

VECTOR_SIZE = 1024

QDRANT_URL = "http://localhost:6333"


# 固定 UUID Namespace。
#
# 同一个 chunk_id 始终会生成同一个 UUID，
# 从而保证重复建库时 Point ID 稳定。
POINT_ID_NAMESPACE = UUID(
    "1db9d177-09f0-4b93-9f7c-4ff8c51b5f22"
)


class QdrantVectorStore:
    """
    Qdrant Dense Vector Store。

    当前职责：

    1. 连接 Qdrant；
    2. 创建 Collection；
    3. 将 KnowledgeChunk + Dense Vector 写入 Qdrant。
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(
            url=url,
        )

    def create_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """
        创建 Dense Vector Collection。

        参数：
            recreate:
                True 时删除旧 Collection 后重新创建。
                仅用于开发阶段明确需要重建索引时。
        """

        exists = self.client.collection_exists(
            self.collection_name
        )

        if exists:
            if not recreate:
                return

            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _build_payload(
        chunk: KnowledgeChunk,
    ) -> dict[str, object]:
        """
        将 KnowledgeChunk 转换成 Qdrant Payload。

        Payload 中保留后续检索、Citation、
        ACL 和 Debug 所需要的字段。
        """

        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,

            "document_type": chunk.document_type,
            "language": chunk.language,
            "version": chunk.version,

            "chapter_number": chunk.chapter_number,
            "chapter_title": chunk.chapter_title,
            "article_number": chunk.article_number,

            "content": chunk.content,
            "retrieval_text": chunk.retrieval_text,

            "source_url": chunk.source_url,
            "access_level": chunk.access_level,

            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
        }

    @staticmethod
    def _build_point_id(
        chunk_id: str,
    ) -> str:
        """
        根据业务 chunk_id 生成稳定 UUID。

        同一个 chunk_id 在重复建库时
        会得到相同的 Qdrant Point ID。
        """

        return str(
            uuid5(
                POINT_ID_NAMESPACE,
                chunk_id,
            )
        )

    def upsert_chunks(
        self,
        chunks: Sequence[KnowledgeChunk],
        vectors: np.ndarray,
    ) -> None:
        """
        将 Chunk 和对应 Dense Vector 写入 Qdrant。

        要求：
            chunks[i]
        必须对应：
            vectors[i]
        """

        if len(chunks) != len(vectors):
            raise ValueError(
                "Chunk 数量与 Vector 数量不一致："
                f"{len(chunks)} != {len(vectors)}"
            )

        if vectors.ndim != 2:
            raise ValueError(
                "vectors 必须是二维数组，"
                f"当前 shape={vectors.shape}"
            )

        if vectors.shape[1] != self.vector_size:
            raise ValueError(
                "Vector 维度与 Collection Schema 不一致："
                f"{vectors.shape[1]} != "
                f"{self.vector_size}"
            )

        points: list[PointStruct] = []

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            point = PointStruct(
                id=self._build_point_id(
                    chunk.chunk_id
                ),

                # qdrant-client 最稳妥的输入形式是普通 list。
                vector=vector.astype(
                    np.float32
                ).tolist(),

                payload=self._build_payload(
                    chunk
                ),
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def get_collection_count(
        self,
    ) -> int:
        """
        返回当前 Collection 中的 Point 数量。
        """

        info = self.client.get_collection(
            self.collection_name
        )

        return int(
            info.points_count or 0
        )