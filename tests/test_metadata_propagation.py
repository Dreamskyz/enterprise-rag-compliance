"""测试异构文档 Metadata 在系统各层之间的传播。"""

import json
from pathlib import Path

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
    write_chunks_jsonl,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)
from enterprise_rag.retrieval.dense import (
    DenseRetriever,
)
from enterprise_rag.vectorstore.qdrant_store import (
    QdrantVectorStore,
)


def _build_technical_chunk() -> KnowledgeChunk:
    """
    构造一个最小技术文档 Chunk。

    它故意没有法规 Chapter / Article，
    用于测试 Day 6 新增的 Section Metadata。
    """

    return KnowledgeChunk(
        chunk_id=(
            "fastapi_dependencies__"
            "classes_as_dependencies__0001"
        ),

        document_id="fastapi_dependencies",

        title="FastAPI Dependencies",

        document_type=(
            "technical_documentation"
        ),

        language="en",

        version="current",

        # 技术文档没有法规结构。
        chapter_number=None,
        chapter_title=None,
        article_number=None,

        content=(
            "A Python class can be used "
            "as a dependency."
        ),

        retrieval_text=(
            "FastAPI Dependencies\n"
            "Tutorial > Dependencies > "
            "Classes as Dependencies\n"
            "A Python class can be used "
            "as a dependency."
        ),

        source_url=(
            "https://fastapi.tiangolo.com/"
        ),

        access_level="public",

        chunk_index=0,

        content_hash="technical-test-hash",

        section_title=(
            "Classes as Dependencies"
        ),

        section_path=(
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        ),
    )


def test_chunk_jsonl_round_trip_preserves_section_metadata(
    tmp_path: Path,
) -> None:
    """
    KnowledgeChunk → JSONL → KnowledgeChunk

    新的 Section Metadata 不应该丢失。
    """

    chunk = _build_technical_chunk()

    output_path = (
        tmp_path / "chunks.jsonl"
    )

    write_chunks_jsonl(
        [chunk],
        output_path,
    )

    loaded_chunks = read_chunks_jsonl(
        output_path
    )

    assert len(loaded_chunks) == 1

    loaded = loaded_chunks[0]

    assert (
        loaded.section_title
        == "Classes as Dependencies"
    )

    assert (
        loaded.section_path
        == (
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        )
    )

    assert loaded.chapter_number is None
    assert loaded.chapter_title is None
    assert loaded.article_number is None


def test_old_jsonl_without_section_fields_is_backward_compatible(
    tmp_path: Path,
) -> None:
    """
    模拟 Day 6 之前生成的旧 chunks.jsonl。

    旧 JSON 中没有：

        section_title
        section_path

    Reader 仍然应该能够正常恢复，
    并让新增字段使用 dataclass 默认值 None。
    """

    old_data = {
        "chunk_id": (
            "cn_genai_interim_2023__第七条"
        ),

        "document_id": (
            "cn_genai_interim_2023"
        ),

        "title": (
            "生成式人工智能服务管理暂行办法"
        ),

        "document_type": "regulation",

        "language": "zh",

        "version": "2023",

        "chapter_number": "第二章",

        "chapter_title": (
            "技术发展与治理"
        ),

        "article_number": "第七条",

        "content": (
            "生成式人工智能服务提供者"
            "开展训练数据处理活动时，"
            "应当遵守有关规定。"
        ),

        "retrieval_text": (
            "生成式人工智能服务管理暂行办法\n"
            "第二章 技术发展与治理\n"
            "第七条\n"
            "生成式人工智能服务提供者"
            "开展训练数据处理活动时，"
            "应当遵守有关规定。"
        ),

        "source_url": (
            "https://example.com/regulation"
        ),

        "access_level": "public",

        "chunk_index": 0,

        "content_hash": "legacy-hash",
    }

    input_path = (
        tmp_path / "legacy_chunks.jsonl"
    )

    input_path.write_text(
        json.dumps(
            old_data,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded_chunks = read_chunks_jsonl(
        input_path
    )

    assert len(loaded_chunks) == 1

    loaded = loaded_chunks[0]

    assert (
        loaded.article_number
        == "第七条"
    )

    assert loaded.section_title is None
    assert loaded.section_path is None


def test_qdrant_payload_preserves_section_metadata() -> None:
    """
    KnowledgeChunk → Qdrant Payload

    Section Metadata 必须真正进入 Vector Store。
    """

    chunk = _build_technical_chunk()

    payload = (
        QdrantVectorStore._build_payload(
            chunk
        )
    )

    assert (
        payload["section_title"]
        == "Classes as Dependencies"
    )

    assert (
        payload["section_path"]
        == (
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        )
    )

    # 技术文档法规字段应保持真正的 None。
    assert payload["chapter_number"] is None
    assert payload["chapter_title"] is None
    assert payload["article_number"] is None


def test_dense_payload_conversion_preserves_none_and_section_metadata() -> None:
    """
    Qdrant Payload → RetrievalCandidate

    这是本次最重要的回归测试之一：

        None

    绝对不能因为 str(None)
    变成：

        "None"
    """

    chunk = _build_technical_chunk()

    payload = (
        QdrantVectorStore._build_payload(
            chunk
        )
    )

    candidate = (
        DenseRetriever
        ._build_candidate_from_payload(
            payload
        )
    )

    assert candidate.chapter_number is None
    assert candidate.chapter_title is None
    assert candidate.article_number is None

    assert (
        candidate.section_title
        == "Classes as Dependencies"
    )

    assert (
        candidate.section_path
        == (
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        )
    )

    # 显式保护：
    # 不允许产生字符串形式的 "None"。
    assert candidate.chapter_number != "None"
    assert candidate.chapter_title != "None"
    assert candidate.article_number != "None"


def test_dense_payload_conversion_supports_legacy_payload() -> None:
    """
    模拟 Day 6 之前的旧 Qdrant Payload。

    旧 Payload 不存在：

        section_title
        section_path

    Dense Retriever 必须仍然可以恢复 Candidate。
    """

    payload: dict[str, object] = {
        "chunk_id": (
            "cn_genai_interim_2023__第七条"
        ),

        "document_id": (
            "cn_genai_interim_2023"
        ),

        "title": (
            "生成式人工智能服务管理暂行办法"
        ),

        "document_type": "regulation",

        "language": "zh",

        "version": "2023",

        "chapter_number": "第二章",

        "chapter_title": (
            "技术发展与治理"
        ),

        "article_number": "第七条",

        "content": (
            "生成式人工智能服务提供者"
            "应当遵守有关规定。"
        ),

        "retrieval_text": (
            "生成式人工智能服务管理暂行办法 "
            "第二章 第七条 "
            "生成式人工智能服务提供者"
            "应当遵守有关规定。"
        ),

        "source_url": (
            "https://example.com/regulation"
        ),

        "access_level": "public",

        "chunk_index": 0,

        "content_hash": "legacy-payload-hash",
    }

    candidate = (
        DenseRetriever
        ._build_candidate_from_payload(
            payload
        )
    )

    assert (
        candidate.article_number
        == "第七条"
    )

    assert candidate.section_title is None
    assert candidate.section_path is None


def test_bm25_preserves_section_metadata() -> None:
    """
    KnowledgeChunk → BM25 → RetrievalCandidate

    BM25 不经过 Qdrant，
    但同样不能把 Section Metadata 丢掉。
    """

    chunk = _build_technical_chunk()

    retriever = BM25Retriever(
        chunks=[chunk]
    )

    results = retriever.search(
        query="FastAPI Dependencies",
        top_k=1,
        role=UserRole.GUEST,
    )

    assert len(results) == 1

    candidate = results[0].candidate

    assert candidate.chapter_number is None
    assert candidate.chapter_title is None
    assert candidate.article_number is None

    assert (
        candidate.section_title
        == "Classes as Dependencies"
    )

    assert (
        candidate.section_path
        == (
            "Tutorial > Dependencies > "
            "Classes as Dependencies"
        )
    )