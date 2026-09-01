"""测试 RAG Runtime Builder 的轻量错误处理。"""

from pathlib import Path

import pytest

from enterprise_rag.runtime.builder import (
    build_rag_runtime,
)


def test_build_rag_runtime_rejects_missing_chunks(
    tmp_path: Path,
) -> None:
    """
    Chunk 文件不存在时，
    应在加载模型之前快速失败。
    """

    missing_path = (
        tmp_path
        / "missing_chunks.jsonl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Chunk 文件不存在",
    ):
        build_rag_runtime(
            chunks_path=missing_path
        )