"""检查真实 KnowledgeChunk 的 Token 长度。"""

from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)


MODEL_NAME = "BAAI/bge-m3"
MAX_LENGTH = 512


def main() -> None:
    """
    使用 BGE-M3 Tokenizer，
    统计真实 retrieval_text 的 Token 长度。
    """

    chunks_path = Path(
        "data/processed/chunks.jsonl"
    )

    chunks = read_chunks_jsonl(
        chunks_path
    )

    if not chunks:
        raise RuntimeError(
            "chunks.jsonl 中没有 Chunk"
        )

    print("=" * 80)
    print("Real Chunk Token Analysis")
    print("=" * 80)

    print(
        "Chunk count:",
        len(chunks),
    )

    # 当前只是为了获得 BGE-M3 自己的 tokenizer。
    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=True,
        devices=["cuda:0"],
    )

    tokenizer = model.tokenizer

    lengths: list[int] = []

    for chunk in chunks:
        encoded = tokenizer(
            chunk.retrieval_text,

            # 把模型实际需要的特殊 Token 也算进去。
            add_special_tokens=True,

            # 这里是做长度分析，
            # 所以绝不能先截断。
            truncation=False,
        )

        token_count = len(
            encoded["input_ids"]
        )

        lengths.append(
            token_count
        )

    print()
    print(
        "Min tokens:",
        min(lengths),
    )

    print(
        "Max tokens:",
        max(lengths),
    )

    print(
        "Average tokens:",
        round(
            sum(lengths)
            / len(lengths),
            2,
        ),
    )

    # 找 Token 最长的 Chunk。
    longest_index = max(
        range(len(lengths)),
        key=lengths.__getitem__,
    )

    longest_chunk = chunks[
        longest_index
    ]

    print()
    print("Longest chunk:")

    print(
        "chunk_id:",
        longest_chunk.chunk_id,
    )

    print(
        "tokens:",
        lengths[longest_index],
    )

    print(
        "article:",
        longest_chunk.article_number,
    )

    # 统计是否真的存在超过当前 max_length 的 Chunk。
    over_limit = [
        chunk
        for chunk, token_count in zip(
            chunks,
            lengths,
            strict=True,
        )
        if token_count > MAX_LENGTH
    ]

    print()
    print(
        f"Chunks over {MAX_LENGTH}:",
        len(over_limit),
    )


if __name__ == "__main__":
    main()