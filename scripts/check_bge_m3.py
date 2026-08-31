"""检查 BGE-M3 本地 Dense Embedding 是否正常。"""

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


MODEL_NAME = "BAAI/bge-m3"


def main() -> None:
    """加载 BGE-M3，并验证 Dense Embedding 与语义相似度。"""

    print("=" * 80)
    print("BGE-M3 Embedding Check")
    print("=" * 80)

    print("Model:", MODEL_NAME)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 CUDA 不可用，暂不继续加载 BGE-M3。"
        )

    # 加载 BGE-M3。
    #
    # use_fp16=True：
    # 使用 FP16 进行推理，可以降低显存占用并提升 GPU 推理速度。
    #
    # devices=["cuda:0"]：
    # 明确指定使用第 0 张 NVIDIA CUDA GPU。
    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=True,
        devices=["cuda:0"],
    )

    # 准备两个语义差异明显的测试文本。
    texts = [
        "生成式人工智能服务提供者应当依法处理训练数据。",
        "今天天气很好，我准备出去散步。",
    ]

    # 对文档文本生成 Dense Embedding。
    output = model.encode(
        texts,
        batch_size=2,
        max_length=512,

        # 当前 V1 只测试 Dense Embedding。
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    # dense_vecs 是一个 NumPy ndarray。
    embeddings = output["dense_vecs"]

    print()
    print(
        "Embedding type:",
        type(embeddings),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Embedding dtype:",
        embeddings.dtype,
    )

    print(
        "First vector norm:",
        np.linalg.norm(embeddings[0]),
    )

    print()
    print("✅ BGE-M3 Dense Embedding 正常")

    # ============================================================
    # 下面开始语义相似度实验
    # ============================================================

    query = [
        "训练数据需要满足什么要求？"
    ]

    # Query 同样要经过相同的 Embedding 模型。
    query_output = model.encode(
        query,
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    # 因为 query 只有 1 条：
    #
    # dense_vecs.shape == (1, 1024)
    #
    # [0] 取出其中唯一的一条向量：
    #
    # shape == (1024,)
    query_embedding = query_output[
        "dense_vecs"
    ][0]

    print()
    print("=" * 80)
    print("Semantic Similarity")
    print("=" * 80)

    for index, text in enumerate(texts):
        # BGE-M3 的 Dense Embedding 已基本归一化。
        #
        # 因此两个归一化向量的点积，
        # 就可以作为 cosine similarity 使用。
        score = float(
            query_embedding
            @ embeddings[index]
        )

        print(
            f"Similarity {index}:",
            round(score, 4),
            "|",
            text,
        )


if __name__ == "__main__":
    main()