"""Retrieval baseline 配置。"""


# Dense Retriever 候选数量。
DENSE_TOP_K = 20


# BM25 Retriever 候选数量。
BM25_TOP_K = 20


# RRF 平滑常数。
RRF_K = 60


# RRF 融合后送入 Reranker 的数量。
HYBRID_TOP_K = 20


# Reranker 最终返回数量。
RERANK_TOP_K = 5


# BGE-M3 最大输入 Token 数。
EMBEDDING_MAX_LENGTH = 512


# BGE-M3 文档批处理大小。
EMBEDDING_BATCH_SIZE = 8