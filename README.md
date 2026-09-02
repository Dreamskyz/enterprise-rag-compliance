# Enterprise RAG Compliance

企业 AI 合规与应用规范助手。

这是一个面向企业内部 RAG / Agent 研发人员与管理员的知识库问答系统，目标不是做“万能知识库”，而是围绕：

- 生成式人工智能合规；
- 深度合成服务规范；
- LLM 应用安全；
- 企业内部技术规范；

构建一套：

```text
有依据才回答
无依据必须拒答
支持 ACL
支持混合检索
支持 Rerank
支持 Citation
支持离线评测
支持 Failure Analysis
```

的可解释 RAG 系统。

> 当前项目仍处于持续迭代阶段。当前 Corpus 和 Evaluation Dataset 规模较小，主要用于验证架构、评测方法与优化闭环，不代表生产环境准确率。

---

# 1. Project Goals

本项目希望解决的核心问题不是：

```text
“怎么调用一个 LLM API”
```

而是：

```text
企业 RAG 系统如何做到：

正确检索
↓
权限隔离
↓
证据约束
↓
无依据拒答
↓
引用可追溯
↓
效果可评测
↓
失败可定位
↓
优化可回归验证
```

因此系统重点实现以下能力：

- 结构化知识库构建；
- Dense / BM25 / Hybrid Retrieval；
- RRF Fusion；
- bge-reranker 精排；
- Retrieval 前置 ACL；
- Evidence Relevance Gate；
- Evidence-Constrained Generation；
- Structured Refusal；
- Deterministic Citation Validation；
- Retrieval Evaluation；
- Answer / Refusal Evaluation；
- Citation Evaluation；
- Latency Benchmark；
- Prompt Regression Evaluation；
- Evaluation Snapshot Persistence。

---

# 2. Tech Stack

当前主要技术栈：

```text
Python 3.11
FastAPI
LangChain ecosystem
Qdrant
BGE-M3
bge-reranker-v2-m3
BM25
Jieba
SiliconFlow OpenAI-compatible API
pytest
Docker
```

本地开发环境：

```text
Windows
32 GB RAM
RTX 4060 Laptop GPU
```

---

# 3. Current Knowledge Base

当前知识库主要包含两篇公开法规：

```text
《生成式人工智能服务管理暂行办法》

《互联网信息服务深度合成管理规定》
```

当前经过结构化解析后得到：

```text
49 regulation chunks
```

后续计划继续加入：

```text
OWASP Top 10 for LLM Applications
FastAPI 官方文档
Qdrant 官方文档
企业内部技术规范示例
```

目标 Corpus：

```text
30–80 documents
```

---

# 4. System Architecture

当前系统核心链路：

```text
Public Documents
        ↓
Document Manifest
        ↓
Raw HTML
        ↓
HTML Loader
        ↓
Article Extraction
        ↓
Normalizer
        ↓
Regulation Parser
        ↓
Structure-aware Chunking
        ↓
KnowledgeChunk
        ↓
chunks.jsonl
        ↓
Embedding / BM25 Index
        ↓
Qdrant
```

在线 Query Pipeline：

```text
HTTP Request
        ↓
FastAPI
        ↓
AccessContext
        ↓
ACL-aware Retrieval
        ↓
┌───────────────────────┐
│ Dense Retrieval       │
│ BM25 Retrieval        │
└───────────────────────┘
        ↓
RRF Fusion
        ↓
bge-reranker-v2-m3
        ↓
Coarse Relevance Gate
        ↓
Evidence-Constrained Generation
        ↓
Answer / Structured Refusal
        ↓
Deterministic Citation Validation
        ↓
HTTP Response
```

---

# 5. Ingestion Pipeline

法规文档不会直接按固定字符长度粗暴切分。

当前采用：

```text
Regulation
↓
Chapter
↓
Article
↓
KnowledgeChunk
```

目前规则：

```text
一个法规条文 = 一个 Chunk
```

不做：

```text
Fixed-size Character Split
Sliding Window
Arbitrary Overlap
```

原因：

法规类文档天然具有较强的条款结构。

如果把：

```text
第七条
```

从中间切开，会破坏：

```text
法规语义完整性
Citation 可读性
Gold Annotation
```

因此当前采用 structure-aware chunking。

---

# 6. KnowledgeChunk

当前 Chunk 核心字段包括：

```text
chunk_id
document_id
title
document_type
language
version

chapter_number
chapter_title
article_number

content
retrieval_text

source_url
access_level

chunk_index
content_hash
```

其中：

```text
content
```

用于保存真实原文。

```text
retrieval_text
```

用于 Retrieval，包括：

```text
Document Title
Chapter
Article Number
Content
```

但不会加入：

```text
source_url
access_level
```

等无关 Retrieval 语义的信息。

---

# 7. Dense Retrieval

Embedding Model：

```text
BAAI/bge-m3
```

当前使用：

```text
Dense Embedding Dimension = 1024
```

向量存储：

```text
Qdrant
```

Collection：

```text
compliance_chunks_v1
```

Distance：

```text
Cosine
```

Dense Retrieval Pipeline：

```text
Query
↓
BGE-M3
↓
1024-d Dense Vector
↓
Qdrant Search
↓
Top-K
```

---

# 8. BM25 Retrieval

BM25 使用：

```text
Jieba
+
rank_bm25
```

BM25 Corpus 使用与 Dense 一致的：

```text
retrieval_text
```

当前没有为了 Seed Evaluation Dataset 人工调节 BM25 参数。

原因：

> 直接根据 Evaluation Dataset 调 Retrieval 参数会造成 Evaluation Leakage。

---

# 9. Hybrid Retrieval

当前 Hybrid Retrieval：

```text
Dense Top-K
+
BM25 Top-K
↓
Reciprocal Rank Fusion
```

RRF：

```text
score(d) = Σ 1 / (60 + rank(d))
```

其中：

```text
rank 从 1 开始
k = 60
```

当前没有加入：

```text
Dense Score Normalization
BM25 Score Normalization
手工 Weight
```

而是先使用 Rank-based Fusion，降低不同 Retrieval Score Scale 带来的影响。

---

# 10. Rerank

Fusion 后使用：

```text
BAAI/bge-reranker-v2-m3
```

进行 Cross-Encoder Rerank。

Pipeline：

```text
Dense Top20
+
BM25 Top20
↓
RRF
↓
Hybrid Top20
↓
bge-reranker-v2-m3
↓
Final Top5
```

需要注意：

```text
Reranker Score
```

不是概率，也不能直接解释为：

```text
80% 相关
```

当前只用于 Ranking 和 Coarse Relevance Analysis。

---

# 11. ACL Design

当前角色：

```text
guest
developer
admin
```

权限映射：

```text
guest
→ public

developer
→ public + developer

admin
→ all
```

ACL 的核心原则：

> 必须在 Retrieval Candidate Generation 之前执行。

正确链路：

```text
User Role
↓
Allowed Access Levels
↓
Qdrant Filter
+
Authorized BM25 Corpus
↓
Retrieval
```

而不是：

```text
Global Retrieval
↓
Top-K
↓
Post Filter
```

原因：

如果先全库检索，再过滤：

```text
Unauthorized Chunk
```

已经参与：

```text
Top-K Competition
RRF
Rerank
```

即使最终被删掉，也已经影响了合法结果排序。

当前 Synthetic Mixed-Access Evaluation：

```text
Unauthorized Final Results = 0
ACL Leakage Rate = 0%
```

> 当前 API 中的 `role` 属于 Demo Input，并不是生产级身份认证。生产环境应由可信 SSO / JWT / IAM 生成 AccessContext。

---

# 12. Evidence Relevance vs Answerability

开发过程中一个重要发现：

```text
Retrieval Relevance
≠
Answerability
```

例如一个问题：

```text
违法内容后必须在几小时内处理？
```

Retriever 可以找回：

```text
违法内容
及时处理
整改
报告
```

相关条款。

但 Evidence 中没有：

```text
具体小时数
```

因此：

```text
Retrieval Relevant
```

并不代表：

```text
Evidence Sufficient
```

当前架构采用：

```text
Coarse Relevance Gate
↓
过滤明显 Out-of-Domain Query
↓
Evidence-Constrained Generation
↓
判断证据是否真正足够回答
```

而没有采用：

```text
Top1 Rerank Score > Threshold
=> Answerable
```

因为实验中 Answerable / Hard Negative 的 Rerank Score 存在明显重叠。

---

# 13. Evidence-Constrained Generation

当前 LLM：

```text
deepseek-ai/DeepSeek-V4-Flash
```

通过：

```text
SiliconFlow OpenAI-compatible API
```

调用。

Generation 规则：

```text
只能使用 Evidence 中明确出现的信息
不能使用模型外部知识补事实
不能猜具体数字 / 时间 / 金额
不能把模糊事实变成精确事实
证据不足必须拒答
只能使用真实 Evidence ID
```

输出必须为结构化 JSON：

```json
{
  "answerable": true,
  "answer": "回答文本",
  "reason": "为什么证据足够",
  "citations": ["E1"]
}
```

或：

```json
{
  "answerable": false,
  "answer": null,
  "reason": "为什么证据不足",
  "citations": []
}
```

---

# 14. Citation Design

LLM 返回：

```text
E1
E2
E3
```

只是内部 Evidence ID。

程序不会直接把这些 ID 当最终 Citation。

而是：

```text
LLM Evidence ID
↓
Deterministic Mapping
↓
KnowledgeChunk
↓
Article
Title
Source URL
```

因此 Citation 最终由程序验证和生成。

LLM 不允许自行编造：

```text
chunk_id
法规来源
URL
条款
```

---

# 15. Minimal Sufficient Evidence

初版 Prompt 存在 Over-Citation：

```text
主题相关
→ 顺手全部引用
```

当前 Prompt v2 使用：

```text
Minimal Sufficient Evidence Set
```

原则：

```text
只引用直接支持最终答案实际陈述的 Evidence

主题相关
≠
需要引用

同一个事实已经有充分 Evidence
→ 不重复添加类似 Evidence

多个独立义务分别由不同 Evidence 支撑
→ 保留所有必要 Evidence

用户明确限定法规 / 制度 / 场景
→ 优先引用同范围 Evidence
```

目标不是：

```text
Citation 越少越好
```

而是：

```text
完整支持答案的最小充分证据集合
```

---

# 16. FastAPI

当前 API：

```text
GET /health

GET /ready

POST /api/v1/retrieve

POST /api/v1/ask
```

其中：

```text
/retrieve
```

只返回 Retrieval / Rerank 结果，不调用 LLM。

```text
/ask
```

执行完整：

```text
Retrieval
↓
Rerank
↓
Gate
↓
Generation / Refusal
↓
Citation
```

---

# 17. Runtime

Heavy Runtime 在 FastAPI Lifespan 中初始化一次。

避免：

```text
每个 HTTP Request
↓
重新加载 BGE-M3
↓
重新加载 Reranker
```

当前 Runtime 包括：

```text
Chunks
Embedding Service
Dense Retriever
BM25 Retriever
Hybrid Retriever
Reranker
Evidence Gate
LLM Service
Grounded Answerer
Query Service
```

---

# 18. Evaluation

当前项目已经建立一套可重复的 Evaluation Pipeline：

```text
Evaluation Dataset
↓
Retrieval Quality
↓
Failure Analysis
↓
Latency Benchmark
↓
Answer / Refusal Eval
↓
Citation Eval
↓
Raw Snapshot Persistence
↓
Offline Analysis
↓
Gold Audit
↓
Prompt Regression
```

完整实验记录见：

```text
docs/evaluation.md
```

---

# 19. Evaluation Dataset

当前 Seed Dataset：

```text
20 cases
```

其中：

```text
Answerable   : 14
Unanswerable : 6
```

包含：

```text
direct
paraphrase
short
ambiguous
hard_negative
out_of_domain
```

当前所有结果都应理解为：

> 20 条 Seed Dataset 上的开发阶段实验结果。

不能解释为生产系统准确率。

---

# 20. Retrieval Quality

当前 14 条 Answerable Query：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.7500 | 0.8929 | 1.0000 | 1.0000 | 0.8750 |
| Hybrid RRF | 0.8929 | 1.0000 | 1.0000 | 1.0000 | 0.9643 |
| Hybrid + Rerank | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

当前结论：

```text
Dense
```

是当前 Corpus 上质量 / 成本非常强的 Baseline。

BM25：

```text
Top1 较弱
但 Recall@5 = 1
```

说明仍有词法补充价值。

当前 Seed Dataset 上：

```text
Hybrid RRF
```

没有稳定超过 Dense。

Rerank 可以修复部分 Fusion 排序问题，但总体质量只恢复到 Dense 水平。

因此：

> 当前不能得出“Hybrid 一定优于 Dense”的结论。

---

# 21. Retrieval Failure Example

典型 Case：

```text
R010
违法内容怎么处理？
```

Dense：

```text
Rank 1
生成式人工智能服务管理暂行办法 第十四条
```

Hybrid RRF：

```text
Rank 1
深度合成管理规定 第十条

Rank 2
生成式人工智能服务管理暂行办法 第十四条
```

原因：

BM25 强化了：

```text
违法内容
处置
```

等词法信号。

经过 Rerank：

```text
生成式 AI 第十四条
```

重新回到 Rank 1。

这说明：

> Hybrid Fusion 可能改善召回，也可能破坏已正确的 Dense 排序。

因此当前没有针对这个 Eval Case 调整 RRF 参数，以避免 Evaluation Leakage。

---

# 22. Retrieval Latency

最终采用：

```text
Query-level Interleaved Benchmark
```

而不是按 Method Block 顺序执行。

原因：

早期 Blocked Benchmark 观察到明显：

```text
GPU Warm-State
Method Order Bias
```

最终测试条件：

```text
Queries        : 20
Rounds         : 5
Samples/method : 100
Warmup         : 2
GPU            : RTX 4060 Laptop
```

结果：

| Method | Mean | P50 | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.39 ms | 0.32 ms | 0.52 ms | 0.09 ms | 6.09 ms |
| Dense | 53.57 ms | 52.20 ms | 75.44 ms | 30.35 ms | 98.18 ms |
| Hybrid RRF | 56.54 ms | 54.71 ms | 77.51 ms | 33.48 ms | 106.12 ms |
| Hybrid + Rerank | 452.33 ms | 455.68 ms | 503.99 ms | 348.80 ms | 538.37 ms |

当前关系：

```text
BM25
<
Dense
≈
Hybrid RRF
<<<<
Hybrid + Rerank
```

Reranker 是当前 Retrieval Pipeline 中最主要的额外延迟来源。

> 该 Benchmark 不包含模型初始化、FastAPI、HTTP 和 LLM Generation。

---

# 23. Answer / Refusal Evaluation

Prompt v2 完整 Full-RAG Evaluation：

```text
20 cases
```

结果：

| Metric | Result |
|---|---:|
| TP | 14 |
| TN | 6 |
| FP | 0 |
| FN | 0 |
| Overall Decision Accuracy | 1.0000 |
| Answerable Accuracy | 1.0000 |
| Refusal Accuracy | 1.0000 |
| Hard Negative Refusal Accuracy | 1.0000 |
| OOD Refusal Accuracy | 1.0000 |

当前 Seed Dataset：

```text
Decision Failure = 0
```

但需要强调：

> 当前结果只代表该 Seed Dataset 的单次 Full-RAG Evaluation，不代表生产环境准确率为 100%。

---

# 24. Citation Gold Audit

初版 Citation Evaluation 直接使用：

```text
Retrieval Gold
```

后来 Failure Analysis 发现：

```text
Retrieval Gold
≠
Citation Gold
```

因此当前 Dataset 拆分：

```text
gold_chunk_ids
→ Retrieval Evaluation

citation_gold_chunk_ids
→ Citation Evaluation
```

另外增加：

```text
strict_citation_eval
```

用于排除：

```text
ambiguous
multiple acceptable source
```

等不适合严格 Citation Precision 的 Query。

---

# 25. Prompt v1 → Prompt v2

Prompt v1 Citation：

| Metric | v1 |
|---|---:|
| Citation Precision | 0.9524 |
| Citation Recall | 0.9821 |
| Strict Citation Precision | 0.9444 |
| Strict Citation Recall | 0.9792 |

主要 Failure：

```text
R001
生成式人工智能服务处理训练数据需要遵守什么规定？
```

Prompt v1 引用了：

```text
核心第七条
+
相关第四条
+
其他法规第十四条
```

形成 Over-Citation。

Prompt v2 加入：

```text
Minimal Sufficient Evidence
+
Scope Matching
```

完整 Re-evaluation：

| Metric | Prompt v1 | Prompt v2 |
|---|---:|---:|
| Decision Accuracy | 1.0000 | 1.0000 |
| Refusal Accuracy | 1.0000 | 1.0000 |
| Citation Precision | 0.9524 | 1.0000 |
| Citation Recall | 0.9821 | 0.9643 |
| Strict Citation Precision | 0.9444 | 1.0000 |
| Strict Citation Recall | 0.9792 | 1.0000 |
| Strict Citation Hit Rate | 1.0000 | 1.0000 |

Prompt v2：

```text
Strict Citation Mismatch = 0

Decision Regression = 0

Evidence-ID Leak = 0
```

因此当前接受 Prompt v2。

All-case Citation Recall 的下降主要来自一个：

```text
strict_citation_eval = false
```

的 ambiguous Case，详见：

```text
docs/evaluation.md
```

---

# 26. Evaluation Snapshot

真实 LLM Evaluation 会受到：

```text
API Cost
Network Latency
Generation Randomness
```

影响。

因此项目不会在每次 Failure Analysis 时重新调用 LLM。

当前：

```text
Expensive Full-RAG Run
↓
JSONL Snapshot
↓
Offline Metrics
↓
Failure Analysis
↓
Prompt Comparison
```

当前代表性 Snapshot：

```text
data/eval/results/
├── answer_eval_v1_run_001.jsonl
├── answer_eval_v1_run_001_audited.jsonl
└── answer_eval_v1_run_002.jsonl
```

含义：

```text
run_001
→ Prompt v1 原始输出

run_001_audited
→ Prompt v1 + 最新 Citation Annotation

run_002
→ Prompt v2 Full-RAG Result
```

---

# 27. Reproduction

## Install

创建 Python 3.11 虚拟环境后：

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

---

## Start Qdrant

```powershell
docker compose up -d
```

检查：

```powershell
docker compose ps
```

Qdrant Dashboard：

```text
http://localhost:6333/dashboard
```

---

## Build Chunks

```powershell
python scripts/build_chunks.py
```

---

## Build Vector Index

```powershell
python scripts/build_vector_index.py
```

---

## Run Retrieval Evaluation

```powershell
python scripts/run_retrieval_eval.py
```

---

## Run Final Retrieval Latency Benchmark

```powershell
python scripts/run_interleaved_retrieval_latency_benchmark.py
```

---

## Inspect Evaluation Dataset

```powershell
python scripts/inspect_retrieval_eval_dataset.py
```

---

## Run Full-RAG Evaluation

需要：

```text
Qdrant
BGE-M3
bge-reranker-v2-m3
SILICONFLOW_API_KEY
```

运行：

```powershell
python scripts/run_answer_eval.py
```

---

## Offline Snapshot Analysis

```powershell
python scripts/analyze_answer_eval_snapshot.py
```

---

## Prompt v1 / v2 Comparison

```powershell
python scripts/compare_answer_eval_snapshots.py
```

Offline Comparison 不调用 LLM。

---

# 28. FastAPI Run

启动：

```powershell
uvicorn enterprise_rag.api.app:create_app --factory --reload
```

启动完成后可以访问：

```text
http://127.0.0.1:8000/docs
```

Swagger 中当前主要接口：

```text
GET /health

GET /ready

POST /api/v1/retrieve

POST /api/v1/ask
```

---

# 29. Example Ask

示例：

```json
{
  "query": "生成式人工智能服务处理训练数据需要遵守什么规定？",
  "role": "guest"
}
```

系统会：

```text
Retrieval
↓
ACL
↓
Hybrid Fusion
↓
Rerank
↓
Evidence Gate
↓
Evidence-Constrained Generation
↓
Citation
```

最终回答只允许基于 Knowledge Base Evidence。

---

# 30. Structured Refusal

对于类似：

```text
违法内容必须在几小时内处理？
```

如果法规 Evidence 只写：

```text
及时处理
```

系统必须拒绝给出：

```text
24 小时
48 小时
```

等不存在于 Evidence 的具体数字。

目标：

```text
不知道
就明确说证据不足
```

而不是利用模型常识补全。

---

# 31. Repository Structure

当前主要目录：

```text
enterprise-rag-compliance1/
│
├── data/
│   ├── manifest/
│   ├── raw/
│   ├── processed/
│   └── eval/
│       ├── retrieval_eval_v1.jsonl
│       └── results/
│
├── docs/
│   └── evaluation.md
│
├── scripts/
│   ├── build_chunks.py
│   ├── build_vector_index.py
│   ├── run_retrieval_eval.py
│   ├── run_retrieval_latency_benchmark.py
│   ├── run_interleaved_retrieval_latency_benchmark.py
│   ├── run_answer_eval.py
│   ├── run_targeted_citation_regression.py
│   ├── analyze_answer_eval_snapshot.py
│   ├── compare_answer_eval_snapshots.py
│   └── ...
│
├── src/
│   └── enterprise_rag/
│       ├── acl/
│       ├── api/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── ingestion/
│       ├── retrieval/
│       ├── runtime/
│       └── service/
│
├── tests/
│
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

# 32. Testing

运行：

```powershell
pytest -v
```

项目包含：

```text
Ingestion
Chunking
Retrieval
ACL
Evidence Gate
Generation
API
Evaluation
Persistence
Regression
```

等模块的测试。

---

# 33. Current Design Conclusions

目前可以得到几个阶段性结论。

## Hybrid Retrieval

Hybrid Retrieval 不一定天然优于 Dense。

当前小型法规 Corpus 上：

```text
Dense
```

已经很强。

RRF 存在个别排序退化。

因此 Hybrid 的价值需要在：

```text
更大
更异构
更多词法精确匹配需求
```

的 Corpus 上继续验证。

---

## Rerank

Rerank 可以修复部分 Fusion Failure。

但当前：

```text
~452 ms mean
```

带来的额外成本明显。

因此：

> 是否始终开启 Rerank 应由 Quality × Latency Evaluation 决定，而不是因为“Rerank 看起来更高级”。

---

## ACL

ACL 必须前置到 Retrieval Candidate Generation。

不能：

```text
先全库检索
再过滤
```

这是当前系统的重要安全设计原则。

---

## Citation

Retrieval Gold 和 Citation Gold 不应该混用。

Citation 目标不是：

```text
引用越多越安全
```

也不是：

```text
引用越少越好
```

而是：

```text
最小充分证据集
```

---

## Evaluation

RAG Optimization 不应依赖：

```text
“看起来回答不错”
```

而应该形成：

```text
Metric
↓
Failure Case
↓
Root Cause
↓
Change
↓
Targeted Regression
↓
Full Re-evaluation
```

的闭环。

---

# 34. Current Limitations

当前仍有明显限制：

1. Corpus 只有两篇法规、49 个 Chunk；
2. Evaluation Dataset 只有 20 条；
3. Prompt v1 / v2 目前均只做单次 Full-RAG Run；
4. 尚未加入 OWASP / FastAPI / Qdrant 等异构文档；
5. 尚未完成独立 Answer Correctness 指标；
6. 尚未完成完整 Faithfulness Evaluation；
7. 尚未进行正式并发 Load Test；
8. API Role 目前只是 Demo，并非生产认证系统；
9. Evaluation Gold 仍主要人工维护；
10. 当前结果不可解释为生产环境准确率或 SLA。

---

# 35. Roadmap

后续计划：

```text
更多公开文档
↓
OWASP LLM Top 10
↓
FastAPI / Qdrant 技术规范
↓
30–80 Documents
↓
50–100 Evaluation Queries
↓
Held-out Eval Set
↓
More Hard Negatives
↓
Cross-document QA
↓
Faithfulness / Answer Correctness
↓
UI / Demo
```

---

# 36. Project Status

当前已完成：

```text
✅ Structured Ingestion
✅ Structure-aware Chunking
✅ BGE-M3 Embedding
✅ Qdrant
✅ Dense Retrieval
✅ BM25 Retrieval
✅ RRF Hybrid Retrieval
✅ bge-reranker
✅ ACL-aware Retrieval
✅ Evidence Gate
✅ Evidence-Constrained Generation
✅ Structured Refusal
✅ Citation Validation
✅ FastAPI Runtime
✅ Retrieval Evaluation
✅ Latency Benchmark
✅ Answer / Refusal Evaluation
✅ Citation Gold Audit
✅ Evaluation Snapshot Persistence
✅ Prompt Regression Evaluation
```

仍在继续：

```text
🚧 Corpus Expansion
🚧 Evaluation Dataset Expansion
🚧 OWASP / Technical Docs
🚧 Answer Correctness / Faithfulness
🚧 Demo UI
```

---

# 37. License / Data

当前知识库内容主要来自公开法规与公开技术文档。

项目仅用于：

```text
学习
研究
RAG 系统设计演示
```

不构成法律意见。
