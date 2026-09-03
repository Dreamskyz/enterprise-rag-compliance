# Enterprise RAG Compliance

企业 AI 合规与应用规范助手。

这是一个面向企业内部 RAG / Agent 研发人员与管理员的知识库问答系统。

项目目标不是构建“万能知识库”，而是围绕：

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
支持 Regression Evaluation
```

的可解释 RAG 系统。

> 当前项目仍处于持续迭代阶段。当前 Corpus 和 Evaluation Dataset 规模仍然较小，主要用于验证系统架构、评测方法与优化闭环，不代表生产环境准确率或 SLA。

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
- 异构文档 Ingestion；
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
- Evaluation Snapshot Persistence；
- Role-aware Evaluation；
- Retrieval Failure Analysis；
- Gold Integrity Validation。

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

当前 Corpus V2 包含 5 篇公开文档，覆盖：

```text
中国法规
+
LLM 应用安全
+
FastAPI 技术文档
```

当前文档：

```text
1. 《生成式人工智能服务管理暂行办法》

2. 《互联网信息服务深度合成管理规定》

3. OWASP LLM01:2025 Prompt Injection

4. FastAPI Dependencies

5. FastAPI Lifespan Events
```

当前结构化后共得到：

```text
104 chunks
```

按文档统计：

```text
cn_genai_interim_2023                  24
cn_deep_synthesis_2022                25
owasp_llm01_prompt_injection_2025     25
fastapi_dependencies                  17
fastapi_lifespan                      13
```

按 Document Type：

```text
regulation                 49
security_guideline         25
technical_documentation    30
```

按 Access Level：

```text
public       74
developer    30
```

其中 FastAPI 官方文档本身是公开资料。

项目中将 FastAPI 文档标记为：

```text
access_level = developer
```

只是为了模拟企业内部技术规范的 ACL 场景，并不表示 FastAPI 官方文档本身具有真实访问限制。

后续计划继续加入：

```text
更多 OWASP LLM Top 10 文档
Qdrant 官方技术文档
更多企业内部技术规范模拟文档
```

长期目标 Corpus：

```text
30–80 documents
```

---

# 4. System Architecture

当前离线知识库构建已经从“法规专用 Pipeline”升级为支持多种文档结构的异构 Ingestion Pipeline：

```text
Public Documents
        ↓
Document Manifest
        ↓
Raw HTML
        ↓
Document-specific Loader
        ↓
Normalizer
        ↓
Document-Type Router
        ↓
┌──────────────────────────────┐
│ Regulation Parser            │
│ Generic Section Parser       │
└──────────────────────────────┘
        ↓
Structure-aware Chunker
        ↓
Unified KnowledgeChunk
        ↓
Corpus Validation
        ↓
chunks.jsonl
        ↓
Embedding / BM25 Index
        ↓
Qdrant
```

当前 Document Type Router：

```text
regulation
→ Regulation Parser
→ Article-aware Chunking

security_guideline
→ Generic Section Parser
→ Section-aware Chunking

technical_documentation
→ Generic Section Parser
→ Section-aware Chunking
```

所有文档最终统一进入：

```text
KnowledgeChunk
```

避免不同数据源向下游暴露不同的数据结构。

---

## Online Query Pipeline

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

## Regulation

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

当前规则：

```text
一个法规条文 = 一个 Chunk
```

不做：

```text
Fixed-size Character Split
Sliding Window
Arbitrary Overlap
```

原因是法规文档天然具有较强的条款结构。

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

因此法规采用 Article-aware structure-aware chunking。

---

## Security / Technical Documentation

OWASP 和 FastAPI 不具有“第几章 / 第几条”的法规结构。

因此采用：

```text
HTML
↓
Heading Hierarchy
↓
GenericSection
↓
section_title
section_path
↓
Section-aware Chunker
↓
KnowledgeChunk
```

例如：

```text
Dependencies
>
First Steps
>
Create a dependency, or "dependable"
```

会保存为：

```text
section_path
```

如果一个 Section 内容过长，则优先：

```text
按自然段聚合
```

只有单个段落本身超长时才进行硬切分。

当前通用 Section Chunk 上限：

```text
1200 characters
```

目标不是让所有 Chunk 长度一致，而是尽量保留结构和语义边界。

---

# 6. KnowledgeChunk

当前统一 Chunk Schema 核心字段包括：

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

section_title
section_path

content
retrieval_text

source_url
access_level

chunk_index
content_hash
```

其中法规字段：

```text
chapter_number
chapter_title
article_number
```

在技术文档中允许为：

```text
None
```

而通用文档使用：

```text
section_title
section_path
```

描述结构层级。

---

## content

```text
content
```

保存真实正文。

---

## retrieval_text

```text
retrieval_text
```

用于 Retrieval。

法规通常包含：

```text
Document Title
Chapter
Article Number
Content
```

通用文档通常包含：

```text
Document Title
Section Path
Content
```

但不会把：

```text
source_url
access_level
```

等与语义检索无关的信息加入 Retrieval Text。

---

# 7. Corpus Validation

构建后的 Corpus 不会直接写入向量库。

首先执行 Schema Validation，检查：

```text
chunk_id
document_id
document_type
content
retrieval_text
access_level
content_hash
chunk_index
```

以及不同 Document Type 对应的结构字段。

例如：

```text
regulation
→ article_number 必须存在

security_guideline
technical_documentation
→ section_title / section_path 必须存在
```

除此之外，还进行了 Corpus-level Inspection：

```text
Document Distribution
Document Type Distribution
ACL Distribution
Chunk Length Distribution
Duplicate Chunk ID
Duplicate Content Hash
Near-max Chunk
Multi-part Section Boundary
```

Corpus V2 当前：

```text
Duplicate Chunk ID = 0
Duplicate Content Hash Group = 0
```

---

# 8. Dense Retrieval

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
ACL Qdrant Filter
↓
Qdrant Search
↓
Top-K
```

Corpus V2 Evaluation 显示：

```text
中文 Query
→ BGE-M3
→ 英文 OWASP / FastAPI
```

能够保持较稳定的跨语言语义检索能力。

---

# 9. BM25 Retrieval

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

BM25 同样执行 ACL 过滤，不允许无权限 Chunk 进入候选结果。

当前没有为了 Evaluation Dataset 人工调整 BM25 参数。

原因：

> 直接根据当前 Evaluation Dataset 调 Retrieval 参数容易造成 Evaluation Leakage。

Corpus V2 中发现 BM25 在：

```text
中文 Query
→ 英文 OWASP / FastAPI 原文
```

场景下存在明显 lexical mismatch。

该现象会在后文 Retrieval Evaluation 中详细说明。

---

# 10. Hybrid Retrieval

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

而是先使用 Rank-based Fusion，降低不同 Retrieval Score Scale 带来的直接影响。

但 Corpus V2 Evaluation 表明：

> Rank-based Fusion 虽然避免了直接比较 Dense Score 与 BM25 Score，但并不能自动判断某个 Retriever 在当前 Query 上是否可靠。

当 BM25 在跨语言场景下发生 lexical mismatch 时：

```text
弱 BM25 排名
+
强 Dense 排名
↓
Unweighted RRF
↓
可能把错误候选推到 Gold 前面
```

因此：

> Hybrid Retrieval 并不天然优于单路 Dense Retrieval。

---

# 11. Rerank

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

当前主要用于：

```text
Ranking
+
Coarse Relevance Analysis
```

Corpus V2 Evaluation 发现：

```text
R021 Rank3 → Rank1
R022 Rank3 → Rank1
R023 Rank4 → Rank1
R027 Rank2 → Rank1
```

说明 Reranker 能够明显修复部分 RRF Fusion Noise。

但当前整体：

```text
Hybrid + Rerank
```

仍未超过 Dense baseline。

因此：

> Reranker 的价值需要结合真实 Failure Case 和 Quality × Latency Evaluation 判断，而不是因为架构更复杂就默认认为更好。

---

# 12. ACL Design

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
→ public + developer + admin
```

ACL 的核心原则：

> 必须在 Retrieval Candidate Generation 之前执行。

正确链路：

```text
User Role
↓
Allowed Access Levels
↓
┌─────────────────────────────┐
│ Qdrant Payload Filter       │
│ Authorized BM25 Candidate   │
└─────────────────────────────┘
↓
Retrieval
↓
Fusion / Rerank
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

即使最终删除，也已经影响合法结果的排名。

当前 ACL Smoke Test 已验证：

```text
guest
查询 FastAPI 技术问题
→ 不会检索 developer FastAPI Chunk

developer
查询 FastAPI 技术问题
→ 能检索对应 FastAPI Chunk
```

当前 Synthetic Mixed-Access Evaluation：

```text
Unauthorized Final Results = 0
ACL Leakage Rate = 0%
```

> 当前 FastAPI 文档被标记为 `developer` 只是项目中的企业 ACL 模拟。FastAPI 官方文档本身是公开资料。

> 当前 API 中的 `role` 属于 Demo Input，并不是生产级身份认证。生产环境应由可信 SSO / JWT / IAM 生成 AccessContext。

---

# 13. Evidence Relevance vs Answerability

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

原因是实验中：

```text
Answerable
Hard Negative
```

的 Rerank Score 存在明显重叠。

---

# 14. Evidence-Constrained Generation

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

# 15. Citation Design

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
Article / Section
Title
Source URL
```

最终 Citation 由程序验证和生成。

LLM 不允许自行编造：

```text
chunk_id
法规来源
URL
条款
```

---

# 16. Minimal Sufficient Evidence

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

# 17. FastAPI

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

# 18. Runtime

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

# 19. Evaluation Architecture

当前项目已经建立一套可重复的 Evaluation Pipeline：

```text
Evaluation Dataset
↓
Gold Integrity Check
↓
Role-aware Retrieval Evaluation
↓
Retrieval Quality
↓
Pairwise Ablation Analysis
↓
Per-Method Failure Inspector
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

# 20. Evaluation Dataset Versioning

当前不再只有一个 Evaluation Dataset。

而是明确区分：

```text
Retrieval Eval V1
+
Retrieval Eval V2
```

---

## Retrieval Eval V1

路径：

```text
data/eval/retrieval_eval_v1.jsonl
```

规模：

```text
20 cases

Answerable   : 14
Unanswerable : 6
```

作用：

```text
Frozen Regression Benchmark
```

即：

> Corpus 扩展、Metadata Schema 升级、ACL Evaluation 升级之后，验证旧能力是否发生回归。

V1 Dataset 已冻结，不因为新的 Corpus 或 Retriever 输出修改 Gold。

---

## Retrieval Eval V2

路径：

```text
data/eval/retrieval_eval_v2.jsonl
```

规模：

```text
34 cases

Answerable   : 26
Unanswerable : 8
```

Role Distribution：

```text
guest       : 27
developer   : 7
```

Category Distribution：

```text
direct          : 14
paraphrase      : 8
short           : 3
ambiguous       : 1
hard_negative   : 6
out_of_domain   : 2
```

V2 在 V1 基础上新增：

```text
OWASP Prompt Injection
FastAPI Dependencies
FastAPI Lifespan
developer-role Retrieval
new Hard Negatives
```

---

# 21. Role-aware Retrieval Evaluation

`RetrievalEvalCase` 当前包含：

```text
role
```

因此一条 Evaluation Case 不再只有：

```text
query
→ gold
```

而是：

```text
query
+
role
↓
authorized candidate space
↓
gold
```

V1 历史 Dataset 没有显式 `role` 字段。

由于旧 Runner 当时默认：

```text
guest
```

因此新 Loader 对缺失 role 的 V1 Case 仍然保持：

```text
guest
```

从而避免 Schema Evolution 偷偷改变历史 benchmark 语义。

V2 中则显式保存：

```json
"role": "guest"
```

或：

```json
"role": "developer"
```

---

# 22. Retrieval Gold Integrity

在执行 Retrieval Evaluation 前，会检查所有：

```text
gold_chunk_ids
citation_gold_chunk_ids
```

是否真实存在于当前：

```text
data/processed/chunks.jsonl
```

Corpus V2 当前结果：

```text
Missing Retrieval Gold = 0
Missing Citation Gold = 0
```

这可以防止：

```text
Chunk ID Schema Evolution
手工 Gold 拼写错误
旧 Chunk ID 漂移
```

导致假的 Recall Failure。

---

# 23. Frozen V1 Retrieval Regression

Corpus 从：

```text
49 regulation chunks
```

扩展为：

```text
104 heterogeneous chunks
```

后，使用冻结的 Retrieval Eval V1 重新测试。

结果：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.8214 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| Hybrid RRF | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid + Rerank | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

与 Corpus V1 相比：

```text
Dense
→ 保持不变

BM25
→ Recall@1 / MRR 提升

Hybrid RRF
→ Recall@1 / MRR 提升

Hybrid + Rerank
→ 保持不变
```

因此：

> Corpus V2 扩展后，在冻结的旧任务上没有观察到 Retrieval Quality Regression。

需要注意：

> 当前 V1 Dataset 仍然只有 14 条 Answerable Query，因此该结果属于 preliminary regression evidence，而不是最终算法结论。

---

# 24. Corpus V2 Retrieval Quality

当前 Retrieval Eval V2：

```text
34 cases
26 answerable
```

结果：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9808 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.5577 | 0.7308 | 0.7692 | 0.7692 | 0.6551 |
| Hybrid RRF | 0.8269 | 0.9615 | 1.0000 | 1.0000 | 0.9006 |
| Hybrid + Rerank | 0.9423 | 1.0000 | 1.0000 | 1.0000 | 0.9808 |

当前质量排序：

```text
Dense
>
Hybrid + Rerank
>
Hybrid RRF
>
BM25
```

需要强调：

```text
Recall@1 = 0.9808
```

并不表示 Dense 存在普通的 Rank1 Failure。

原因是 Dataset 中存在 multi-gold Query。

General Failure Inspector 结果：

```text
Dense Non-Top1 Gold Cases
→ No cases
```

即：

> 26 条 Answerable Query 中，每条 Query 至少有一个核心 Retrieval Gold 位于 Dense Rank1。

---

# 25. Cross-Lingual Dense Retrieval

Corpus V2 新增了英文：

```text
OWASP
FastAPI
```

而 Evaluation Query 主要仍然使用中文。

例如：

```text
提示词注入 最小权限
```

Gold：

```text
Enforce privilege control
and least privilege access
```

Dense 能够稳定完成：

```text
中文 Query
↓
BGE-M3
↓
英文 Evidence
```

的跨语言语义匹配。

因此当前实验支持：

> BGE-M3 对该中英混合知识库具有较强的跨语言 Dense Retrieval 能力。

---

# 26. BM25 Cross-Lingual Failure

Corpus V2 中：

```text
BM25 Recall@10 = 0.7692
```

General Failure Inspector 找到 6 条：

```text
Top10 中完全没有任何 Gold
```

的 Query：

```text
R021
OWASP 所说的直接提示词注入是什么？

R022
LLM 从网页或文件读取到隐藏恶意指令并因此改变行为，属于什么风险？

R023
提示词注入 最小权限

R024
OWASP 对模型执行高风险或高权限操作有什么人工控制建议？

R025
如何通过渗透测试和攻击模拟来降低提示词注入风险？

R030
FastAPI 应用启动时加载共享机器学习模型，并在关闭时释放资源，推荐怎么做？
```

这些 Query 的共同特点是：

```text
中文 Query
→ 英文 OWASP / FastAPI 文档
```

普通 BM25 主要依赖 lexical overlap。

因此：

```text
最小权限
```

并不会自然等价于：

```text
least privilege
```

这形成：

```text
Cross-lingual Lexical Mismatch
```

因此：

> BM25 在当前 Corpus V2 中仍具有低成本词法检索价值，但不适合作为中英混合知识库的主 Retriever。

---

# 27. Hybrid RRF Failure Analysis

Corpus V2 中发现 4 个：

```text
Dense
→
Hybrid RRF
```

排名退化 Case：

```text
R021
R022
R023
R027
```

其中：

```text
R021
Dense Gold Rank1
→ Hybrid Rank3

R022
Dense Gold Rank1
→ Hybrid Rank3

R023
Dense Gold Rank1
→ Hybrid Rank4

R027
Dense Gold Rank1
→ Hybrid Rank2
```

前三个 OWASP Case 与 BM25 的跨语言 lexical mismatch 高度重合。

原因：

RRF 使用：

```text
Rank
```

进行融合。

它不会知道：

```text
当前 BM25 Rank1 是否真的可靠
```

因此可能出现：

```text
Strong Dense
+
Weak BM25
↓
Unweighted RRF
↓
Weak Retriever Contaminates Fusion
```

所以当前实验进一步确认：

> Hybrid Retrieval 并不天然优于 Dense。融合效果依赖 Corpus、语言分布以及各子 Retriever 在当前 Query 上的可靠性。

---

# 28. Reranker Failure Recovery

Corpus V2 中：

```text
Hybrid RRF
→
bge-reranker
```

成功修复：

```text
R021
Rank3 → Rank1

R022
Rank3 → Rank1

R023
Rank4 → Rank1

R027
Rank2 → Rank1
```

说明 Cross-Encoder 能进一步结合：

```text
Query
+
Candidate Content
```

判断细粒度相关性，从而修复部分 RRF Fusion Noise。

但是总体：

```text
Hybrid + Rerank
Recall@1 = 0.9423

Dense
Recall@1 = 0.9808
```

当前 Hybrid + Rerank 仍未超过 Dense baseline。

因此当前结论是：

> Reranker 在复杂异构 Corpus 上具有明确的 Ranking Repair 价值，但不能因为增加了 Reranker 就默认认为整体效果一定优于 Dense。

---

# 29. Remaining Reranker Failure

当前：

```text
Hybrid + Rerank Non-Top1 Gold Cases
```

只有一个：

```text
R030
```

Query：

```text
FastAPI 应用启动时加载共享机器学习模型，
并在关闭时释放资源，推荐怎么做？
```

Gold：

```text
FastAPI Lifespan
> Lifespan
```

Reranker：

```text
Rank1
Alternative Events (deprecated)
> startup and shutdown together

Rank2
Lifespan
```

这里 Rank1 并非完全无关。

它讨论：

```text
startup / shutdown
resource acquire / release
recommended lifespan
```

而 Gold Section 更直接给出了：

```text
lifespan parameter
@asynccontextmanager
yield 前加载
yield 后清理
```

因此该 Case 更接近：

```text
高度相关 Supporting Section
vs
更精确的 Minimal Core Evidence
```

属于细粒度 Section Ranking Failure。

当前没有为了这一条 Case：

```text
修改 Gold
调整 RRF
调整 TopK
修改 Query
```

以避免 Evaluation Overfitting。

---

# 30. Retrieval Latency

正式 Retrieval Latency Benchmark 使用：

```text
Query-level Interleaved Benchmark
```

而不是按：

```text
Method Block
```

顺序执行。

原因：

早期 Blocked Benchmark 观察到明显：

```text
GPU Warm-State
Method Order Bias
```

因此正式实验采用：

```text
对每个 Query
↓
随机四种 Method 执行顺序
↓
Repeated Runs
↓
P50 / P95
```

正式 Benchmark 条件：

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

> 该正式 Benchmark 来自此前冻结的 Query-level Interleaved Latency Experiment。Corpus V2 Ablation Summary 中附带的单次 Mean Latency 仅用于诊断，不用于覆盖这组正式 latency 结论。

> Benchmark 不包含模型初始化、FastAPI、HTTP 和 LLM Generation。

---

# 31. Answer / Refusal Evaluation

当前 Full-RAG Answer / Refusal Evaluation 仍基于：

```text
Retrieval Eval V1 对应的 20 条 Seed Cases
```

尚未在最新 Retrieval Eval V2 的 34 条 Case 上重新执行完整 LLM Evaluation。

Prompt v2 当前结果：

```text
20 cases
```

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

> 当前结果只代表该 Seed Dataset 的一次 Full-RAG Evaluation，不代表生产环境准确率为 100%。

---

# 32. Citation Gold Audit

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

一个重要原则是：

> Gold 不能根据当前 Retriever 或 LLM 的输出反向生成。

Citation Gold 必须经过独立 Corpus / Source Audit。

---

# 33. Prompt v1 → Prompt v2

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

的 ambiguous Case。

详见：

```text
docs/evaluation.md
```

---

# 34. Evaluation Snapshot

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
Immutable JSONL Snapshot
↓
Offline Metrics
↓
Failure Analysis
↓
Gold Audit
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

这种设计实现：

```text
Expensive Inference Once
↓
Offline Analysis Many Times
```

---

# 35. Failure Analysis

当前 Retrieval Evaluation 不只输出 Aggregate Metrics。

还支持：

```text
Dense → Hybrid Rank Degradation

Hybrid → Rerank Rank Improvement

Dense Non-Top1 Gold Cases

BM25 Gold Misses @10

Hybrid RRF Non-Top1 Gold Cases

Hybrid + Rerank Non-Top1 Gold Cases
```

这使 Retrieval Optimization 不再停留在：

```text
Recall 下降了
```

而是可以定位到：

```text
哪个 Query
↓
哪个 Gold
↓
在哪个 Method
↓
掉到了第几名
↓
前面出现了什么错误候选
```

当前 Evaluation 原则：

```text
Measure
↓
Inspect
↓
Explain
↓
Then Tune
```

而不是：

```text
指标不好
↓
改参数
↓
直到测试集好看
```

---

# 36. Reproduction

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

当前 Corpus V2 预期：

```text
104 chunks
```

---

## Inspect Corpus

```powershell
python scripts/inspect_corpus.py
```

用于检查：

```text
Document Distribution
Document Type Distribution
ACL Distribution
Chunk Length
Duplicate Content
Long Chunk
Multi-part Section
```

---

## Build Vector Index

```powershell
python scripts/build_vector_index.py
```

当前预期：

```text
Embedding Shape
→ (104, 1024)

Qdrant Point Count
→ 104
```

---

## Run Frozen Retrieval Regression V1

```powershell
python scripts/run_retrieval_eval.py --dataset v1
```

用于验证：

```text
Corpus / Schema / Retrieval 改动
是否破坏旧能力
```

---

## Run Retrieval Capability Evaluation V2

```powershell
python scripts/run_retrieval_eval.py --dataset v2
```

用于评估当前 Corpus V2：

```text
Dense
BM25
Hybrid RRF
Hybrid + Rerank
```

以及 Failure Analysis。

---

## Run Final Retrieval Latency Benchmark

```powershell
python scripts/run_interleaved_retrieval_latency_benchmark.py
```

正式 Latency 采用：

```text
Query-level Interleaved Benchmark
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

# 37. FastAPI Run

启动：

```powershell
uvicorn enterprise_rag.api.app:create_app --factory --reload
```

启动完成后访问：

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

# 38. Example Ask

示例：

```json
{
  "query": "生成式人工智能服务处理训练数据需要遵守什么规定？",
  "role": "guest"
}
```

系统会：

```text
AccessContext
↓
ACL-aware Retrieval
↓
Hybrid Fusion
↓
Rerank
↓
Evidence Gate
↓
Evidence-Constrained Generation
↓
Citation Validation
```

最终回答只允许基于 Knowledge Base Evidence。

---

# 39. Structured Refusal

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

# 40. Repository Structure

当前主要目录：

```text
enterprise-rag-compliance1/
│
├── data/
│   ├── manifest/
│   │   └── documents.yaml
│   │
│   ├── raw/
│   │
│   ├── processed/
│   │   └── chunks.jsonl
│   │
│   └── eval/
│       ├── retrieval_eval_v1.jsonl
│       ├── retrieval_eval_v2.jsonl
│       └── results/
│
├── docs/
│   └── evaluation.md
│
├── scripts/
│   ├── build_chunks.py
│   ├── inspect_corpus.py
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
│       ├── reranking/
│       ├── retrieval/
│       ├── runtime/
│       ├── service/
│       └── vectorstore/
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

# 41. Testing

运行：

```powershell
pytest -v
```

项目当前测试覆盖：

```text
Manifest
HTML Loader
Normalization
Regulation Parser
Generic Section Parser
Regulation Chunking
Generic Section Chunking
Corpus Validation
Embedding
Retrieval
Hybrid Fusion
Rerank
ACL
Evidence Gate
Generation
Citation
FastAPI
Evaluation
Snapshot Persistence
Regression
```

每次：

```text
Schema Evolution
Corpus Expansion
Retrieval Modification
Prompt Modification
```

之后都应重新运行测试。

---

# 42. Current Design Conclusions

## Dense Retrieval

当前 Corpus V2 上：

```text
Dense
```

是最强单阶段 Retrieval Baseline。

尤其在：

```text
中文 Query
→ 英文 OWASP / FastAPI
```

场景下，多语言 Dense Retrieval 明显优于纯 lexical BM25。

当前不因为项目使用了 Hybrid Architecture，就刻意弱化 Dense Baseline。

---

## Hybrid Retrieval

Hybrid Retrieval 不一定天然优于 Dense。

Corpus V1 中已经观察到过 RRF 排名退化。

Corpus V2 中，随着英文 OWASP / FastAPI 加入：

```text
BM25 Cross-lingual Lexical Mismatch
```

进一步导致：

```text
R021
R022
R023
R027
```

出现：

```text
Dense Rank1
↓
RRF Rank2–4
```

因此：

> Hybrid 的价值必须通过真实 Corpus Evaluation 验证，而不是仅凭架构复杂度判断。

---

## Rerank

Rerank 能够修复多个 Fusion Failure：

```text
R021
R022
R023
R027
```

均恢复到：

```text
Gold Rank1
```

但当前 Hybrid + Rerank：

```text
质量仍未超过 Dense
+
Latency 成本明显更高
```

因此：

> 是否始终开启 Rerank，应由 Quality × Latency Evaluation 决定，而不是因为“Rerank 看起来更高级”。

---

## ACL

ACL 必须前置到 Retrieval Candidate Generation。

不能：

```text
先全库检索
再过滤
```

这是当前系统的重要安全设计原则。

另外：

```text
ACL Visibility
```

与：

```text
Answerability
```

不是同一个概念。

“知识库中有答案，但当前用户无权访问”不能简单标记成普通 Hard Negative。

后续将单独建立 ACL-specific Evaluation。

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
Minimal Sufficient Evidence Set
```

Gold 也不能根据当前模型输出反向修改。

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

另外应明确区分：

```text
Regression Benchmark
```

和：

```text
Current Capability Benchmark
```

避免 Corpus 扩展后失去历史对照。

---

# 43. Current Limitations

当前仍有明显限制：

1. 当前 Corpus 只有 5 篇文档、104 个 Chunk，距离目标 30–80 篇文档仍有较大差距；
2. 当前只引入 OWASP LLM01 Prompt Injection，尚未覆盖完整 OWASP Top 10 for LLM Applications；
3. 尚未加入 Qdrant 官方技术文档；
4. Retrieval Eval V2 只有 34 条，其中 Answerable Query 26 条，仍属于小型 Seed Evaluation；
5. 当前 Retrieval Eval V2 尚未拆分独立 Development Set / Held-out Test Set，因此没有基于该 Dataset 调整 RRF 权重或 Retrieval 参数；
6. 最新 Full-RAG Answer / Refusal / Citation Evaluation 仍基于 V1 的 20 条 Seed Cases，尚未完整扩展到 Retrieval Eval V2；
7. Prompt v1 / v2 目前均主要基于单次 Full-RAG Snapshot 进行比较；
8. 尚未完成独立 Answer Correctness 指标；
9. 尚未完成完整 Faithfulness Evaluation；
10. 尚未建立独立 ACL Unauthorized Retrieval Benchmark；
11. 尚未进行正式并发 Load Test；
12. API Role 目前只是 Demo Input，并非生产认证系统；
13. Evaluation Gold 仍主要依赖人工 Source Audit；
14. 当前 Generic Markdown-like Pipeline 面向文本型技术文档，尚未实现完整 Code-aware Chunking；
15. 当前结果不可解释为生产环境准确率、可靠性保证或 SLA。

---

# 44. Roadmap

后续计划：

```text
更多 OWASP LLM Top 10
↓
Qdrant 官方技术文档
↓
更多公开 / 模拟内部规范
↓
30–80 Documents
↓
50–100 Evaluation Queries
↓
Development / Held-out Eval Split
↓
ACL-specific Evaluation
↓
More Hard Negatives
↓
Cross-document QA
↓
Full-RAG Evaluation V2
↓
Faithfulness
↓
Answer Correctness
↓
UI / Demo
```

Retrieval 参数优化只有在：

```text
Development Set
+
Held-out Evaluation
```

建立之后再进行，避免直接对当前 V2 Benchmark 过拟合。

---

# 45. Project Status

当前已完成：

```text
✅ Structured Regulation Ingestion
✅ Heterogeneous Document Ingestion
✅ Generic Section Parser
✅ Section-aware Chunking
✅ Unified KnowledgeChunk Schema
✅ Corpus Validation
✅ Corpus Inspection
✅ OWASP LLM01 Ingestion
✅ FastAPI Dependencies Ingestion
✅ FastAPI Lifespan Ingestion
✅ Corpus V2 / 104 Chunks
✅ BGE-M3 Embedding
✅ Qdrant
✅ Qdrant Payload Metadata
✅ Dense Retrieval
✅ BM25 Retrieval
✅ RRF Hybrid Retrieval
✅ bge-reranker
✅ ACL-aware Retrieval
✅ ACL Pre-filter Candidate Generation
✅ Evidence Gate
✅ Evidence-Constrained Generation
✅ Structured Refusal
✅ Citation Validation
✅ FastAPI Runtime
✅ Frozen Retrieval Eval V1
✅ Retrieval Eval V2
✅ Role-aware Retrieval Evaluation
✅ Gold Integrity Check
✅ Retrieval Failure Inspector
✅ Retrieval Regression Evaluation
✅ Query-level Interleaved Latency Benchmark
✅ Answer / Refusal Evaluation V1
✅ Citation Gold Audit
✅ Evaluation Snapshot Persistence
✅ Prompt Regression Evaluation
```

仍在继续：

```text
🚧 More Corpus Expansion
🚧 More OWASP Documents
🚧 Qdrant Technical Documentation
🚧 ACL-specific Evaluation
🚧 Evaluation Dataset 50–100 Cases
🚧 Held-out Evaluation Set
🚧 Full-RAG Evaluation V2
🚧 Answer Correctness
🚧 Faithfulness
🚧 Demo UI
```

---

# 46. License / Data

当前知识库内容主要来自：

```text
公开法规
公开安全规范
公开技术文档
```

其中部分公开技术文档在项目中被人为赋予：

```text
developer
```

Access Level，仅用于模拟企业内部 ACL 场景。

项目仅用于：

```text
学习
研究
RAG 系统设计演示
```

不构成法律意见。