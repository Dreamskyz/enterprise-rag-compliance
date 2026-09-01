# Enterprise RAG Compliance

企业 AI 合规与应用规范助手。

面向企业内部 RAG / Agent 研发人员和管理员，基于公开法规与技术规范提供：

- 可追溯的知识库检索；
- ACL 权限过滤；
- 混合检索与精排；
- 基于证据的回答；
- 无充分依据时拒答；
- 确定性 Citation。

> 当前项目为个人完成的校招作品级 RAG 工程骨架，重点展示可解释的 Retrieval、ACL、Evidence Control、Generation 与 Evaluation 设计，而不是构建大规模生产平台。

---

## Status

🚧 Under development

当前已完成：

- 法规 HTML 数据清洗与结构化解析；
- Structure-aware Chunking；
- BGE-M3 Dense Retrieval；
- BM25 Sparse Retrieval；
- RRF Hybrid Retrieval；
- bge-reranker-v2-m3 精排；
- Retrieval-time ACL；
- Evidence Gate 实验；
- Evidence-Constrained Generation；
- Structured Refusal；
- Deterministic Citation；
- QueryService；
- FastAPI Runtime；
- `/health`；
- `/ready`；
- `/api/v1/retrieve`；
- `/api/v1/ask`。

后续计划：

- 扩充法规 / OWASP / 技术规范语料；
- 构建正式 Evaluation Dataset；
- 检索消融实验；
- Evidence Gate 阈值正式标定；
- RAG 回答质量与拒答能力评测；
- Streamlit Demo；
- 完善 README 实验表与架构图。

---

## Project Goal

项目目标不是构建“万能知识库”。

当前系统聚焦：

**企业 AI 合规与应用规范**

知识来源以真实公开文档为主，包括：

- 《生成式人工智能服务管理暂行办法》；
- 《互联网信息服务深度合成管理规定》；
- 后续计划加入 OWASP Top 10 for LLM Applications；
- 少量 FastAPI / Qdrant 官方文档作为内部技术规范样例。

核心产品原则：

> 无充分知识库依据时，不强行生成答案。

---

## Tech Stack

- Python 3.11
- FastAPI
- Qdrant
- BGE-M3
- BM25Okapi
- Jieba
- RRF
- bge-reranker-v2-m3
- SiliconFlow OpenAI-compatible API
- DeepSeek-V4-Flash
- PyTorch + CUDA
- Docker
- Pytest

当前本地开发环境：

- Windows
- RTX 4060 Laptop GPU
- 32 GB RAM

---

## Architecture

### Offline Knowledge Base Pipeline

```text
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
NormalizedDocument
        ↓
Regulation Parser
        ↓
Chapter / Article
        ↓
Structure-aware Chunker
        ↓
 KnowledgeChunk
        ↓
   chunks.jsonl
        ↓
 Data Validation
        ↓
 BGE-M3 Embedding
        ↓
      Qdrant
```

当前法规采用：

```text
1 Article = 1 Chunk
```

而不是固定字符长度切分。

对于法规类文档，条款本身具有明确语义边界，因此当前 V1 优先保留法律结构。

---

### Online Retrieval Pipeline

```text
User Query
    ↓
AccessContext
    ↓
ACL Authorized Search Space
    ↓
┌──────────────────────┐
│                      │
↓                      ↓
Dense Retrieval      BM25 Retrieval
BGE-M3 + Qdrant      Jieba + BM25Okapi
Top 20               Top 20
│                      │
└──────────┬───────────┘
           ↓
        RRF Fusion
         k = 60
           ↓
     Hybrid Top 20
           ↓
 bge-reranker-v2-m3
           ↓
     Reranked Top 5
```

当前系统没有直接比较 Dense Score 与 BM25 Score。

两种检索器的原始分数量纲不同，因此使用 rank-based Reciprocal Rank Fusion：

```text
RRF(d) = Σ 1 / (k + rank)
```

当前：

```text
k = 60
```

---

### Online RAG Query Pipeline

```text
HTTP Request
      ↓
FastAPI
      ↓
AccessContext
      ↓
ACL-aware Retrieval
      ↓
Dense + BM25
      ↓
RRF
      ↓
Reranker
      ↓
Coarse Relevance Gate
      ↓
┌─────────────────────────┐
│                         │
明显无关                  通过
│                         │
↓                         ↓
Programmatic Refusal   Evidence-Constrained
                       Generation
                           ↓
                   Evidence Sufficiency
                      /           \
                    足够           不足
                     ↓             ↓
                   Answer       Refusal
                     ↓
             Deterministic Citation
```

---

## Knowledge Chunk

当前核心知识单元为 `KnowledgeChunk`。

主要字段包括：

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

用于 Generation。

```text
retrieval_text
```

用于 Dense / BM25 Retrieval。

当前 `retrieval_text` 由：

```text
标题
+
章节
+
条款编号
+
正文
```

组成。

---

## Current Corpus

当前已完成两份真实法规的结构化处理：

1. 《生成式人工智能服务管理暂行办法》
2. 《互联网信息服务深度合成管理规定》

当前共生成：

```text
49 Knowledge Chunks
```

当前所有真实法规 Chunk 均为：

```text
access_level = public
```

ACL 的 developer/admin 场景使用专门构造的 synthetic mixed-access corpus 进行验证，避免篡改真实法规数据。

---

## Dense Retrieval

Embedding Model：

```text
BAAI/bge-m3
```

Embedding Dimension：

```text
1024
```

Vector Database：

```text
Qdrant
```

Distance：

```text
Cosine
```

Collection：

```text
compliance_chunks_v1
```

当前 Dense Retrieval：

```text
Query
  ↓
BGE-M3
  ↓
1024-d Vector
  ↓
Qdrant
  ↓
Dense Top K
```

---

## Sparse Retrieval

当前 Sparse Retrieval 使用：

```text
Jieba
+
BM25Okapi
```

BM25 使用 `retrieval_text` 建立索引。

当前 V1 暂未加入：

- 自定义停用词；
- 自定义领域词典；
- k1 / b 参数调优。

这些参数将在 Evaluation 阶段根据数据决定是否需要调整。

---

## Hybrid Retrieval

当前 Hybrid Retrieval：

```text
Dense Top 20
+
BM25 Top 20
        ↓
       RRF
      k = 60
        ↓
Hybrid Top 20
```

当前选择 RRF 的主要原因：

1. Dense 与 BM25 原始 Score 不可直接比较；
2. RRF 只依赖 Rank；
3. 实现简单且容易解释；
4. 适合作为可复现的 Hybrid Retrieval Baseline。

正式效果将在后续 Evaluation 中通过消融实验验证：

```text
Dense
vs
BM25
vs
Hybrid RRF
vs
Hybrid + Rerank
```

---

## Reranking

Reranker：

```text
BAAI/bge-reranker-v2-m3
```

当前流程：

```text
Hybrid Top 20
      ↓
Cross-Encoder Reranker
      ↓
Top 5
```

需要特别注意：

> Rerank raw score 是相关性信号，不是概率。

因此不能把：

```text
score > 0
```

简单理解成：

```text
“知识库一定可以回答”
```

---

## ACL

当前实现轻量级 RBAC：

```text
guest
    ↓
public

developer
    ↓
public + developer

admin
    ↓
public + developer + admin
```

### Dense ACL

Qdrant 使用 payload filter：

```text
Query
  ↓
ACL Filter
  ↓
Authorized Vector Search Space
  ↓
Dense Top K
```

也就是说 ACL 在向量 Candidate Generation **之前**生效。

### BM25 ACL

BM25 为不同角色建立授权 Corpus：

```text
Role
  ↓
Authorized Chunks
  ↓
BM25 Scoring
```

同样不是：

```text
全库检索
↓
结果后过滤
```

### ACL Experiment

使用 synthetic：

```text
public
developer
admin
```

三种访问级别 Chunk 进行了完整：

```text
Dense
+
BM25
+
RRF
+
Reranker
```

链路验证。

结果：

```text
guest unauthorized results      = 0
developer unauthorized results  = 0
admin unauthorized results      = 0
```

当前结论：

> ACL 应在 Retrieval Candidate Generation 前限制搜索空间，而不是在全库检索之后进行 post-filter。

当前 API 中 `role` 为了 Demo 由客户端显式传入。

生产环境中不应信任客户端自行声明的角色，而应由：

```text
JWT / SSO / Trusted Identity Claims
```

生成 `AccessContext`。

---

## Evidence Gate Experiment

项目最初尝试使用：

```text
Top1 Cross-Encoder Rerank Score
```

作为单一 Answerability Gate。

### Preliminary Result

在最初：

```text
8 Answerable
+
8 Unanswerable
```

样本中，两类出现明显分离：

```text
Answerable:
min = 5.9062
max = 7.7852
avg = 6.9038

Unanswerable:
min = -8.6641
max = 0.4089
avg = -5.1149
```

简单 Gap：

```text
5.4973
```

但该结果主要来自相对标准的问题表达。

---

### Hard Case Experiment

随后加入：

```text
Hard Positive
```

例如：

```text
数据标注质量评估
训练语料有哪些合规要求？
违法内容怎么处理？
```

以及：

```text
Hard Negative
```

例如：

```text
发现违法内容后必须在几小时内处理？
数据标注人员最低学历是什么？
深度合成服务必须使用哪种审核算法？
```

实验结果：

```text
Hard Positive:
min = -2.0898
max = 4.0469
avg = 1.6316

Hard Negative:
min = 1.4775
max = 4.8008
avg = 3.4961
```

出现明显 Score Overlap：

```text
Hard Gap = -6.8906
```

因此项目不再采用：

```text
Rerank Score
→ 单阈值
→ 直接决定是否能够回答
```

而调整为：

```text
Rerank Score
→ Coarse Relevance Gate
→ 过滤明显 OOD Query

然后：

Evidence-Constrained Generation
→ 判断 Evidence Sufficiency
→ Answer / Refusal
```

当前实验结论：

> Relevance ≠ Answerability。

这是当前项目 Evidence Control 设计的核心依据。

---

## Evidence-Constrained Generation

LLM：

```text
deepseek-ai/DeepSeek-V4-Flash
```

API：

```text
SiliconFlow OpenAI-compatible API
```

Generation Prompt 要求：

- 只能使用提供的 Evidence；
- 不允许补充 Evidence 之外的事实；
- Evidence 相关但不能回答具体问题时必须拒答；
- 不允许把“及时”等模糊描述推断成具体小时数；
- 只能引用程序提供的 Evidence ID；
- Evidence 不足时输出结构化拒答。

---

## Structured Refusal

例如问题：

```text
《生成式人工智能服务管理暂行办法》
规定发现违法内容后必须在几小时内处理？
```

Retrieval Top1 能正确命中：

```text
第十四条
```

且 rerank score 较高：

```text
≈ 4.543
```

但 Evidence 只规定：

```text
“及时采取停止生成、停止传输、消除等处置措施”
```

没有明确：

```text
具体几小时
```

因此最终结果为：

```text
gate_reason = passed

answerable = false
answer = null
citations = []
```

说明系统没有将高相关性错误等价为“足够回答”。

---

## Deterministic Citation

LLM 不直接决定真实法规来源。

Generation 前，程序为 Evidence 分配：

```text
E1
E2
E3
...
```

LLM 只能返回：

```json
{
  "citations": ["E1"]
}
```

Parser 会验证：

```text
E1 是否真实存在
```

然后程序再从内部 Evidence Mapping 获取：

```text
chunk_id
title
article_number
source_url
```

如果 LLM 返回：

```text
E99
```

但本次请求不存在该 Evidence：

```text
→ Parser Reject
```

因此 Citation Metadata 不依赖模型自由生成。

---

## QueryService

`QueryService` 是当前核心 Application Service。

它负责：

```text
Query
  ↓
AccessContext
  ↓
RerankedRetriever
  ↓
Coarse Relevance Gate
  ↓
EvidenceGroundedAnswerer
  ↓
QueryResult
```

FastAPI Router 不直接编排：

```text
Dense
BM25
RRF
Reranker
LLM
```

HTTP 层只负责：

```text
Request Validation
↓
AccessContext
↓
QueryService
↓
Response Mapping
```

---

## FastAPI Runtime

重型资源通过 FastAPI Application Lifespan 在服务启动阶段初始化一次：

```text
FastAPI Startup
       ↓
Read chunks.jsonl
       ↓
Load BGE-M3
       ↓
Build BM25
       ↓
Connect Dense Retrieval
       ↓
Load Reranker
       ↓
Create Evidence Gate
       ↓
Create LLM Client
       ↓
Create Grounded Answerer
       ↓
Create QueryService
       ↓
Runtime Ready
```

HTTP 请求复用已经初始化的 Runtime。

不会每次请求重新加载：

```text
BGE-M3
bge-reranker
BM25 Index
```

---

## API

### Health

```http
GET /health
```

用途：

```text
检查 FastAPI 服务进程是否存活。
```

示例：

```json
{
  "status": "ok",
  "service": "enterprise-rag-compliance",
  "version": "0.1.0"
}
```

---

### Readiness

```http
GET /ready
```

用途：

```text
检查 RAG Runtime 是否已经完成初始化。
```

示例：

```json
{
  "status": "ready",
  "runtime_ready": true,
  "retrieval_ready": true,
  "query_ready": true,
  "chunk_count": 49
}
```

---

### Retrieve

```http
POST /api/v1/retrieve
```

Request：

```json
{
  "query": "生成式人工智能服务处理训练数据需要遵守什么规定？",
  "role": "guest",
  "top_k": 5
}
```

功能：

```text
ACL-aware Dense
+
ACL-aware BM25
+
RRF
+
Reranker
```

该接口：

```text
不调用 LLM。
```

主要用于：

```text
Retrieval Debug
Evaluation
Observability
ACL Demo
```

---

### Ask

```http
POST /api/v1/ask
```

Request：

```json
{
  "query": "生成式人工智能服务处理训练数据需要遵守什么规定？",
  "role": "guest"
}
```

Response 示例：

```json
{
  "query": "生成式人工智能服务处理训练数据需要遵守什么规定？",
  "role": "guest",
  "answerable": true,
  "answer": "根据相关规定……",
  "reason": "E1 提供了明确依据。",
  "citations": [
    {
      "evidence_id": "E1",
      "chunk_id": "cn_genai_interim_2023__第七条",
      "title": "生成式人工智能服务管理暂行办法",
      "article_number": "第七条",
      "source_url": "..."
    }
  ],
  "retrieval_count": 5,
  "top_rerank_score": 7.1015625,
  "gate_reason": "passed"
}
```

---

## API Status Model

当前服务区分：

```text
Process Health
      ↓
GET /health
```

和：

```text
RAG Runtime Readiness
      ↓
GET /ready
```

以及：

```text
Retrieval Debug
      ↓
POST /api/v1/retrieve
```

和：

```text
Business QA
      ↓
POST /api/v1/ask
```

---

## Run Qdrant

启动：

```powershell
docker compose up -d
```

检查：

```powershell
docker compose ps
```

Qdrant Dashboard：

```text
http://127.0.0.1:6333/dashboard
```

---

## Run FastAPI

确保：

```text
Qdrant 已启动
.env 已配置
知识库 chunks.jsonl 已生成
Qdrant Collection 已构建
```

启动：

```powershell
python -m uvicorn enterprise_rag.api.app:app --log-level info
```

当前真实 Runtime 会在 Startup 阶段加载：

```text
BGE-M3
BM25
bge-reranker-v2-m3
QueryService
```

启动成功后会看到：

```text
Starting RAG runtime...

RAG runtime ready |
chunks=49 |
startup_ms=... |
llm_model=deepseek-ai/DeepSeek-V4-Flash

Application startup complete.
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

OpenAPI：

```text
http://127.0.0.1:8000/openapi.json
```

---

## Environment Variables

真实密钥只保存在：

```text
.env
```

示例：

```env
SILICONFLOW_API_KEY=your_api_key_here
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

`.env` 已加入 `.gitignore`。

仓库只提交：

```text
.env.example
```

禁止将真实 API Key 写入：

```text
Python source code
README
Git history
```

---

## Build Knowledge Chunks

```powershell
python scripts/build_chunks.py
```

验证：

```powershell
python scripts/validate_chunks.py
```

---

## Build Vector Index

确保 Qdrant 已启动，然后：

```powershell
python scripts/build_vector_index.py
```

---

## Tests

运行全部测试：

```powershell
pytest -v
```

API Unit Test 使用 Fake Retriever / Fake QueryService，不加载真实 GPU Runtime，也不调用真实 LLM。

真实 GPU / Qdrant / SiliconFlow 链路通过独立 Smoke Test 验证。

---

## Smoke Tests

Retrieval Runtime：

```powershell
python scripts/check_api_retrieval_runtime.py
```

完整 Ask Runtime：

```powershell
python scripts/check_api_ask_runtime.py
```

其他开发期检查脚本位于：

```text
scripts/
```

---

## Current Engineering Conclusions

### 1. Structure-aware Chunking

法规类文档优先根据：

```text
Chapter / Article
```

切分，而不是机械固定字符长度。

当前：

```text
1 Article = 1 Chunk
```

---

### 2. Hybrid Retrieval

Dense 与 BM25 原始分数量纲不同，因此当前使用 RRF 做 rank-level fusion，而不是直接线性相加原始 score。

---

### 3. ACL

ACL 必须在 Candidate Generation 前限制 Authorized Search Space。

不采用：

```text
全库 Retrieval
↓
Post-filter
```

当前 synthetic mixed-access 全链路实验未发现 unauthorized result。

---

### 4. Rerank Relevance Is Not Answerability

Hard Case 实验显示：

```text
Hard Positive
与
Hard Negative
```

的 rerank score 存在明显 overlap。

因此：

```text
Cross-Encoder relevance score
```

不能作为唯一 Answerability 判断。

---

### 5. Evidence Control

当前采用：

```text
Coarse Relevance Gate
+
Evidence-Constrained Generation
+
Output Validation
```

三层控制。

明显 OOD Query 可以程序化拒绝。

主题相关但 Evidence 缺少具体事实的问题，由 Evidence-Constrained Generation 继续执行 Structured Refusal。

---

### 6. Citation

Citation ID 由 LLM 选择，但真实：

```text
title
article
source_url
chunk_id
```

由程序根据 Evidence Mapping 确定。

模型不能自由生成最终 Citation Metadata。

---

### 7. Runtime

GPU 模型只在 FastAPI Application Startup 阶段初始化一次。

HTTP Request 复用共享 Runtime，避免每请求重复加载模型。

---

## Evaluation Plan

正式 Evaluation 将包含约：

```text
50–100 条 QA
```

并覆盖：

```text
Answerable
Hard Positive
Out-of-domain Unanswerable
In-domain Unsupported Hard Negative
```

计划评测：

```text
Recall@K
MRR
Faithfulness
Answer Correctness
Refusal Accuracy
Citation Accuracy
ACL Leakage Rate
Latency
```

Retrieval Ablation：

```text
Dense

vs

BM25

vs

Hybrid RRF

vs

Hybrid + Rerank
```

当前 README 中的 sanity / preliminary experiment 不视为正式 Benchmark。

---

## Repository

GitHub repository:

```text
enterprise-rag-compliance1
```

---

## License

当前项目用于个人学习展示。