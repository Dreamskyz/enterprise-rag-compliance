# Enterprise RAG Compliance

企业 AI 合规与应用规范助手。

面向企业内部 RAG / Agent 研发人员、管理员等用户，围绕生成式人工智能合规法规、AI 安全规范与内部技术规范，构建一个可检索、可引用、可拒答、可评测的企业知识库助手。

> 🚧 当前项目仍处于开发阶段。  
> 已完成基础工程、法规类文档 Ingestion Pipeline，以及 Dense + BM25 + RRF + Reranker 检索主链路。  
> ACL、Evidence Gate、LLM Generation、Citation、API、UI 与 Evaluation 正在后续阶段实现。

---

## Project Goals

本项目不是通用聊天机器人，而是一个面向企业 AI 合规与应用规范场景的 RAG 系统。

核心目标：

- 基于真实公开法规与技术规范构建知识库；
- 通过结构化 Ingestion 保留文档原始语义边界；
- 使用 Dense Retrieval 与 BM25 建立互补召回能力；
- 使用 RRF 融合不同检索通路；
- 使用 Cross-Encoder Reranker 提升候选精排质量；
- 回答尽量基于检索证据，降低无依据生成；
- 在缺乏充分检索证据时支持拒答；
- 后续支持 ACL 检索权限过滤；
- 支持 Retrieval / Generation 自动评测与消融实验；
- 核心模块保持可独立测试、解释和替换。

项目强调：

```text
不是：
一个 LangChain Chain 包住所有逻辑

而是：
Ingestion
Embedding
Dense Retrieval
BM25
RRF
Reranker
ACL
Evidence Gate
Generation
Evaluation

各层职责明确、可独立验证
```

---

## Current Progress

### ✅ 已完成

#### Project Bootstrap

- Python 3.11 + `src/` 工程结构
- `pyproject.toml` 项目与依赖管理
- `.env` / `.gitignore` 环境变量与密钥隔离
- Docker Compose 启动 Qdrant
- Docker Named Volume 数据持久化
- pytest 单元测试与回归测试

#### Data Ingestion

- Document Manifest 文档清单
- 国家网信办 HTML 正文抽取
- 文本 Normalizer
- 法规章节 / 条款结构解析
- Structure-aware Chunking
- `KnowledgeChunk` 标准数据模型
- `chunks.jsonl` 标准中间数据格式
- Chunk 数据质量校验

#### Embedding & Vector Index

- PyTorch + CUDA 本地 GPU 环境
- BGE-M3 本地 Dense Embedding
- FP16 GPU 推理
-真实 Chunk Token 长度分析
- Qdrant Collection
- Stable Point ID
- Vector Upsert
- Payload Metadata
- Vector / Chunk 对齐校验

#### Retrieval

- BGE-M3 + Qdrant Dense Retrieval
- Jieba + BM25 Sparse Retrieval
- Reciprocal Rank Fusion（RRF）
- `bge-reranker-v2-m3` Cross-Encoder 精排
- Dense / BM25 / RRF / Reranker 分层实现
- Brute-force Dense 与 Qdrant Dense 一致性验证
- Retrieval latency 本地观测
- RRF 与 Rerank 纯逻辑自动测试

### 🚧 下一阶段

- ACL-aware Retrieval
- Qdrant Payload Filter
- Evidence Gate
- LLM Generation
- Citation
- Refusal
- QueryService
- FastAPI API
- Streamlit UI
- Retrieval Evaluation
- Generation Evaluation
- Ablation Study

---

## Architecture

### Offline Knowledge Pipeline

当前已完成的离线知识库构建流程：

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
     BGE-M3
        ↓
 Dense Embedding
        ↓
      Qdrant
```

### Current Retrieval Pipeline

当前已完成的检索主链路：

```text
                         ┌─ BGE-M3 ─ Qdrant ─ Dense Top 20
Query ───────────────────┤
                         └─ Jieba ─ BM25 ───── BM25 Top 20
                                      ↓
                                     RRF
                                      ↓
                               Hybrid Top 20
                                      ↓
                           bge-reranker-v2-m3
                                      ↓
                                  Final Top 5
```

### V1 Target Architecture

完整 V1 目标架构：

```text
                              Streamlit
                                  │
                                  ▼
                               FastAPI
                                  │
                                  ▼
                             QueryService
                                  │
                         ACL / Role Context
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Dense Retrieval            BM25 Retrieval
              BGE-M3 + Qdrant
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                              RRF Fusion
                                  │
                                  ▼
                               Reranker
                                  │
                                  ▼
                            Evidence Gate
                           ┌──────┴──────┐
                           ▼             ▼
                         Accept        Refusal
                           │
                           ▼
                     Prompt Builder
                           │
                           ▼
                          LLM
                           │
                           ▼
                  Answer + Citation
```

设计原则：

```text
ACL
应在 Retrieval 阶段过滤

而不是：
先全局召回
再在结果末尾过滤
```

```text
Evidence Gate
负责判断证据是否足够

LLM Prompt
只是第二道约束

不依赖 LLM 自己决定是否应该拒答
```

---

## Data Sources

当前首批知识库文档：

- 《生成式人工智能服务管理暂行办法》
- 《互联网信息服务深度合成管理规定》

当前语料：

```text
2 篇公开法规
49 个法规级 Chunk
```

后续计划扩展：

- OWASP Top 10 for LLM / GenAI
- OWASP 必要英文原文
- FastAPI 官方文档选段
- Qdrant 官方文档选段
- 少量企业内部技术规范模拟数据

项目不计划构建“万能知识库”。

目标是围绕：

```text
AI 合规法规
+
LLM / RAG 安全规范
+
企业 AI 应用技术规范
```

形成一个领域边界明确的知识库。

---

## Document Manifest

项目使用：

```text
data/manifest/documents.yaml
```

统一管理知识库文档的文档级元数据。

Manifest 示例：

```yaml
document_id: cn_genai_interim_2023
title: 生成式人工智能服务管理暂行办法
document_type: regulation
language: zh-CN
version: "2023"
access_level: public
source_url: ...
local_path: ...
enabled: true
```

Manifest 主要维护：

- 文档身份
- 文档标题
- 数据来源
- 文档类型
- 文档版本
- 文档语言
- 权限级别
- 本地文件路径
- 是否启用

通过 Manifest 将原始文件与代码逻辑解耦，为后续：

```text
多文档
版本管理
重新建库
ACL
数据追踪
```

提供基础。

---

## Ingestion Pipeline

### 1. HTML Loading

从本地保存的官方 HTML 页面中读取原始内容。

针对国家网信办页面结构定位实际正文容器，并移除正文内部可能存在的：

```text
script
style
```

等非正文节点。

Loader 只负责：

```text
Source File
→ 正文文本
```

不负责 Chunk。

---

### 2. Text Normalization

对提取出的正文进行统一标准化，包括：

- 换行符统一
- 全角空格处理
- NBSP 处理
- 行首尾空白清理
- 多余空格压缩
- 空行清理

同时保留法规中的：

```text
章
条
（一）
（二）
```

等结构边界。

没有简单使用：

```python
" ".join(text.split())
```

将全文压成单行，因为这会破坏后续法规结构解析需要的信息。

---

### 3. Regulation Parsing

针对中文法规识别：

```text
章
↓
条
↓
条款正文
```

并转换为结构化法规对象。

Parser 会保留：

- 章节编号
- 章节标题
- 条款编号
- 条款正文

例如：

```text
第二章 技术发展与治理
↓
第七条
↓
条款正文
```

Parser 同时覆盖文件结束时最后一个条款 flush、正文误识别等边界问题。

---

### 4. Structure-aware Chunking

法规类文档没有直接采用固定字符窗口切分。

当前策略：

```text
法规全文
   ↓
识别“章”
   ↓
识别“条”
   ↓
一条法规 ≈ 一个 Chunk
```

原因：

> 法规中的“条”本身就是天然且稳定的法律语义边界。

相比固定字符窗口，可以减少一个完整法律条款被从中间截断的问题。

---

### 5. Data Validation

Chunk 构建完成后，在进入 Embedding / Vector Index 之前执行独立数据质量校验。

整体：

```text
Parser
   ↓
Chunker
   ↓
chunks.jsonl
   ↓
Validator
   ↓
Embedding
```

避免明显错误数据直接进入向量数据库。

---

## Chunking Strategy

### Current Strategy

当前：

```text
1 Article
≈
1 KnowledgeChunk
```

暂时不使用：

```text
固定 500 字
+
Overlap
```

作为法规语料的主切分方式。

### First-batch Character Statistics

| 文档 | Chunk 数 | 最短字符数 | 最长字符数 | 平均字符数 |
| --- | ---: | ---: | ---: | ---: |
| 《生成式人工智能服务管理暂行办法》 | 24 | 18 | 401 | 127.79 |
| 《互联网信息服务深度合成管理规定》 | 25 | 18 | 491 | 130.04 |
| **合计** | **49** | - | - | - |

根据当前语料：

> 以“条”为 Chunk 边界能够较好保持完整法律语义，当前暂时没有必要进行条内二次切分。

对于较短条款，也不会仅因为字符少而强制与相邻条款合并。

例如：

```text
本办法自某年某月某日起施行。
```

虽然很短，但它本身就是完整且具有独立法律含义的证据。

后续如果出现数千字超长条款，再根据真实数据加入：

```text
Article
↓
Secondary Split
```

而不是提前增加不必要复杂度。

---

## Chunk Data Model

标准 `KnowledgeChunk` 当前包含：

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

字段说明：

| 字段 | 作用 |
| --- | --- |
| `chunk_id` | Chunk 稳定业务标识 |
| `document_id` | 所属文档 ID |
| `title` | 文档标题 |
| `document_type` | 文档类型 |
| `language` | 文档语言 |
| `version` | 文档版本 |
| `chapter_number` | 所属章节编号 |
| `chapter_title` | 所属章节标题 |
| `article_number` | 法规条款编号 |
| `content` | 原始条款正文 |
| `retrieval_text` | 用于 Embedding / BM25 / Reranker 的增强检索文本 |
| `source_url` | 原始数据来源 |
| `access_level` | ACL 检索过滤依据 |
| `chunk_index` | Chunk 在文档中的顺序 |
| `content_hash` | 内容一致性与增量更新判断依据 |

---

## content vs retrieval_text

项目明确区分：

```text
content
```

与：

```text
retrieval_text
```

`content`：

> 保存干净的原始条款正文，用于 Citation、展示和证据返回。

`retrieval_text`：

```text
文档标题
+
章节信息
+
条款编号
+
正文
```

例如：

```text
生成式人工智能服务管理暂行办法
第二章 技术发展与治理
第七条
生成式人工智能服务提供者……
```

当前以下组件统一使用 `retrieval_text`：

```text
BGE-M3
BM25
bge-reranker-v2-m3
```

从而减少不同检索阶段使用不同文本表示带来的额外变量。

---

## Chunk ID

法规类 Chunk 当前采用可读且稳定的业务 ID，例如：

```text
cn_genai_interim_2023__第一条
cn_genai_interim_2023__第七条
```

构成：

```text
document_id
+
article_number
```

业务 `chunk_id` 与 Qdrant Point ID 分离。

Qdrant Point ID 使用基于 `chunk_id` 生成的稳定 UUID5：

```text
chunk_id
↓
UUID5
↓
Qdrant Point ID
```

这样重复执行 Upsert 时，同一个 Chunk 会得到相同 Point ID，避免随机 UUID 导致重复数据。

---

## Intermediate Data

标准 Chunk 保存为：

```text
data/processed/chunks.jsonl
```

采用 JSONL：

```text
一行
=
一个 KnowledgeChunk
```

整体：

```text
Raw Documents
      ↓
Ingestion Pipeline
      ↓
chunks.jsonl
      ↓
Embedding
      ↓
Vector Index
```

`chunks.jsonl` 是可复用的标准中间产物。

Qdrant 被视为：

```text
Derived Index
```

而不是原始数据唯一来源。

即使 Qdrant Volume 被删除，仍可以：

```text
chunks.jsonl
↓
BGE-M3
↓
Vector Upsert
↓
重新构建 Qdrant
```

---

## Data Quality Validation

在进入 Embedding 与 Vector Index 前，`chunks.jsonl` 会经过独立数据质量校验。

当前检查包括：

- `chunk_id` 唯一性
- 核心字段不能为空
- `access_level` 是否合法
- `retrieval_text` 是否包含必要上下文
- `content_hash` 是否与正文一致
- 同一文档内 `chunk_index` 是否连续

运行：

```bash
python scripts/validate_chunks.py
```

当前：

```text
Chunk 总数：49

数据质量校验通过
```

---

# Retrieval

## BGE-M3 Dense Embedding

当前 Dense Embedding 模型：

```text
BAAI/bge-m3
```

运行方式：

```text
Framework:
FlagEmbedding

Device:
cuda:0

Precision:
FP16
```

当前实测 Dense Vector Dimension：

```text
1024
```

模型基础 sanity check：

```text
Query:
训练数据需要满足什么要求？

相关文本：
生成式人工智能服务提供者应当依法处理训练数据。

Similarity:
0.6816

无关文本：
今天天气很好，我准备出去散步。

Similarity:
0.3560
```

该实验用于验证：

```text
语义相关文本
在 Dense Embedding Space 中
具有更高相似度
```

不作为正式 Retrieval Evaluation。

---

## Chunk Token Analysis

使用 BGE-M3 Tokenizer 对实际 `retrieval_text` 统计：

| 指标 | 当前结果 |
| --- | ---: |
| Chunk 数量 | 49 |
| Minimum Tokens | 31 |
| Maximum Tokens | 327 |
| Average Tokens | 94.94 |
| Chunks over 512 | 0 |

最长 Chunk：

```text
chunk_id:
cn_deep_synthesis_2022__第二十三条

Token Count:
327
```

因此当前法规语料下：

```text
max_length = 512
```

能够完整覆盖全部 `retrieval_text`。

当前没有发生 Embedding 输入截断。

该参数来自实际语料统计，而不是简单使用 BGE-M3 支持的最大上下文长度。

---

## Qdrant Vector Index

当前 Collection：

```text
compliance_chunks_v1
```

当前 Schema：

```text
Point Count:
49

Vector Size:
1024

Distance:
Cosine
```

一个 Qdrant Point 对应：

```text
1 KnowledgeChunk
+
1 Dense Vector
+
Payload Metadata
```

Payload 当前包含：

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
access_level
```

将在后续 ACL-aware Retrieval 中作为 Qdrant Payload Filter 的基础字段。

---

## Dense Retrieval Sanity Check

测试 Query：

```text
生成式人工智能服务处理训练数据需要遵守什么规定？
```

先对全部 49 Chunk 进行 brute-force Dense Retrieval：

```text
Query Embedding
      ↓
document_vectors @ query_vector
      ↓
Top-K
```

再通过 Qdrant 执行相同 Dense Retrieval。

两者 Top-5 排序保持一致：

| Rank | Chunk | Brute-force Score | Qdrant Score |
| ---: | --- | ---: | ---: |
| 1 | 《生成式人工智能服务管理暂行办法》第七条 | 0.8208 | 0.8211 |
| 2 | 《生成式人工智能服务管理暂行办法》第四条 | 0.7417 | 0.7418 |
| 3 | 《生成式人工智能服务管理暂行办法》第十九条 | 0.7246 | 0.7250 |
| 4 | 《生成式人工智能服务管理暂行办法》第二十三条 | 0.6997 | 0.6998 |
| 5 | 《生成式人工智能服务管理暂行办法》第八条 | 0.6982 | 0.6985 |

该实验主要验证：

```text
Chunk
↓
retrieval_text
↓
BGE-M3
↓
Vector
↓
Point Alignment
↓
Qdrant Upsert
↓
Qdrant Search
```

整条数据链路没有明显错位。

> 该结果属于 sanity check，不代表正式 Retrieval Accuracy。

---

## BM25 Sparse Retrieval

当前 Sparse / Lexical Retrieval 使用：

```text
Jieba
+
rank-bm25
+
BM25Okapi
```

中文文本首先通过 Jieba 精确模式分词：

```text
Chinese Text
↓
Jieba Tokenization
↓
BM25
```

当前 V1 暂时：

```text
不使用复杂停用词表
不自定义 k1 / b
不引入自定义专业词典
```

先保留清晰 baseline，后续通过 Evaluation 判断优化是否真实有效。

---

## Dense vs BM25

对于 Query：

```text
生成式人工智能服务处理训练数据需要遵守什么规定？
```

Dense：

```text
《生成式人工智能服务管理暂行办法》第七条

Dense Rank = 1
```

BM25：

```text
《生成式人工智能服务管理暂行办法》第七条

BM25 Rank = 2
```

BM25 Rank 1：

```text
《互联网信息服务深度合成管理规定》第十四条
```

该条款中高频出现：

```text
训练数据
训练数据管理
训练数据安全
```

等词面高度匹配内容，因此 BM25 排名很高。

当前观察：

```text
Dense Retrieval
→ 更擅长语义相似、自然语言表达和同义匹配

BM25 Retrieval
→ 更擅长关键词、专有术语和固定表述匹配
```

对于关键词 Query：

```text
数据标注质量评估
```

BM25 将：

```text
《生成式人工智能服务管理暂行办法》第八条
```

排在 Top 1。

正文直接包含：

```text
开展数据标注质量评估
```

说明 BM25 对法规固定词面表达具有明显优势。

---

## RRF Hybrid Retrieval

Dense Cosine Score 与 BM25 Score 不属于同一数值空间。

例如：

```text
Dense:
0.x

BM25:
数个到十几个
```

因此系统不执行：

```text
dense_score + bm25_score
```

而使用：

```text
Reciprocal Rank Fusion
```

公式：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

当前 baseline：

```text
RRF k = 60
```

对于 Query：

```text
生成式人工智能服务处理训练数据需要遵守什么规定？
```

核心证据：

```text
《生成式人工智能服务管理暂行办法》第七条

Dense Rank = 1
BM25 Rank = 2
Hybrid Rank = 1
```

BM25 单独 Rank 1 的：

```text
《互联网信息服务深度合成管理规定》第十四条
```

最终：

```text
Dense Rank = 19
BM25 Rank = 1
Hybrid Rank = 4
```

当前结果表明：

> RRF 能够保留 BM25 的关键词召回能力，同时结合 Dense Retrieval 的语义排名信息，避免单一路检索结果直接支配最终候选顺序。

---

## Reranking

当前 Reranker：

```text
BAAI/bge-reranker-v2-m3
```

使用：

```text
FlagReranker
+
CUDA
+
FP16
```

整体：

```text
Dense Top 20
        \
         → RRF Top 20
        /
BM25 Top 20
        ↓
bge-reranker-v2-m3
        ↓
Final Top 5
```

Reranker 输入：

```text
Query
+
Candidate retrieval_text
```

与 Dense 不同，Reranker 使用 Cross-Encoder 形式直接联合建模 Query 和 Candidate。

---

## Reranking Example

Query：

```text
生成式人工智能服务处理训练数据需要遵守什么规定？
```

核心证据：

```text
《生成式人工智能服务管理暂行办法》第七条

Dense Rank  = 1
BM25 Rank   = 2
RRF Rank    = 1
Rerank Rank = 1

Rerank Score = 7.1016
```

Query：

```text
数据标注质量评估
```

核心证据：

```text
《生成式人工智能服务管理暂行办法》第八条

Dense Rank  = 1
BM25 Rank   = 1
RRF Rank    = 1
Rerank Rank = 1

Rerank Score = 2.4219
```

同时其余候选出现：

```text
Top 2: -4.1719
Top 3: -5.7383
Top 4: -5.8633
```

说明 Cross-Encoder 可以进一步拉开核心证据与弱相关候选之间的相关性差距。

需要注意：

- 当前使用 `normalize=False`
- Rerank Score 是模型 Raw Relevance Score
- Raw Score 可以为负
- Raw Score 不是概率
- 不同 Query 的 Raw Score 不应直接进行数值横向比较
- Evidence Gate 阈值后续必须通过评测数据校准

---

## Retrieval Baseline Parameters

当前 V1 Retrieval baseline：

| 参数 | 当前值 |
| --- | ---: |
| Dense Top-K | 20 |
| BM25 Top-K | 20 |
| RRF k | 60 |
| Hybrid Candidate Top-K | 20 |
| Rerank Final Top-K | 5 |
| BGE-M3 Dense Dimension | 1024 |
| BGE-M3 max_length | 512 |
| Embedding batch_size | 8 |
| Embedding Precision | FP16 |
| Reranker Precision | FP16 |
| Device | CUDA / RTX 4060 Laptop |

这些参数目前属于：

```text
V1 Baseline
```

最终值将由后续 Evaluation 与消融实验决定，而不是仅凭经验固定。

---

## Local GPU Observation

当前开发机器：

```text
GPU:
NVIDIA GeForce RTX 4060 Laptop GPU

Dedicated VRAM:
8 GB
```

BGE-M3 与 `bge-reranker-v2-m3` 同时加载，并运行：

```text
Dense Top 20
+
BM25 Top 20
+
RRF Top 20
+
Reranker 20 → 5
```

本地观察峰值专用显存约：

```text
3.3 GB
```

未发生 CUDA OOM。

> 该数据仅为当前开发环境中的 observed peak VRAM，不代表系统最低显存要求。

---

## Local Latency Observation

测试 Query：

```text
生成式人工智能服务处理训练数据需要遵守什么规定？
```

单次本地观测：

| Stage | Observed Latency |
| --- | ---: |
| Initialization | 16068.51 ms |
| Dense Retrieval | 633.95 ms |
| BM25 Retrieval | 0.54 ms |
| RRF Fusion | 0.05 ms |
| Reranker | 1374.50 ms |
| Total Query Pipeline | 2009.04 ms |

其中：

```text
Initialization
```

包括：

- BGE-M3 加载
- Jieba / BM25 初始化
- bge-reranker-v2-m3 加载

模型在正式服务中应在应用启动阶段初始化，而不是每个 Query 重新加载。

当前在线检索阶段的主要成本来自：

```text
1. Reranker
2. Dense Query Embedding + Qdrant Search
```

BM25 与 RRF 在当前 49 Chunk 规模下耗时很低。

当前数据：

> 仅为开发环境中的单次 observation，不属于正式性能 benchmark。

后续 Evaluation 阶段将统计：

```text
mean latency
p50 latency
p95 latency
```

并比较：

```text
Dense
BM25
Hybrid
Hybrid + Rerank
```

之间的效果 / 延迟权衡。

---

## Current Retrieval Conclusions

当前阶段已经得到以下工程结论：

1. BGE-M3 Dense Embedding 在当前运行环境中实测为 1024 维。
2. 当前首批法规 `retrieval_text` Token 长度全部低于 512，因此 `max_length=512` 不产生截断。
3. Brute-force Dense Retrieval 与 Qdrant Dense Retrieval 在基准 Query 上的 Top-5 排序保持一致。
4. Dense 与 BM25 在法规检索场景中表现出明显互补特征。
5. Dense Cosine Score 与 BM25 Score 不同量纲，不能直接相加。
6. Hybrid Retrieval 当前采用基于排名的 RRF。
7. RRF 能同时利用 Dense 的语义匹配与 BM25 的关键词召回能力。
8. `bge-reranker-v2-m3` 可以对 Hybrid Candidate 进行 Cross-Encoder 二阶段精排。
9. 当前 RTX 4060 Laptop 8GB 环境可以同时运行 BGE-M3 与 Reranker。
10. 当前本地观察峰值专用显存约 3.3GB。
11. 当前单次 Retrieval Pipeline 本地观测约为 2 秒，其中 Reranker 是主要延迟来源。
12. 当前结果均属于开发阶段 sanity check，最终 Recall@K、MRR 与 Rerank 收益仍需要 Evaluation Dataset 验证。

---

## Tests

项目使用：

```text
pytest
```

对核心数据处理与 Retrieval 逻辑进行自动化测试。

运行：

```bash
pytest -v
```

当前全量 pytest 已通过。

### Current Coverage

#### Ingestion

- 法规章节结构解析
- 法规条款结构解析
- 文件结束时最后一条不会丢失
- 正文中类似“第X条……”的普通文本不会误识别为新条款
- Structure-aware Chunk 构建
- `content_hash` 稳定性
- 正文变化后 `content_hash` 变化
- 重复 `chunk_id` 检测
- 非法 `access_level` 检测

#### RRF

- 同时出现在 Dense / BM25 中的候选获得双路排名贡献
- RRF 使用 Rank 而不是直接使用 Raw Score
- 非法 `top_k` 检测
- 非法 `rrf_k` 检测

#### Reranked Retrieval

- Reranker 能重新排序 Hybrid Candidate
- 精排后保留原始 RRF Rank
- 非法最终 `top_k` 检测

需要真实：

```text
GPU
BGE-M3
Qdrant
Reranker
```

的检查暂时通过 `scripts/check_*.py` 作为 Integration / Sanity Check，而不放入默认单元测试中，避免普通 `pytest` 每次加载大型模型。

---

## Regression Test

开发过程中，自动测试曾发现法规 Parser 的真实边界问题。

测试文本：

```text
第一条
第一条内容。
第二条
这是第二条内容。
```

早期正则可能将：

```text
第一条内容。
```

错误识别成新的条款开头：

```text
第一条
```

导致产生重复条款。

问题修复后增加对应 Regression Test，避免未来修改 Parser 时重新引入该问题。

这体现了当前项目的数据处理验证原则：

```text
真实数据人工检查
        +
自动化边界测试
        ↓
提高 Ingestion Pipeline 稳定性
```

---

## Development Scripts

当前开发阶段包含若干独立验证脚本。

### Ingestion

```text
scripts/inspect_manifest.py
scripts/inspect_document.py
scripts/inspect_regulation.py
scripts/inspect_chunks.py
scripts/build_chunks.py
scripts/validate_chunks.py
```

### GPU / Embedding

```text
scripts/check_torch_cuda.py
scripts/check_bge_m3.py
scripts/inspect_bge_chunks.py
scripts/check_embedding_service.py
```

### Vector Index

```text
scripts/build_vector_index.py
```

### Retrieval

```text
scripts/check_dense_retriever.py
scripts/check_bm25_retriever.py
scripts/check_hybrid_retriever.py
scripts/check_bge_reranker.py
scripts/check_reranked_retrieval.py
scripts/check_retrieval_latency.py
```

这些脚本主要用于：

```text
技术假设验证
+
Integration Sanity Check
+
问题定位
```

而不是作为最终用户接口。

---

## Project Structure

当前主要目录：

```text
enterprise-rag-compliance/
│
├── data/
│   ├── manifest/
│   │   └── documents.yaml
│   ├── raw/
│   └── processed/
│       └── chunks.jsonl
│
├── scripts/
│   ├── build_chunks.py
│   ├── validate_chunks.py
│   ├── build_vector_index.py
│   ├── check_torch_cuda.py
│   ├── check_bge_m3.py
│   ├── inspect_bge_chunks.py
│   ├── check_embedding_service.py
│   ├── check_dense_retriever.py
│   ├── check_bm25_retriever.py
│   ├── check_hybrid_retriever.py
│   ├── check_bge_reranker.py
│   ├── check_reranked_retrieval.py
│   └── check_retrieval_latency.py
│
├── src/
│   └── enterprise_rag/
│       │
│       ├── ingestion/
│       │   ├── loaders/
│       │   ├── manifest.py
│       │   ├── normalizer.py
│       │   ├── models.py
│       │   ├── regulation_parser.py
│       │   ├── document_builder.py
│       │   ├── chunker.py
│       │   ├── chunk_store.py
│       │   └── validator.py
│       │
│       ├── embeddings/
│       │   └── bge_m3.py
│       │
│       ├── vectorstore/
│       │   └── qdrant_store.py
│       │
│       ├── retrieval/
│       │   ├── config.py
│       │   ├── models.py
│       │   ├── dense.py
│       │   ├── bm25.py
│       │   ├── rrf.py
│       │   ├── hybrid.py
│       │   └── reranked.py
│       │
│       └── reranking/
│           └── bge_reranker.py
│
├── tests/
│   ├── fixtures/
│   ├── test_chunker.py
│   ├── test_regulation_parser.py
│   ├── test_validator.py
│   ├── test_rrf.py
│   └── test_reranked_retriever.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

> 项目目录仍会随着 ACL、Generation、API 与 Evaluation 模块继续扩展。

---

## Current Tech Stack

当前已经实际使用：

### Language / Runtime

- Python 3.11

### Data Processing

- BeautifulSoup
- PyYAML

### Machine Learning

- PyTorch
- CUDA
- FlagEmbedding
- BGE-M3
- bge-reranker-v2-m3
- NumPy

### Retrieval

- Qdrant
- qdrant-client
- Jieba
- rank-bm25
- BM25Okapi
- Reciprocal Rank Fusion

### Infrastructure

- Docker
- Docker Compose

### Testing

- pytest

---

## Planned Tech Stack

后续 V1 计划接入：

- FastAPI
- Streamlit
- LangChain
- SiliconFlow OpenAI-compatible API
- DeepSeek-V4-Flash
- ACL Payload Filtering
- Evidence Gate
- Citation Formatter
- Retrieval Evaluation
- Generation Evaluation

LangChain 在本项目中计划作为：

```text
组件生态
```

而不是：

```text
整个 RAG 架构唯一控制器
```

核心 Retrieval / Fusion / Rerank / Gate 逻辑仍保持显式实现和独立测试能力。

---

## Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd enterprise-rag-compliance
```

### 2. Create Virtual Environment

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Project

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

对于 NVIDIA GPU 环境，请根据 PyTorch 官方安装方式确保当前虚拟环境中的 PyTorch 可以访问 CUDA。

检查：

```powershell
python scripts/check_torch_cuda.py
```

### 4. Start Qdrant

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

### 5. Build Chunks

```powershell
python scripts/build_chunks.py
```

### 6. Validate Chunks

```powershell
python scripts/validate_chunks.py
```

### 7. Build Vector Index

```powershell
python scripts/build_vector_index.py
```

### 8. Run Tests

```powershell
pytest -v
```

---

## Security

项目不会将真实 API Key 写入仓库。

本地配置：

```text
.env
```

公开模板：

```text
.env.example
```

`.env` 已加入 `.gitignore`。

后续 LLM API Key、外部服务 Token 等均通过环境变量读取。

例如未来：

```text
SILICONFLOW_API_KEY=...
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

真实密钥不会提交 GitHub。

---

## Development Roadmap

### Phase 1 — Project Bootstrap

- [x] Python Project Structure
- [x] `pyproject.toml`
- [x] Environment Configuration
- [x] Docker Compose
- [x] Qdrant

### Phase 2 — Data Ingestion

- [x] Document Manifest
- [x] HTML Loader
- [x] Article Extraction
- [x] Text Normalizer
- [x] Normalized Document
- [x] Regulation Parser
- [x] Structure-aware Chunking
- [x] KnowledgeChunk
- [x] JSONL Output
- [x] Data Validation
- [x] Unit / Regression Tests

### Phase 3 — Embedding & Indexing

- [x] PyTorch + CUDA
- [x] BGE-M3 Embedding
- [x] Embedding Batch Pipeline
- [x] Token Length Analysis
- [x] Qdrant Collection
- [x] Stable Point ID
- [x] Vector Upsert
- [x] Payload Metadata
- [x] Index Validation

### Phase 4 — Retrieval

- [x] Dense Retrieval
- [x] BM25 Retrieval
- [x] RRF Fusion
- [x] Reranker
- [x] Retrieval Sanity Check
- [x] Local Latency Observation
- [ ] ACL Filtering

### Phase 5 — Generation

- [ ] QueryService
- [ ] Evidence Gate
- [ ] Prompt Construction
- [ ] LLM Generation
- [ ] Citation
- [ ] Refusal

### Phase 6 — Application

- [ ] FastAPI
- [ ] `/health`
- [ ] `/api/v1/retrieve`
- [ ] `/api/v1/ask`
- [ ] Streamlit
- [ ] End-to-End Demo

### Phase 7 — Evaluation

- [ ] Evaluation Dataset
- [ ] Dense Recall@K
- [ ] BM25 Recall@K
- [ ] Hybrid Recall@K
- [ ] MRR
- [ ] Rerank Comparison
- [ ] Latency Comparison
- [ ] Citation Accuracy
- [ ] Faithfulness
- [ ] Answer Correctness
- [ ] Refusal Accuracy
- [ ] ACL Leakage Test
- [ ] Ablation Study

---

## Evaluation Plan

后续 Retrieval Evaluation 将至少对比：

```text
Dense Only
vs
BM25 Only
vs
Dense + BM25 + RRF
vs
Dense + BM25 + RRF + Reranker
```

主要指标：

```text
Recall@K
MRR
Latency
```

Generation 阶段计划评测：

```text
Faithfulness
Answer Correctness
Citation Accuracy
Refusal Accuracy
```

ACL 阶段需要验证：

> 未授权 Chunk 不进入候选集合，而不是召回后再过滤。

最终希望形成类似：

| Method | Recall@5 | Recall@20 | MRR | Latency |
| --- | ---: | ---: | ---: | ---: |
| Dense | TBD | TBD | TBD | TBD |
| BM25 | TBD | TBD | TBD | TBD |
| Hybrid RRF | TBD | TBD | TBD | TBD |
| Hybrid + Rerank | TBD | TBD | TBD | TBD |

在正式 Evaluation 完成前，README 中现有 Retrieval 结果均明确标记为：

```text
sanity check
或
local observation
```

避免把单条 Query 演示结果误写成正式评测指标。

---

## Current Milestone

当前已经完成：

```text
Raw Regulation
      ↓
HTML Extraction
      ↓
Normalization
      ↓
Structure Parsing
      ↓
Structure-aware Chunking
      ↓
KnowledgeChunk
      ↓
chunks.jsonl
      ↓
Data Validation
      ↓
BGE-M3 Embedding
      ↓
Qdrant Vector Index
      ↓
Dense Retrieval ─────────┐
                         ├─ RRF
BM25 Retrieval ──────────┘
                           ↓
                        Top 20
                           ↓
                 bge-reranker-v2-m3
                           ↓
                       Final Top 5
```

当前已经验证：

```text
2 篇公开法规
49 个法规 Chunk

BGE-M3 Dense Dimension = 1024

Dense + Qdrant
BM25
RRF
Reranker

全部能够在本地完整运行
```

当前 Retrieval V1 主干已经完成。

下一阶段：

```text
User Role
   ↓
ACL-aware Retrieval
   ↓
Hybrid Retrieval
   ↓
Reranker
   ↓
Evidence Gate
   ↓
LLM
   ↓
Answer / Refusal
   ↓
Citation
```

---

## Current Engineering Conclusions

截至当前阶段：

1. 法规类文档使用“条”作为主要 Chunk 边界能够较好保持完整法律语义。
2. `content` 与 `retrieval_text` 分离，兼顾干净证据展示和检索上下文增强。
3. `chunks.jsonl` 作为标准中间产物，使 Ingestion 与 Embedding / Vector Index 解耦。
4. BGE-M3 Dense Embedding 当前实测为 1024 维。
5. 当前法规 Chunk Token 长度全部低于 512，无输入截断。
6. Brute-force Dense 与 Qdrant Dense 的基准 Top-5 排序保持一致。
7. Dense 与 BM25 在法规知识库中具有互补特征。
8. Dense 与 BM25 原始分数不可直接相加，因此 Hybrid 使用 RRF。
9. RRF 能综合语义排名与关键词排名。
10. Reranker 可以进一步对 Hybrid Candidate 做 Query-Passage Cross-Encoder 精排。
11. 在当前 RTX 4060 Laptop 8GB 环境下，BGE-M3 与 Reranker 可以同时运行。
12. 当前观察峰值专用显存约为 3.3GB。
13. 当前单次 Retrieval Pipeline 本地观测约为 2 秒，Reranker 是主要耗时组件。
14. 当前实验仍属于 sanity check，正式 Retrieval 收益必须通过 Evaluation Dataset 和消融实验量化。
15. 后续 ACL 必须在检索阶段进行过滤，目标是未经授权的 Chunk 从源头不进入候选集合。
16. Evidence Gate 阈值不会凭单个 Rerank Score 人工猜测，而会通过 answerable / unanswerable 评测集校准。

---

## License / Notice

本项目主要使用公开法规与官方技术文档作为知识库数据来源。

项目用于：

- 学习
- 技术研究
- 工程实践
- RAG 系统设计展示

本项目不构成法律意见。

如用于实际企业环境，应进一步完成：

```text
数据授权
安全审查
权限体系
日志审计
模型与 Prompt 安全测试
正式评测
```

等生产级工程要求。