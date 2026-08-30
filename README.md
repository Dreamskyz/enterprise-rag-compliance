# Enterprise RAG Compliance

企业 AI 合规与应用规范助手。

面向企业内部 RAG / Agent 研发人员、管理员等用户，
围绕生成式人工智能合规法规、AI 安全规范与内部技术规范，
构建一个可检索、可引用、可拒答、可评测的企业知识库助手。

> 🚧 当前项目仍处于开发阶段，现已完成基础工程与法规类文档 Ingestion Pipeline。

---

## Project Goals

本项目不是通用聊天机器人，而是一个面向企业 AI 合规与应用规范场景的 RAG 系统。

核心目标：

- 基于真实公开法规与技术规范构建知识库；
- 回答尽量基于检索证据，降低无依据生成；
- 在缺乏充分检索证据时支持拒答；
- 支持后续接入 Hybrid Retrieval、Rerank、ACL 与自动评测；
- 核心模块保持可独立测试、解释和替换。

---

## Current Progress

### ✅ 已完成

- Python 3.11 + `src/` 工程结构
- `pyproject.toml` 项目与依赖管理
- `.env` / `.gitignore` 环境变量与密钥隔离
- Docker Compose 启动 Qdrant
- Docker Named Volume 数据持久化
- Document Manifest 文档清单
- 国家网信办 HTML 正文抽取
- 文本 Normalizer
- 法规章节 / 条款结构解析
- Structure-aware Chunking
- `KnowledgeChunk` 数据模型
- `chunks.jsonl` 标准中间数据格式
- Chunk 数据质量校验
- pytest 单元测试与回归测试

### 🚧 开发中 / 计划实现

- BGE-M3 本地 Embedding
- Qdrant Vector Index
- Dense Retrieval
- BM25 Sparse Retrieval
- RRF Hybrid Retrieval
- bge-reranker
- ACL 检索权限过滤
- Evidence Gate
- LLM 生成与 Citation
- FastAPI API
- Streamlit UI
- Retrieval / Generation Evaluation

---

## Architecture

### 当前已完成的离线知识库构建流程

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
```

### V1 目标架构

```text
                        Streamlit
                            │
                            ▼
                         FastAPI
                            │
                            ▼
                       QueryService
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             Dense Retrieval     BM25 Retrieval
             BGE-M3 + Qdrant
                   │                 │
                   └────────┬────────┘
                            ▼
                         RRF Fusion
                            │
                            ▼
                        Reranker
                            │
                            ▼
                      Evidence Gate
                      │           │
                      ▼           ▼
                    LLM         Refusal
                      │
                      ▼
                Answer + Citation
```

---

## Data Sources

当前首批知识库文档包括：

- 《生成式人工智能服务管理暂行办法》
- 《互联网信息服务深度合成管理规定》

后续计划扩展：

- OWASP Top 10 for LLM / GenAI
- FastAPI 官方文档选段
- Qdrant 官方文档选段
- 少量企业内部技术规范模拟数据

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

Manifest 负责维护：

- 文档身份
- 文档标题
- 数据来源
- 文档类型
- 版本
- 语言
- 权限级别
- 本地文件路径
- 是否启用

通过 Manifest 将原始文件与代码逻辑解耦，为后续增加更多知识库文档、版本管理以及 ACL 权限过滤提供基础。

---

## Ingestion Pipeline

当前法规类文档的离线处理流程主要包括以下阶段。

### 1. HTML Loading

从本地保存的官方 HTML 页面中读取原始内容。

针对国家网信办页面结构定位正文容器，并移除正文内部可能存在的：

```text
script
style
```

等非正文节点。

Loader 只负责提取正文，不负责 Chunk 切分。

### 2. Text Normalization

对提取出的正文进行统一文本标准化处理，包括空白字符与文本格式清理，为后续结构解析提供稳定输入。

### 3. Regulation Parsing

针对中文法规文本识别：

```text
章
↓
条
↓
条款正文
```

并转换为结构化的法规数据。

Parser 会保留：

- 章节编号
- 章节标题
- 条款编号
- 条款正文

同时通过自动测试覆盖法规解析中的边界情况。

### 4. Structure-aware Chunking

根据法规本身的语义结构进行 Chunk 划分，而不是直接采用固定字符窗口。

### 5. Data Validation

Chunk 构建完成后，在进入 Embedding / Vector Index 之前执行独立的数据质量校验。

---

## Chunking Strategy

法规类文档没有直接采用固定字符窗口切分。

当前采用：

```text
法规全文
   ↓
识别“章”
   ↓
识别“条”
   ↓
一条法规 ≈ 一个 Chunk
```

这样做的主要原因是：

> 法规中的“条”本身就是天然的语义边界。

相比单纯按照固定字符数进行切分，可以减少一个完整法律条款被从中间截断的问题。

### 首批法规 Chunk 统计

| 文档 | Chunk 数 | 最短字符数 | 最长字符数 | 平均字符数 |
|---|---:|---:|---:|---:|
| 《生成式人工智能服务管理暂行办法》 | 24 | 18 | 401 | 127.79 |
| 《互联网信息服务深度合成管理规定》 | 25 | 18 | 491 | 130.04 |
| **合计** | **49** | - | - | - |

根据当前首批法规语料的实际统计：

> 以“条”为 Chunk 边界可以较好保持完整法律语义，当前暂时没有必要进行条内二次切分。

对于较短条款，也不会仅因为字符数量较少而强制与相邻条款合并。

例如：

```text
本办法自某年某月某日起施行。
```

虽然字符数量较少，但它本身就是一个完整且具有独立法律含义的检索证据。

---

## Chunk Data Model

当前标准 `KnowledgeChunk` 包含：

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

主要字段说明：

| 字段 | 作用 |
|---|---|
| `chunk_id` | Chunk 全局唯一标识 |
| `document_id` | 所属文档 ID |
| `title` | 文档标题 |
| `document_type` | 文档类型 |
| `language` | 文档语言 |
| `version` | 文档版本 |
| `chapter_number` | 所属章节编号 |
| `chapter_title` | 所属章节标题 |
| `article_number` | 法规条款编号 |
| `content` | 原始条款正文 |
| `retrieval_text` | 用于后续 Embedding / Retrieval 的增强文本 |
| `source_url` | 原始数据来源 |
| `access_level` | 后续 ACL 检索过滤依据 |
| `chunk_index` | Chunk 在文档中的顺序 |
| `content_hash` | 内容一致性与后续增量更新判断 |

### content 与 retrieval_text

项目区分：

```text
content
```

和：

```text
retrieval_text
```

其中：

- `content` 保存原始条款正文；
- `retrieval_text` 在正文基础上加入文档标题、章节和条款等上下文信息。

这样既可以保留干净的原始证据，又可以让后续 Embedding 获得更完整的语义上下文。

---

## Chunk ID

法规类 Chunk 当前采用可读、稳定的 ID，例如：

```text
cn_genai_interim_2023__第一条
cn_genai_interim_2023__第二条
```

Chunk ID 由：

```text
document_id
+
article_number
```

共同构成。

这样可以直接从 Chunk ID 判断其所属文档和法规条款，同时方便后续进行数据更新、去重和问题排查。

---

## Intermediate Data

经过 Ingestion Pipeline 处理后的标准 Chunk 会保存为：

```text
data/processed/chunks.jsonl
```

采用 JSONL（JSON Lines）格式：

```text
一行 = 一个 KnowledgeChunk
```

离线数据处理与后续 Embedding / Indexing 阶段通过该中间文件解耦。

整体流程：

```text
Raw Documents
      ↓
Ingestion Pipeline
      ↓
chunks.jsonl
      ↓
Embedding
      ↓
Vector Database
```

这样后续调整 Embedding Model 或 Vector Database 时，不需要重复执行原始 HTML 解析与法规结构解析。

---

## Data Quality Validation

在 Chunk 进入 Embedding 与 Vector Index 之前，`chunks.jsonl` 会经过独立的数据质量校验。

当前检查包括：

- `chunk_id` 唯一性
- 核心字段不能为空
- `access_level` 值是否合法
- `retrieval_text` 是否包含必要上下文
- `content_hash` 是否与正文一致
- 同一文档内 `chunk_index` 是否连续

运行：

```bash
python scripts/validate_chunks.py
```

当前首批数据验证结果：

```text
Chunk 总数：49

数据质量校验通过
```

通过在 Embedding 之前增加 Validation 层，尽可能避免错误数据进入 Vector Database。

---

## Tests

项目使用 `pytest` 对核心 Ingestion Pipeline 进行自动化测试。

运行全部测试：

```bash
pytest -v
```

当前测试覆盖：

- 法规章节结构解析
- 法规条款结构解析
- 文件结束时最后一条不会丢失
- 正文中类似“第X条...”的普通文本不会被误识别为新条款
- Structure-aware Chunk 构建
- `content_hash` 稳定性
- 正文变化后 `content_hash` 应发生变化
- 重复 `chunk_id` 检测
- 非法 `access_level` 检测

当前：

```text
9 passed
```

### Regression Test

在开发过程中，自动测试曾发现法规 Parser 的一个边界问题。

例如：

```text
第一条
第一条内容。
第二条
这是第二条内容。
```

原始 Parser 正则可能将：

```text
第一条内容。
```

错误识别为新的：

```text
第一条
```

导致产生重复条款。

问题修复后增加了对应的 Regression Test，确保未来修改 Parser 时不会重新引入该问题。

这也是当前项目引入自动测试的主要目的之一：

```text
真实数据验证
        +
自动化边界测试
        ↓
提高 Ingestion Pipeline 稳定性
```

---

## Project Structure

当前项目主要目录结构：

```text
enterprise-rag-compliance/
│
├── data/
│   ├── manifest/
│   │   └── documents.yaml
│   ├── raw/
│   └── processed/
│
├── scripts/
│
├── src/
│   └── enterprise_rag/
│       └── ingestion/
│           ├── loaders/
│           ├── manifest.py
│           ├── normalizer.py
│           ├── models.py
│           ├── regulation_parser.py
│           ├── document_builder.py
│           ├── chunker.py
│           └── validator.py
│
├── tests/
│
├── .env.example.py
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

> 项目目录仍会随着 Retrieval、Generation、API 与 Evaluation 模块的开发继续扩展。

---

## Current Tech Stack

当前已经实际用于项目中的主要技术：

- Python 3.11
- BeautifulSoup
- PyYAML
- Docker
- Docker Compose
- Qdrant
- pytest

---

## Planned Tech Stack

后续 V1 计划接入：

- FastAPI
- Streamlit
- LangChain
- BGE-M3
- Qdrant Vector Search
- BM25
- Reciprocal Rank Fusion（RRF）
- bge-reranker
- LLM
- Retrieval Evaluation

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

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 4. Start Qdrant

```bash
docker compose up -d
```

检查容器状态：

```bash
docker compose ps
```

### 5. Run Tests

```bash
pytest -v
```

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
- [x] Text Normalizer
- [x] Normalized Document
- [x] Regulation Parser
- [x] Structure-aware Chunking
- [x] KnowledgeChunk
- [x] JSONL Output
- [x] Data Validation
- [x] Unit / Regression Tests

### Phase 3 — Embedding & Indexing

- [ ] BGE-M3 Embedding
- [ ] Embedding Batch Pipeline
- [ ] Qdrant Collection
- [ ] Vector Upsert
- [ ] Payload Metadata
- [ ] Index Validation

### Phase 4 — Retrieval

- [ ] Dense Retrieval
- [ ] BM25 Retrieval
- [ ] RRF Fusion
- [ ] Reranker
- [ ] ACL Filtering

### Phase 5 — Generation

- [ ] Query Service
- [ ] Evidence Gate
- [ ] Prompt Construction
- [ ] LLM Generation
- [ ] Citation
- [ ] Refusal

### Phase 6 — Application

- [ ] FastAPI
- [ ] Streamlit
- [ ] Query API
- [ ] Health Check
- [ ] End-to-End Demo

### Phase 7 — Evaluation

- [ ] Retrieval Dataset
- [ ] Recall@K
- [ ] MRR
- [ ] Rerank Comparison
- [ ] Citation Evaluation
- [ ] Refusal Evaluation

---

## Current Milestone

当前已完成：

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
pytest
```

当前知识库数据：

```text
2 篇公开法规
49 个法规级 Chunks
9 个自动化测试
```

下一阶段：

```text
chunks.jsonl
      ↓
BGE-M3
      ↓
Embedding Vectors
      ↓
Qdrant
      ↓
Dense Retrieval
```

---

## License / Notice

本项目主要使用公开法规与官方技术文档作为知识库数据来源。

项目用于学习、技术研究与工程实践，不构成法律意见。