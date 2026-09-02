# RAG Evaluation Report

本文档记录「企业 AI 合规与应用规范助手」当前阶段的离线评测设计、实验结果与 Failure Analysis。

> 当前评测基于小规模 Seed Dataset，主要用于验证系统设计、评测链路与迭代方法，不代表生产环境准确率，也不构成大规模统计结论。

---

## 1. Evaluation Goals

当前 RAG 系统不只评估“回答看起来是否正确”，而是分别评估：

1. Retrieval 是否找回正确证据；
2. Hybrid Retrieval / Rerank 是否真正改善排序；
3. ACL 是否会产生越权检索；
4. Evidence Gate 是否能够拒绝明显无依据问题；
5. Full-RAG 是否能够正确回答或拒答；
6. Citation 是否引用正确且必要的证据；
7. Prompt 优化是否产生新的 Regression；
8. 不同 Retrieval Pipeline 的延迟成本。

因此当前 Evaluation 被拆分为：

```text
Retrieval Evaluation
        ↓
Latency Evaluation
        ↓
Answer / Refusal Evaluation
        ↓
Citation Evaluation
        ↓
Failure Analysis
        ↓
Prompt Regression Evaluation
```

---

## 2. Current Evaluation Dataset

当前 Seed Evaluation Dataset：

```text
data/eval/retrieval_eval_v1.jsonl
```

共：

```text
20 cases
```

其中：

```text
Answerable   : 14
Unanswerable : 6
```

Query 类型包括：

```text
direct
paraphrase
short
ambiguous
hard_negative
out_of_domain
```

当前知识库 Corpus 为两篇公开中文法规经过结构化切分后的：

```text
49 regulation chunks
```

因此该 Dataset 的主要目标不是证明系统已经具有大规模泛化能力，而是验证：

- Retrieval Evaluation 是否可执行；
- Answer / Refusal 是否可量化；
- Citation 是否可审计；
- Failure 是否能够被定位；
- Prompt 修改是否能够进行 Regression Test。

后续随着 OWASP LLM、FastAPI、Qdrant 等文档加入，将继续扩展到更大规模、更多领域的 Evaluation Dataset。

---

# 3. Retrieval Evaluation

## 3.1 Retrieval Methods

当前比较四种 Retrieval Pipeline：

### Dense

```text
Query
↓
BGE-M3
↓
Qdrant Dense Search
```

### BM25

```text
Query
↓
Jieba Tokenization
↓
BM25
```

### Hybrid RRF

```text
Dense Top-K
+
BM25 Top-K
↓
Reciprocal Rank Fusion
```

RRF 当前使用：

```text
score(d) = Σ 1 / (60 + rank(d))
```

### Hybrid + Rerank

```text
Dense
+
BM25
↓
RRF
↓
bge-reranker-v2-m3
```

---

# 4. Retrieval Quality Results

当前 Retrieval Quality Evaluation 仅统计：

```text
14 answerable cases
```

结果如下：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.7500 | 0.8929 | 1.0000 | 1.0000 | 0.8750 |
| Hybrid RRF | 0.8929 | 1.0000 | 1.0000 | 1.0000 | 0.9643 |
| Hybrid + Rerank | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 4.1 Current Interpretation

在当前小规模法规 Corpus 上：

```text
Dense
```

已经表现非常强。

BM25 的 Top-1 排序明显弱于 Dense，但：

```text
Recall@5 = 1.0
```

说明它仍然能够提供有价值的词法候选。

当前 Seed Dataset 上：

```text
Hybrid RRF
```

没有超过 Dense，甚至出现了一个排序退化 Case。

而：

```text
Hybrid + Rerank
```

能够修复该退化，但总体质量只恢复到 Dense 水平，没有超过 Dense。

因此目前不能得出：

> Hybrid Retrieval 一定优于 Dense Retrieval。

更准确的结论是：

> 在当前规模较小、语义较明确的法规 Corpus 上，Dense 已经达到较高召回率；BM25 提供低成本词法补充，而 RRF 与 Rerank 的实际收益需要在未来更大、更异构的 Corpus 上继续验证。

---

# 5. Retrieval Failure Analysis

当前一个典型 Case：

```text
R010
违法内容怎么处理？
```

Retrieval Gold：

```text
cn_genai_interim_2023__第十四条
```

Dense Ranking：

```text
Rank 1
cn_genai_interim_2023__第十四条
```

Hybrid RRF Ranking：

```text
Rank 1
cn_deep_synthesis_2022__第十条

Rank 2
cn_genai_interim_2023__第十四条
```

原因：

BM25 对“违法内容”等词法表达具有较强匹配能力，将另一篇法规中同样涉及违法内容处理的条款提升到了更高位置。

经过 Reranker 后：

```text
cn_genai_interim_2023__第十四条
```

重新回到 Rank 1。

这个 Case 表明：

> Hybrid Fusion 并不是天然单调提升。词法信号可能提升相关候选，也可能改变已经正确的 Dense 排序；Reranker 的价值之一就是重新判断融合后的候选相关性。

当前没有基于这个单一 Case 调整 RRF 参数，因为直接在 Evaluation Dataset 上调参会造成 Evaluation Leakage。

---

# 6. Retrieval Latency Benchmark

早期按 Method Block 顺序执行 Benchmark 时观察到明显的 GPU Warm-State / Method Order Bias。

因此最终采用：

```text
Query-level Interleaved Benchmark
```

即在相同 Runtime 中交错执行不同 Retrieval Method。

测试条件：

```text
Queries        : 20
Rounds         : 5
Samples/method : 100
Warmup         : 2
GPU            : RTX 4060 Laptop
```

GPU Method 在计时前后执行 CUDA Synchronization。

Benchmark 不包含：

```text
Model Initialization
FastAPI
HTTP
LLM Generation
```

最终结果：

| Method | Mean (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.39 | 0.32 | 0.52 | 0.09 | 6.09 |
| Dense | 53.57 | 52.20 | 75.44 | 30.35 | 98.18 |
| Hybrid RRF | 56.54 | 54.71 | 77.51 | 33.48 | 106.12 |
| Hybrid + Rerank | 452.33 | 455.68 | 503.99 | 348.80 | 538.37 |

当前延迟关系符合预期：

```text
BM25
<
Dense
≈
Hybrid RRF
<<<<
Hybrid + Rerank
```

其中 Reranker 是当前 Retrieval Pipeline 中最明显的额外延迟来源。

因此未来是否始终开启 Rerank，需要结合更大 Evaluation Dataset 上的实际质量收益，而不能只依据“理论上更高级”。

---

# 7. ACL Evaluation

当前系统支持：

```text
guest
developer
admin
```

对应不同 Access Level。

ACL 约束发生在 Retrieval Candidate Generation 之前：

```text
User Role
↓
AccessContext
↓
Allowed Access Levels
↓
Qdrant Filter / Authorized BM25 Corpus
↓
Retrieval
```

而不是：

```text
Global Retrieval
↓
Post Filter
```

这样能够避免 Unauthorized Chunk 先参与：

```text
Top-K Competition
Fusion
Rerank
```

当前 Synthetic Mixed-Access Evaluation 中：

```text
Unauthorized Final Results = 0
ACL Leakage Rate = 0%
```

该结果用于验证当前 ACL Retrieval Architecture。

当前 API 中的 `role` 仍然属于 Demo Input，并不等价于生产级身份认证。

生产环境应由可信：

```text
SSO
JWT
IAM
```

等身份系统生成 AccessContext。

---

# 8. Answer / Refusal Evaluation

Full-RAG Evaluation Pipeline：

```text
Query
↓
ACL-aware Retrieval
↓
Hybrid Fusion
↓
Rerank
↓
Coarse Relevance Gate
↓
Evidence-Constrained Generation
↓
Answer / Structured Refusal
↓
Deterministic Citation Validation
```

当前 Prompt v2 Full-RAG Evaluation：

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

当前没有观察到 Answer / Refusal Decision Failure。

需要强调：

> 该结果来自 20 条 Seed Dataset 的单次 Full-RAG Evaluation，仅用于当前开发阶段的系统验证，不能解释为生产环境准确率达到 100%。

---

# 9. Retrieval Relevance != Answerability

开发过程中曾尝试分析：

```text
Top-1 Rerank Score
```

能否直接作为 Answerability Threshold。

实验结果发现：

```text
Answerable Query
和
Hard Negative Query
```

的 Rerank Score 存在明显重叠。

因此最终不采用：

```text
rerank_score > threshold
=> 可以回答
```

这种单层决策。

当前设计为：

```text
Coarse Relevance Gate
↓
只过滤明显 Out-of-Domain Query
↓
Evidence-Constrained Generation
↓
LLM 判断当前 Evidence 是否真正足以回答
```

这一区分非常重要：

> Reranker 判断的是 Query 与 Evidence 的相关程度，而 Answerability 判断的是 Evidence 是否包含回答该问题所需的充分事实。

相关并不等于足以回答。

---

# 10. Citation Evaluation Design

初版 Citation Evaluation 曾直接复用：

```text
Retrieval Gold
```

作为 Citation Gold。

Failure Analysis 后发现这个设计并不准确。

例如一个 Retrieval Case 的目标可能只是：

> Retriever 至少应找回核心条款。

而 Citation Evaluation 的问题是：

> 最终答案实际引用的所有证据是否直接支持回答？

因此当前正式拆分为：

```text
gold_chunk_ids
→ Retrieval Gold

citation_gold_chunk_ids
→ Citation Gold
```

并增加：

```text
strict_citation_eval
```

对于范围过宽或存在多种合理法规来源的 Query：

```text
strict_citation_eval = false
```

这些 Query 仍然参与：

```text
Retrieval Evaluation
Answerability Evaluation
All-case Citation Analysis
```

但不进入严格 Citation Precision / Recall。

---

# 11. Citation Gold Audit

人工 Audit 曾重点检查：

```text
R001
R006
R008
R010
R014
```

发现原始 Citation mismatch 混合了三类不同原因：

```text
True Over-Citation
Gold Annotation Incomplete
Ambiguous Query Scope
```

因此不能简单把所有：

```text
Actual Citation - Retrieval Gold
```

都判定为模型错误。

这一步最终形成结论：

> Citation Evaluation 必须独立标注 Citation Gold，不能机械复用 Retrieval Gold。

---

# 12. Prompt v1 Citation Results

在修正 Citation Annotation 后，Prompt v1 的主要指标为：

| Metric | Prompt v1 |
|---|---:|
| All-case Citation Precision | 0.9524 |
| All-case Citation Recall | 0.9821 |
| Citation Hit Rate | 1.0000 |
| Strict Citation Precision | 0.9444 |
| Strict Citation Recall | 0.9792 |
| Strict Citation Hit Rate | 1.0000 |

主要 Strict Failure 为：

```text
R001
生成式人工智能服务处理训练数据需要遵守什么规定？
```

核心 Citation Gold：

```text
生成式人工智能服务管理暂行办法 第七条
```

但 Prompt v1 同时引用：

```text
生成式 AI 第四条
深度合成 第十四条
```

形成明显 Over-Citation。

---

# 13. Prompt v2 Optimization

针对上述 Failure，没有调整：

```text
Dense Retrieval
BM25
RRF
Reranker
Evidence Gate
```

因为核心证据已经正确召回。

问题定位在：

```text
Generation / Citation Selection
```

因此 Prompt v2 增加：

```text
Minimal Sufficient Evidence Set
```

原则：

1. 只引用直接支持最终答案中实际陈述的 Evidence；
2. 主题相关不等于需要引用；
3. 已有 Evidence 足以支持同一事实时，不重复增加类似 Evidence；
4. 多个不同 Evidence 分别支持多个独立事实时，应保留多个必要 Citation；
5. 用户明确限定法规、制度或场景时，优先引用范围一致的 Evidence；
6. Evidence ID 仅用于结构化 citations，不进入最终 answer 正文。

目标不是：

```text
Citation 越少越好
```

而是：

```text
最小充分证据集
```

---

# 14. Targeted Prompt Regression

Prompt v2 首先针对：

```text
R001
R006
R008
```

执行 Targeted Regression。

设计：

```text
R001
→ Over-Citation Failure Case

R006
→ 多独立义务 Control Case

R008
→ 多证据 Control Case
```

这样可以防止 Prompt 为修复 R001 而退化成：

```text
“无论什么问题都只引用一条证据”
```

Targeted Evaluation 表明：

- R001 可以收敛到唯一核心 Citation；
- R006 仍能够保留多个独立义务的 Citation；
- R008 仍能够保留两个必要条款。

---

# 15. Prompt v2 Full-RAG Re-evaluation

Prompt v2 在完整 20 条 Seed Dataset 上重新运行，并保存独立 Snapshot。

结果：

| Metric | Prompt v1 | Prompt v2 | Delta |
|---|---:|---:|---:|
| Overall Decision Accuracy | 1.0000 | 1.0000 | +0.0000 |
| Answerable Accuracy | 1.0000 | 1.0000 | +0.0000 |
| Refusal Accuracy | 1.0000 | 1.0000 | +0.0000 |
| Hard Negative Refusal Accuracy | 1.0000 | 1.0000 | +0.0000 |
| OOD Refusal Accuracy | 1.0000 | 1.0000 | +0.0000 |
| Citation Precision | 0.9524 | 1.0000 | +0.0476 |
| Citation Recall | 0.9821 | 0.9643 | -0.0179 |
| Citation Hit Rate | 1.0000 | 1.0000 | +0.0000 |
| Strict Citation Precision | 0.9444 | 1.0000 | +0.0556 |
| Strict Citation Recall | 0.9792 | 1.0000 | +0.0208 |
| Strict Citation Hit Rate | 1.0000 | 1.0000 | +0.0000 |

Prompt v2：

```text
Strict Citation Mismatch = 0
Evidence-ID Leak = 0
Decision Regression = 0
```

因此当前接受 Prompt v2。

---

# 16. Why All-case Citation Recall Decreased

Prompt v2 的：

```text
All-case Citation Recall
```

从：

```text
0.9821
```

下降到：

```text
0.9643
```

主要来自非 Strict Case：

```text
R010
违法内容怎么处理？
```

Citation Gold 中同时包含：

```text
生成式 AI 第十四条
深度合成 第十条
```

Prompt v2 只使用：

```text
生成式 AI 第十四条
```

即可完整回答问题。

因此：

```text
Precision = 1.0
Recall = 0.5
```

但 R010 本身没有限定具体法规，两条 Gold 更接近：

```text
multiple acceptable evidence sources
```

而不是：

```text
both citations are mandatory
```

因此该 Case 已被：

```text
strict_citation_eval = false
```

排除出严格 Citation 指标。

这也是当前优先报告 Strict Citation Metrics 的原因。

---

# 17. Evaluation Snapshot Persistence

真实 LLM Evaluation 存在：

```text
API Cost
Network Latency
Generation Randomness
```

因此不会在每次 Failure Analysis 时重新调用 LLM。

当前流程：

```text
Expensive Full-RAG Run
↓
JSONL Raw Snapshot
↓
Offline Metrics
↓
Failure Analysis
↓
Gold Audit
↓
Prompt Comparison
```

代表性结果：

```text
data/eval/results/
├── answer_eval_v1_run_001.jsonl
├── answer_eval_v1_run_001_audited.jsonl
└── answer_eval_v1_run_002.jsonl
```

其中：

```text
run_001
→ Prompt v1 原始 Full-RAG Result

run_001_audited
→ Prompt v1 Result + 修正后的 Citation Annotation

run_002
→ Prompt v2 Full-RAG Result
```

这样能够避免：

```text
修改分析代码
↓
重新跑 LLM
↓
模型输出又变化
↓
无法复现实验
```

---

## 18. Reproduction Commands

### 18.1 Retrieval Quality Evaluation

```powershell
python scripts/run_retrieval_eval.py
```

用于比较：

```text
Dense
BM25
Hybrid RRF
Hybrid + Rerank
```

主要输出：

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
```

---

### 18.2 Retrieval Latency Benchmark

项目中保留了两版 Latency Benchmark：

```text
run_retrieval_latency_benchmark.py
```

为早期 Blocked Benchmark。

最终更推荐使用：

```powershell
python scripts/run_interleaved_retrieval_latency_benchmark.py
```

该版本采用：

```text
Query-level Interleaving
+
Deterministic Shuffle
+
CUDA Synchronization
```

主要用于减少 Method Order / GPU Warm-State Bias。

README 和最终 Evaluation Report 中的正式延迟结果均以 Interleaved Benchmark 为准。

---

### 18.3 Inspect Evaluation Dataset

```powershell
python scripts/inspect_retrieval_eval_dataset.py
```

用于检查：

```text
Case 数量
Answerable / Unanswerable 分布
Category 分布
Gold Chunk 标注
```

---

### 18.4 Full-RAG Answer / Refusal Evaluation

运行前需要：

```text
Qdrant 正常运行
BGE-M3 可加载
bge-reranker-v2-m3 可加载
.env 中配置有效 SILICONFLOW_API_KEY
```

执行：

```powershell
python scripts/run_answer_eval.py
```

当前脚本会执行完整：

```text
Retrieval
↓
Rerank
↓
Evidence Gate
↓
LLM Generation / Refusal
↓
Citation Validation
↓
Evaluation Snapshot Persistence
```

并生成：

```text
data/eval/results/answer_eval_v1_run_002.jsonl
```

---

### 18.5 Targeted Citation Regression

```powershell
python scripts/run_targeted_citation_regression.py
```

当前主要用于：

```text
R001
R006
R008
```

验证 Prompt Citation Optimization 是否：

```text
修复 Over-Citation
同时不破坏合理多证据引用
```

---

### 18.6 Offline Snapshot Analysis

```powershell
python scripts/analyze_answer_eval_snapshot.py
```

该脚本只读取已经持久化的 Evaluation Snapshot。

不会：

```text
加载 BGE-M3
加载 Reranker
查询 Qdrant
调用 SiliconFlow
```

因此可用于低成本重复进行：

```text
Metrics
Failure Analysis
Citation Audit
```

---

### 18.7 Prompt v1 / Prompt v2 Comparison

```powershell
python scripts/compare_answer_eval_snapshots.py
```

用于比较：

```text
answer_eval_v1_run_001_audited.jsonl
vs
answer_eval_v1_run_002.jsonl
```

主要输出：

```text
Aggregate Metric Delta
Case-level Changes
Prompt Regression Check
```

整个 Comparison 为纯离线过程，不调用 LLM。

---

### 18.8 Citation Gold Migration

```powershell
python scripts/migrate_citation_gold.py
```

用于将旧 Evaluation Dataset 升级为：

```text
Retrieval Gold
+
Citation Gold
+
strict_citation_eval
```

该脚本属于 Dataset Annotation Migration 工具，不需要在日常 Evaluation 中重复执行。

---

### 18.9 Snapshot Annotation Enrichment

```powershell
python scripts/enrich_answer_eval_snapshot.py
```

用于将新的 Citation Gold Annotation 合并到旧 Full-RAG Snapshot。

只修改：

```text
citation_gold_chunk_ids
strict_citation_eval
```

不会修改历史：

```text
answer
cited_chunk_ids
reason
rerank score
latency
```

因此不需要重新调用 LLM。

---

# 19. Current Conclusions

当前阶段可以得到以下结论。

### Retrieval

- Dense 是当前 Corpus 上质量 / 延迟最均衡的 Baseline；
- BM25 延迟极低，能够提供词法补充；
- RRF 在当前小规模 Seed Dataset 上没有稳定超过 Dense；
- Hybrid Fusion 可能出现排序退化；
- Reranker 能够修复部分 Fusion Failure，但具有明显额外延迟成本；
- 当前不能得出“Hybrid 一定优于 Dense”的结论。

### ACL

- ACL 必须在 Candidate Generation 前执行；
- 当前 Mixed-Access Test 中未观察到 Unauthorized Result；
- API Role 目前仅用于 Demo，并不替代生产身份认证。

### Evidence / Refusal

- Retrieval Relevance 不等价于 Answerability；
- 单一 Rerank Threshold 无法可靠判断证据是否足够；
- 当前采用 Coarse Relevance Gate + Evidence-Constrained Generation；
- 无依据问题必须结构化拒答。

### Citation

- Retrieval Gold 和 Citation Gold 必须解耦；
- Ambiguous Query 不适合直接进入严格 Citation Precision；
- Citation 应选择最小充分证据集，而不是简单追求数量最少；
- Prompt 优化必须配合 Targeted Regression，防止修复一个 Case 的同时破坏合理多证据 Case。

### Evaluation

- 当前结果来自小规模 Seed Dataset；
- 主要目标是建立 Evaluation-driven Development 闭环；
- 后续需要扩大 Corpus 和 held-out Evaluation Set 后重新验证所有结论。

---

# 20. Limitations

当前 Evaluation 仍存在明显限制：

1. Corpus 当前仍以两篇中文法规为主；
2. Evaluation Dataset 仅 20 条；
3. Prompt v1 / v2 目前均为单次 Full-RAG Run；
4. 尚未进行高并发 API Load Test；
5. 尚未系统评估自然语言 Answer Correctness；
6. 尚未独立评估完整 Faithfulness；
7. Citation Gold 仍由人工小规模维护；
8. 当前 ACL Evaluation 使用 Synthetic Access Levels；
9. 当前 Full-RAG Latency 包含远程 LLM 网络波动，不属于正式 Retrieval Benchmark。

因此当前所有百分比结果均应理解为：

> 当前 Seed Dataset 上的开发阶段实验结果。

而不是生产系统 SLA 或准确率承诺。

---

# 21. Next Evaluation Work

随着知识库继续扩展，后续计划：

```text
30–80 public documents
↓
50–100 evaluation queries
↓
More hard negatives
↓
Cross-document questions
↓
Technical + Regulation mixed corpus
↓
Held-out evaluation set
```

重点继续验证：

```text
Dense vs Hybrid
Reranker real gain
ACL leakage
Refusal robustness
Citation quality
Answer correctness
Faithfulness
Latency / Quality trade-off
```

最终目标不是追求某个单一 100% 指标，而是建立：

```text
可复现
可解释
可定位 Failure
可验证改动
```

的 RAG Evaluation Pipeline。