# RAG Evaluation Report

本文档记录「企业 AI 合规与应用规范助手」当前阶段的离线评测设计、实验结果、Failure Analysis 与 Regression Evaluation。

当前 Evaluation 已覆盖：

```text
Retrieval Quality
Retrieval Regression
Role-aware Retrieval
ACL
Latency
Answer / Refusal
Citation
Prompt Regression
Failure Analysis
Snapshot Persistence
```

> 当前 Corpus 与 Evaluation Dataset 规模仍然较小。本文中的指标主要用于验证系统设计、评测方法和迭代闭环，不代表生产环境准确率，也不构成大规模统计结论或 SLA。

---

# 1. Evaluation Goals

当前 RAG 系统不只评估：

```text
“回答看起来是否正确”
```

而是将问题拆分为多个独立层次：

1. Retrieval 是否找回正确证据；
2. Corpus 扩展后旧 Retrieval 能力是否发生 Regression；
3. Dense / BM25 / Hybrid / Rerank 的真实收益分别是什么；
4. Hybrid Fusion 是否可能破坏已经正确的排序；
5. Reranker 是否能够修复 Fusion Failure；
6. ACL 是否在 Candidate Generation 前正确限制搜索空间；
7. Retrieval Relevance 是否真的意味着 Evidence 足够回答；
8. Full-RAG 是否能够正确回答或拒答；
9. Citation 是否引用正确且必要的证据；
10. Prompt 修改是否产生新的 Regression；
11. 不同 Retrieval Pipeline 的延迟成本；
12. Failure 是否能够定位到具体 Query 和具体 Ranking。

因此当前 Evaluation Architecture 为：

```text
Evaluation Dataset
        ↓
Gold Integrity Check
        ↓
Role-aware Retrieval Evaluation
        ↓
Retrieval Quality
        ↓
Regression Evaluation
        ↓
Pairwise Ablation Analysis
        ↓
Per-Method Failure Inspector
        ↓
Latency Benchmark
        ↓
Answer / Refusal Evaluation
        ↓
Citation Evaluation
        ↓
Snapshot Persistence
        ↓
Offline Failure Analysis
        ↓
Prompt Regression Evaluation
```

---

# 2. Current Corpus

当前 Corpus V2 已从最初的：

```text
2 篇中文法规
49 regulation chunks
```

扩展为：

```text
5 篇公开文档
104 chunks
```

当前文档包括：

```text
1. 《生成式人工智能服务管理暂行办法》

2. 《互联网信息服务深度合成管理规定》

3. OWASP LLM01:2025 Prompt Injection

4. FastAPI Dependencies

5. FastAPI Lifespan Events
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

其中：

```text
FastAPI Dependencies
FastAPI Lifespan
```

本身属于公开官方文档。

项目将其标记为：

```text
access_level = developer
```

仅用于模拟企业内部技术规范的 ACL 场景，并不表示 FastAPI 官方文档本身具有真实访问限制。

---

# 3. Evaluation Dataset Versioning

当前 Retrieval Evaluation 不再只有一个 Dataset。

而是明确区分：

```text
Retrieval Eval V1
+
Retrieval Eval V2
```

两者承担不同职责。

---

## 3.1 Retrieval Eval V1

路径：

```text
data/eval/retrieval_eval_v1.jsonl
```

规模：

```text
20 cases
```

其中：

```text
Answerable   : 14
Unanswerable : 6
```

Query 类型：

```text
direct
paraphrase
short
ambiguous
hard_negative
out_of_domain
```

V1 最初建立时，Corpus 主要由：

```text
2 篇中文法规
49 chunks
```

组成。

现在 V1 不再作为“当前完整能力测试集”，而是作为：

```text
Frozen Regression Benchmark
```

其作用是：

> 当 Corpus、Schema、ACL 或 Retrieval Pipeline 发生升级之后，验证旧能力是否发生 Regression。

V1 Dataset 已冻结。

不会因为：

```text
Corpus 扩展
Retriever 新输出
Reranker 新排序
```

而修改 Gold。

---

## 3.2 Retrieval Eval V2

路径：

```text
data/eval/retrieval_eval_v2.jsonl
```

规模：

```text
34 cases
```

其中：

```text
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

V2 在 V1 基础上增加：

```text
OWASP Prompt Injection
FastAPI Dependencies
FastAPI Lifespan
developer-role Retrieval
new Hard Negatives
```

因此：

```text
V1
→ Regression Benchmark

V2
→ Current Capability Benchmark
```

两者不互相替代。

---

# 4. Gold Annotation Design

当前 Evaluation Dataset 同时包含：

```text
gold_chunk_ids
citation_gold_chunk_ids
strict_citation_eval
role
```

它们承担不同职责。

---

## 4.1 Retrieval Gold

```text
gold_chunk_ids
```

用于：

```text
Recall@K
MRR
```

它表示：

> Retriever 应优先找回的核心 Evidence。

---

## 4.2 Citation Gold

```text
citation_gold_chunk_ids
```

用于：

```text
Citation Precision
Citation Recall
Citation Hit Rate
```

它表示：

> 最终答案允许直接引用、且能够支持答案陈述的 Evidence。

一个重要结论是：

```text
Retrieval Gold
≠
Citation Gold
```

两者不能机械复用。

---

## 4.3 Strict Citation Evaluation

```text
strict_citation_eval
```

用于区分：

```text
严格可判定的 Citation Case
```

与：

```text
存在多个合理来源
范围不够明确
ambiguous scope
```

的 Case。

例如：

```text
训练语料有哪些合规要求？
```

同时可能涉及多篇法规。

因此这类 Query 可以：

```text
继续参与 Retrieval
继续参与 Answerability
继续参与 All-case Citation
```

但不进入严格 Citation Precision / Recall。

---

# 5. Gold Integrity Check

Evaluation Dataset 中的 Gold 不能只“看起来像 Chunk ID”。

在运行正式 Retrieval Evaluation 前，需要检查：

```text
gold_chunk_ids
citation_gold_chunk_ids
```

是否真实存在于：

```text
data/processed/chunks.jsonl
```

Corpus V2 当前检查结果：

```text
Missing Retrieval Gold = []
Missing Citation Gold  = []
```

即：

```text
Missing Retrieval Gold = 0
Missing Citation Gold  = 0
```

该检查可以防止：

```text
Chunk ID Schema Evolution
手工 Gold 拼写错误
旧 Chunk ID 漂移
```

造成假的 Retrieval Failure。

---

# 6. Role-aware Retrieval Evaluation

最初 Retrieval Evaluation 中所有 Query 默认使用：

```text
guest
```

随着 Corpus V2 引入：

```text
developer
```

级技术文档，Evaluation Dataset 增加：

```text
role
```

字段。

因此现在一条 Retrieval Case 的语义从：

```text
Query
↓
Gold
```

升级为：

```text
Query
+
Role
↓
Authorized Candidate Space
↓
Gold
```

例如：

```text
FastAPI 中 Dependency Injection 是什么意思？
```

可以定义：

```text
role = developer
```

从而使：

```text
FastAPI developer chunks
```

进入合法 Candidate Space。

---

## 6.1 Backward Compatibility

Retrieval Eval V1 历史上没有显式：

```text
role
```

字段。

而当时 Retrieval Runner 的默认值本来就是：

```text
guest
```

因此新 Dataset Loader 对缺失 role 的 V1 Case 保持：

```text
role = guest
```

这样能够保证：

```text
Schema Evolution
```

不会偷偷改变：

```text
Frozen V1 Benchmark
```

的实验语义。

---

# 7. Retrieval Methods

当前比较四种 Retrieval Pipeline。

---

## 7.1 Dense

```text
Query
↓
BGE-M3
↓
1024-d Embedding
↓
ACL Qdrant Filter
↓
Qdrant Dense Search
```

Embedding Model：

```text
BAAI/bge-m3
```

Distance：

```text
Cosine
```

---

## 7.2 BM25

```text
Query
↓
Jieba Tokenization
↓
Authorized BM25 Corpus
↓
BM25
```

BM25 使用与 Dense 相同的：

```text
retrieval_text
```

但工作机制主要依赖：

```text
Lexical Overlap
```

---

## 7.3 Hybrid RRF

```text
Dense Top20
+
BM25 Top20
↓
Reciprocal Rank Fusion
↓
Hybrid Top20
```

RRF 当前使用：

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
人工 Weight
```

---

## 7.4 Hybrid + Rerank

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
Final Ranking
```

Reranker：

```text
BAAI/bge-reranker-v2-m3
```

Reranker Score 当前只用于：

```text
Ranking
Coarse Relevance Analysis
```

不能直接解释为概率。

---

# 8. Frozen V1 Retrieval Regression

Corpus 从：

```text
49 regulation chunks
```

扩展到：

```text
104 heterogeneous chunks
```

后，没有修改 Retrieval Eval V1 Gold，而是直接重新运行冻结 Benchmark。

Corpus V2 上的 V1 Regression 结果：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.8214 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| Hybrid RRF | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid + Rerank | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

与 Corpus V1 初始结果相比：

```text
Dense
→ 保持不变

BM25
→ Recall@1 提升
→ MRR 提升

Hybrid RRF
→ Recall@1 提升
→ MRR 提升

Hybrid + Rerank
→ 保持不变
```

Corpus V1 中曾出现的：

```text
R010
Dense → Hybrid RRF Rank Degradation
```

在 Corpus V2 上的冻结 V1 Regression 中没有再次出现。

因此当前可以得到：

> Corpus 扩展到 104 Chunk 后，在冻结的旧任务上没有观察到 Retrieval Quality Regression。

同时需要强调：

> V1 只有 14 条 Answerable Query，因此该结果属于 preliminary regression evidence，而不是最终算法结论。

---

# 9. Corpus V2 Retrieval Quality

当前 Retrieval Eval V2：

```text
34 total cases
26 answerable cases
```

Retrieval Quality 只统计：

```text
answerable = true
```

的 Case。

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

---

# 10. Understanding Dense Recall@1

Dense：

```text
Recall@1 = 0.9808
MRR@10   = 1.0000
```

乍看可能会认为两者矛盾。

实际上这是因为 Evaluation Dataset 存在：

```text
Multi-Gold Query
```

例如：

```text
Gold = A, B
Top1 = A
```

则：

```text
Recall@1 = 0.5
```

因为只覆盖了一半 Gold。

但：

```text
RR = 1.0
```

因为第一个 Gold 已经位于 Rank1。

General Failure Inspector 结果：

```text
Dense Non-Top1 Gold Cases
=========================

No cases.
```

因此：

> 26 条 Answerable Query 中，每条 Query 至少有一个核心 Retrieval Gold 位于 Dense Rank1。

---

# 11. Cross-Lingual Dense Retrieval

Corpus V2 新增：

```text
英文 OWASP
英文 FastAPI
```

但 Evaluation Query 主要仍为中文。

例如：

```text
提示词注入 最小权限
```

对应 Gold：

```text
Enforce privilege control
and least privilege access
```

Dense 能够完成：

```text
中文 Query
↓
BGE-M3
↓
英文 Evidence
```

的语义匹配。

多个 OWASP / FastAPI Case 中：

```text
Dense Gold Rank = 1
```

因此当前实验支持：

> BGE-M3 对当前中英混合 Corpus 具有较稳定的跨语言 Dense Retrieval 能力。

这也是当前选择 BGE-M3 作为 Dense Embedding Model 的一个实验依据。

---

# 12. BM25 Cross-Lingual Failure

BM25 在 Corpus V2：

```text
Recall@10 = 0.7692
MRR@10    = 0.6551
```

General Failure Inspector 找到 6 条：

```text
Top10 中完全没有任何 Retrieval Gold
```

的 Case。

---

## 12.1 R021

```text
OWASP 所说的直接提示词注入是什么？
```

Gold：

```text
Direct Prompt Injections
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.2 R022

```text
LLM 从网页或文件读取到隐藏恶意指令并因此改变行为，
属于什么风险？
```

Gold：

```text
Indirect Prompt Injections
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.3 R023

```text
提示词注入 最小权限
```

Gold：

```text
Enforce privilege control
and least privilege access
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.4 R024

```text
OWASP 对模型执行高风险或高权限操作
有什么人工控制建议？
```

Gold：

```text
Require human approval
for high-risk actions
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.5 R025

```text
如何通过渗透测试和攻击模拟
来降低提示词注入风险？
```

Gold：

```text
Conduct adversarial testing
and attack simulations
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.6 R030

```text
FastAPI 应用启动时加载共享机器学习模型，
并在关闭时释放资源，推荐怎么做？
```

Gold：

```text
FastAPI Lifespan
```

BM25：

```text
RR@10 = 0
Recall@10 = 0
```

---

## 12.7 Root Cause

这些 Failure 的共同特征：

```text
中文 Query
→ 英文 Corpus
```

BM25 主要依赖：

```text
Token Overlap
Term Frequency
Inverse Document Frequency
```

例如：

```text
最小权限
```

与：

```text
least privilege
```

在普通 Lexical Retrieval 中不存在天然匹配关系。

因此形成：

```text
Cross-lingual Lexical Mismatch
```

当前结论：

> BM25 在 Corpus V2 中仍具有极低延迟和词法补充价值，但在中文 Query → 英文 Evidence 场景下存在明显失效，因此不适合作为当前异构 Corpus 的主 Retriever。

---

# 13. Hybrid RRF Failure Analysis

Corpus V2 中：

```text
Dense
↓
Hybrid RRF
```

出现 4 个明确 Rank Degradation Case：

```text
R021
R022
R023
R027
```

---

## 13.1 R021

Query：

```text
OWASP 所说的直接提示词注入是什么？
```

Dense：

```text
Gold Rank1
RR = 1.0000
```

Hybrid：

```text
Gold Rank3
RR = 0.3333
```

---

## 13.2 R022

Query：

```text
LLM 从网页或文件读取到隐藏恶意指令并因此改变行为，
属于什么风险？
```

Dense：

```text
Gold Rank1
RR = 1.0000
```

Hybrid：

```text
Gold Rank3
RR = 0.3333
```

---

## 13.3 R023

Query：

```text
提示词注入 最小权限
```

Dense：

```text
Gold Rank1
RR = 1.0000
```

Hybrid：

```text
Gold Rank4
RR = 0.2500
```

---

## 13.4 R027

Query：

```text
FastAPI 中 Dependency Injection 是什么意思，
它能解决哪些问题？
```

Dense：

```text
Gold Rank1
RR = 1.0000
```

Hybrid：

```text
Rank1
Dependencies

Rank2
What is "Dependency Injection"
```

Gold：

```text
Rank2
RR = 0.5000
```

这里 Hybrid Rank1 并非完全无关，而是：

```text
更宽泛的 Parent Section
```

压过了：

```text
更精确的 Definition Section
```

---

# 14. Why RRF Can Hurt Ranking

RRF：

```text
score(d)
=
Σ 1 / (k + rank_i(d))
```

关注的是：

```text
Rank Position
```

而不是：

```text
Retriever 当前 Query 上是否可靠
```

因此：

```text
Strong Dense
+
Weak BM25
```

进入未加权 RRF 后，BM25 的高排名同样会贡献较大的 Fusion Score。

RRF 并不知道：

```text
BM25 当前为什么排 Rank1
```

是因为：

```text
真实语义匹配
```

还是：

```text
跨语言词法失配后的偶然高分
```

因此会出现：

```text
Strong Dense Signal
+
Weak Lexical Signal
↓
Unweighted RRF
↓
Weak Retriever Contaminates Fusion
```

这说明：

> Hybrid Retrieval 并不是 Dense + BM25 后天然更强。Fusion 会继承各个 Retriever 的正确排序，也会继承它们的错误。

---

# 15. Reranker Failure Recovery

Corpus V2 中：

```text
Hybrid RRF
↓
bge-reranker-v2-m3
```

成功修复 4 个 RRF Ranking Failure。

---

## 15.1 R021

```text
Hybrid Rank3
↓
Rerank Rank1
```

---

## 15.2 R022

```text
Hybrid Rank3
↓
Rerank Rank1
```

---

## 15.3 R023

```text
Hybrid Rank4
↓
Rerank Rank1
```

---

## 15.4 R027

```text
Hybrid Rank2
↓
Rerank Rank1
```

因此：

> Cross-Encoder Reranker 能重新结合 Query 和 Candidate Content，对 Fusion 后候选进行细粒度相关性判断，并修复部分弱 Retriever 引入的 Ranking Noise。

---

# 16. Reranker Is Useful, But Not Automatically Best

Corpus V2：

```text
Dense
Recall@1 = 0.9808
MRR@10   = 1.0000
```

而：

```text
Hybrid + Rerank
Recall@1 = 0.9423
MRR@10   = 0.9808
```

因此：

```text
Hybrid + Rerank
```

虽然明显优于：

```text
Hybrid RRF
```

但整体仍没有超过：

```text
Dense
```

所以当前不能得出：

```text
Reranker 一定让最终 Retrieval 更好
```

更准确的结论是：

> Reranker 在 Corpus V2 中具有明确的 Fusion Failure Repair 价值，但当前 Quality × Latency 综合表现仍不足以证明 Hybrid + Rerank 应无条件替代 Dense baseline。

---

# 17. Remaining Reranker Failure: R030

当前：

```text
Hybrid + Rerank Non-Top1 Gold Cases
```

只有：

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
fastapi_lifespan__lifespan_events_lifespan__0001
```

Reranker Top5：

```text
Rank1
Alternative Events (deprecated)
> startup and shutdown together

Rank2
Lifespan

Rank3
Lifespan Part 2

Rank4
Use Case

Rank5
Async Context Manager
```

Rank1 Section 并不是完全无关。

它讨论：

```text
startup / shutdown
resource acquire / release
recommended lifespan
```

而真正 Gold Section：

```text
Lifespan
```

更直接给出：

```text
lifespan parameter
@asynccontextmanager
yield 前加载
yield 后清理
```

因此该 Failure 更接近：

```text
高度相关 Supporting Section
vs
更精确的 Minimal Core Evidence
```

属于：

```text
Fine-grained Section Ranking Failure
```

而不是完全错误的跨文档 Retrieval。

当前没有为了这一条 Case：

```text
修改 Gold
修改 Query
调整 RRF k
调整 Retriever TopK
修改 Reranker 参数
```

因为这会增加：

```text
Evaluation Overfitting
```

风险。

---

# 18. Evaluation Leakage Policy

当前项目明确避免以下流程：

```text
Retriever 找到了 X
↓
把 X 加进 Gold
↓
指标提高
```

以及：

```text
看到一个 Evaluation Failure
↓
立刻调 RRF / TopK / Weight
↓
在同一个 Dataset 上宣称提升
```

当前 Evaluation 原则为：

```text
Measure
↓
Inspect
↓
Explain
↓
Then Tune
```

如果未来需要正式优化：

```text
RRF Weight
Dense / BM25 Weight
Candidate TopK
Reranker Configuration
```

应该先建立：

```text
Development Set
+
Held-out Test Set
```

然后：

```text
Dev
→ 调参

Held-out
→ 最终验证
```

当前 Retrieval Eval V2 暂不承担：

```text
调参集
+
最终测试集
```

双重职责。

---

# 19. Retrieval Failure Inspector

当前 Retrieval Evaluation 不再只输出 Aggregate Metrics。

还支持：

```text
Dense → Hybrid Rank Degradation

Hybrid → Rerank Rank Improvement

Dense Non-Top1 Gold Cases

BM25 Gold Misses @10

Hybrid RRF Non-Top1 Gold Cases

Hybrid + Rerank Non-Top1 Gold Cases
```

这样 Retrieval Failure 可以定位到：

```text
Query ID
Query
Role
Gold
RR
Recall@K
Top-N Retrieved Chunk IDs
```

因此问题从：

```text
“Hybrid 指标下降了”
```

进一步变成：

```text
哪一个 Query？
↓
哪一个 Gold？
↓
原来排第几？
↓
融合后排第几？
↓
什么错误候选压到了前面？
↓
Root Cause 是 lexical mismatch、
parent section competition，
还是其他问题？
```

这是当前 Evaluation-driven Development 的核心能力之一。

---

# 20. Retrieval Latency Benchmark

早期按 Method Block 顺序执行 Retrieval Benchmark 时观察到明显：

```text
GPU Warm-State
Method Order Bias
```

因此最终正式采用：

```text
Query-level Interleaved Benchmark
```

而不是：

```text
Dense 全部跑完
↓
BM25 全部跑完
↓
Hybrid 全部跑完
↓
Rerank 全部跑完
```

---

## 20.1 Benchmark Design

对每个 Query：

```text
Dense
BM25
Hybrid RRF
Hybrid + Rerank
```

采用随机顺序执行。

通过：

```text
Query-level Interleaving
+
Deterministic Shuffle
+
Repeated Runs
+
CUDA Synchronization
+
P50 / P95
```

减少：

```text
Method Order Bias
GPU Warm-State Bias
```

---

## 20.2 Benchmark Conditions

```text
Queries        : 20
Rounds         : 5
Samples/method : 100
Warmup         : 2
GPU            : RTX 4060 Laptop
```

Benchmark 不包含：

```text
Model Initialization
FastAPI
HTTP
LLM Generation
```

---

## 20.3 Results

| Method | Mean (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.39 | 0.32 | 0.52 | 0.09 | 6.09 |
| Dense | 53.57 | 52.20 | 75.44 | 30.35 | 98.18 |
| Hybrid RRF | 56.54 | 54.71 | 77.51 | 33.48 | 106.12 |
| Hybrid + Rerank | 452.33 | 455.68 | 503.99 | 348.80 | 538.37 |

关系：

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

---

## 20.4 Important Measurement Boundary

`run_retrieval_eval.py` 的 Ablation Summary 中仍会显示：

```text
Mean Latency(ms)
```

但这些值只是当前 Evaluation Run 的：

```text
附带诊断数据
```

它们不是正式 Latency Benchmark。

正式 Retrieval Latency 结论始终使用：

```text
Query-level Interleaved Benchmark
```

因此不同运行中偶尔出现：

```text
Hybrid mean < Dense mean
```

等现象，不应被用于推翻正式 Interleaved Benchmark。

---

# 21. ACL Evaluation

当前系统支持：

```text
guest
developer
admin
```

权限关系：

```text
guest
→ public

developer
→ public + developer

admin
→ public + developer + admin
```

ACL 发生在：

```text
Candidate Generation
```

之前。

正确链路：

```text
User Role
↓
AccessContext
↓
Allowed Access Levels
↓
┌─────────────────────────────┐
│ Qdrant Payload Filter       │
│ Authorized BM25 Corpus      │
└─────────────────────────────┘
↓
Retrieval
↓
Fusion
↓
Rerank
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

如果 Unauthorized Chunk 先参与：

```text
Top-K Competition
Fusion
Rerank
```

即使最后删除，也可能已经影响合法结果排序。

---

## 21.1 Corpus V2 ACL Smoke Test

FastAPI 文档在项目中人为标记：

```text
developer
```

用于模拟内部技术文档。

Query：

```text
How does FastAPI dependency injection work?
```

Guest：

```text
Top5
→ public OWASP chunks
→ no FastAPI developer chunks
```

Developer：

```text
Top5
→ fastapi_dependencies
```

其中 Gold-like Relevant Section：

```text
Dependencies
>
What is "Dependency Injection"
```

进入 Top Result。

另一个 Query：

```text
How should I load a shared machine learning model
during FastAPI startup
and clean it up during shutdown?
```

Guest：

```text
no FastAPI developer chunks
```

Developer：

```text
Top results
→ fastapi_lifespan
```

因此：

> ACL 已经真实作用于 Retrieval Candidate Space，而不是只在最终 Response 阶段过滤。

---

## 21.2 ACL != Answerability

需要区分：

```text
知识库没有答案
```

与：

```text
知识库有答案
但当前角色无权访问
```

后者属于：

```text
Authorization Negative
```

不能简单伪装成：

```text
hard_negative
```

因此当前 Retrieval Eval V2 主要通过：

```text
role-aware positive cases
```

验证合法 Candidate Space。

未来计划建立独立：

```text
ACL-specific Evaluation
```

指标可以包括：

```text
Forbidden Hit Count
Unauthorized Retrieval Rate
ACL Leakage Rate
```

---

# 22. Retrieval Relevance != Answerability

开发过程中曾分析：

```text
Top-1 Rerank Score
```

能否直接作为：

```text
Answerability Threshold
```

实验发现：

```text
Answerable Query
```

与：

```text
Hard Negative Query
```

的 Rerank Score 存在明显重叠。

例如：

```text
生成式人工智能服务管理暂行办法
规定发现违法内容后
必须在几小时内处理？
```

Retriever 可以找到：

```text
违法内容
及时处理
整改
报告
```

高度相关条款。

但是 Evidence 中并没有：

```text
具体小时数
```

因此：

```text
Evidence Relevant
```

不等于：

```text
Evidence Sufficient
```

最终不采用：

```text
rerank_score > threshold
=> answerable
```

这种单层决策。

当前架构：

```text
Coarse Relevance Gate
↓
过滤明显 Out-of-Domain
↓
Evidence-Constrained Generation
↓
判断 Evidence 是否足够
```

关键结论：

> Reranker 主要判断 Query 与 Evidence 的相关程度；Answerability 判断 Evidence 是否包含回答问题所需的充分事实。这是两个不同任务。

---

# 23. Full-RAG Evaluation Scope

需要特别区分：

```text
Retrieval Eval V2
```

与：

```text
Full-RAG Answer Eval
```

当前最新 Retrieval Evaluation 已经使用：

```text
34-case V2
```

但当前最新 Full-RAG：

```text
Answer / Refusal
Citation
Prompt v1 → v2
```

实验仍然基于：

```text
V1 对应的 20 条 Seed Cases
```

尚未对 Retrieval Eval V2 的全部：

```text
34 cases
```

执行完整 LLM Re-evaluation。

因此后文的：

```text
Decision Accuracy
Refusal Accuracy
Citation Metrics
```

均属于：

```text
Full-RAG Eval V1
```

阶段结果。

---

# 24. Answer / Refusal Evaluation

Full-RAG Pipeline：

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

当前：

```text
Decision Failure = 0
```

但需要强调：

> 该结果来自 20 条 Seed Dataset 的单次 Full-RAG Evaluation，只用于开发阶段验证，不能解释为生产环境准确率达到 100%。

---

# 25. Citation Evaluation Design

初版 Citation Evaluation 曾直接复用：

```text
Retrieval Gold
```

作为 Citation Gold。

Failure Analysis 后发现：

```text
Retrieval Gold
≠
Citation Gold
```

例如：

Retrieval Evaluation 可能只关心：

```text
核心条款是否被召回
```

Citation Evaluation 则关心：

```text
最终答案实际引用的所有 Evidence
是否都直接支持答案
```

因此正式拆分为：

```text
gold_chunk_ids
→ Retrieval Gold

citation_gold_chunk_ids
→ Citation Gold
```

并加入：

```text
strict_citation_eval
```

---

# 26. Citation Gold Audit

人工重点 Audit：

```text
R001
R006
R008
R010
R014
```

发现原始 Citation mismatch 实际混合了三种不同问题：

```text
True Over-Citation

Gold Annotation Incomplete

Ambiguous Query Scope
```

因此不能把：

```text
Actual Citation - Retrieval Gold
```

全部判为模型错误。

---

## 26.1 R001

Query：

```text
生成式人工智能服务处理训练数据
需要遵守什么规定？
```

严格 Citation Gold：

```text
生成式人工智能服务管理暂行办法 第七条
```

其他相关条款虽然主题相关，但不是该问题的核心最小证据。

---

## 26.2 R006

Query：

```text
深度合成服务提供者
提供人脸、人声编辑功能时有什么要求？
```

Citation Gold Audit 后包括：

```text
第十四条
第十五条
第十六条
第十七条
```

说明：

> Multiple Citations 并不一定代表 Over-Citation。

如果多个不同 Evidence 分别支持多个独立义务，就应该保留。

---

## 26.3 R008

Citation Gold：

```text
深度合成 第七条
深度合成 第十五条
```

属于多个必要证据。

---

## 26.4 R010

Query：

```text
违法内容怎么处理？
```

可能由：

```text
生成式 AI 第十四条
深度合成 第十条
```

分别提供合理 Evidence。

但 Query 没有限定具体法规。

因此：

```text
strict_citation_eval = false
```

---

## 26.5 R014

Query：

```text
训练语料有哪些合规要求？
```

范围较宽。

人工 Audit 后 Citation Gold：

```text
生成式 AI 第七条
生成式 AI 第四条
深度合成 第十四条
```

但同样属于：

```text
strict_citation_eval = false
```

---

# 27. Prompt v1 Citation Results

修正 Citation Annotation 后，Prompt v1：

| Metric | Prompt v1 |
|---|---:|
| All-case Citation Precision | 0.9524 |
| All-case Citation Recall | 0.9821 |
| Citation Hit Rate | 1.0000 |
| Strict Citation Precision | 0.9444 |
| Strict Citation Recall | 0.9792 |
| Strict Citation Hit Rate | 1.0000 |

主要 Strict Failure：

```text
R001
```

Prompt v1 引用了：

```text
核心第七条
+
相关第四条
+
其他法规第十四条
```

属于：

```text
Over-Citation
```

---

# 28. Prompt v2 Optimization

R001 的 Retrieval 已经正确。

因此没有修改：

```text
Dense
BM25
RRF
Reranker
Evidence Gate
```

问题定位为：

```text
Generation / Citation Selection
```

Prompt v2 增加：

```text
Minimal Sufficient Evidence Set
+
Scope Matching
```

规则：

1. 只引用直接支持最终答案实际陈述的 Evidence；
2. 主题相关不等于需要引用；
3. 一个 Evidence 已经充分支持某事实时，不重复增加类似 Evidence；
4. 多个不同 Evidence 分别支持多个独立事实时，应保留多个必要 Citation；
5. 用户明确限定法规、制度或场景时，应优先匹配同范围 Evidence；
6. Evidence ID 只出现在结构化 Citation 字段，不进入最终 Answer 正文。

目标不是：

```text
Citation 越少越好
```

而是：

```text
Minimal Sufficient Evidence
```

---

# 29. Targeted Prompt Regression

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
→ Over-Citation Failure

R006
→ Multiple Independent Obligations Control Case

R008
→ Multi-Evidence Control Case
```

目标是防止：

```text
为了修复 R001
↓
Prompt 退化为
“所有问题只能引用一个 Evidence”
```

结果：

```text
R001
→ 收敛到核心 Citation

R006
→ 仍保留多个合理 Evidence

R008
→ 仍保留两个必要 Evidence
```

因此：

```text
Minimal Sufficient Evidence
```

并不等于：

```text
One Citation Only
```

---

# 30. Prompt v2 Full-RAG Re-evaluation

Prompt v2 在完整：

```text
20-case V1 Full-RAG Dataset
```

重新运行。

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

# 31. Why All-case Citation Recall Decreased

Prompt v2：

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

主要来自：

```text
R010
```

R010 Citation Gold：

```text
生成式 AI 第十四条
深度合成 第十条
```

Prompt v2 实际只引用：

```text
生成式 AI 第十四条
```

即可完整回答。

因此：

```text
Precision = 1.0
Recall    = 0.5
```

但这并不一定是 Citation Failure。

因为 R010 没有限定法规。

两个 Gold 更接近：

```text
Multiple Acceptable Evidence Sources
```

而不是：

```text
Both Citations Are Mandatory
```

因此：

```text
strict_citation_eval = false
```

严格 Citation 指标中没有该 mismatch。

---

# 32. Evaluation Snapshot Persistence

真实 LLM Evaluation 会受到：

```text
API Cost
Network Latency
Generation Randomness
```

影响。

因此项目不会在每次：

```text
修改 Metrics
修改 Failure Analysis
修改 Citation Annotation
```

后都重新调用 LLM。

当前流程：

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

代表性结果：

```text
data/eval/results/
├── answer_eval_v1_run_001.jsonl
├── answer_eval_v1_run_001_audited.jsonl
└── answer_eval_v1_run_002.jsonl
```

含义：

```text
run_001
→ Prompt v1 原始 Full-RAG Result

run_001_audited
→ Prompt v1 Output
  + 最新 Citation Annotation

run_002
→ Prompt v2 Full-RAG Result
```

这样可以避免：

```text
只改分析逻辑
↓
重新跑 LLM
↓
模型输出变化
↓
实验失去可比性
```

---

# 33. Reproduction Commands

## 33.1 Frozen Retrieval Regression V1

```powershell
python scripts/run_retrieval_eval.py --dataset v1
```

用途：

```text
Frozen Regression Benchmark
```

主要验证：

```text
Corpus Expansion
Schema Evolution
ACL Evaluation Upgrade
```

之后旧 Retrieval 能力是否发生 Regression。

---

## 33.2 Retrieval Capability Evaluation V2

```powershell
python scripts/run_retrieval_eval.py --dataset v2
```

用于当前 Corpus V2：

```text
Dense
BM25
Hybrid RRF
Hybrid + Rerank
```

比较。

主要输出：

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
```

同时输出：

```text
Dense → Hybrid Rank Degradations

Hybrid → Rerank Rank Improvements

Dense Non-Top1 Gold Cases

BM25 Gold Misses @10

Hybrid RRF Non-Top1 Gold Cases

Hybrid + Rerank Non-Top1 Gold Cases
```

---

## 33.3 Retrieval Latency Benchmark

早期 Blocked Benchmark：

```text
scripts/run_retrieval_latency_benchmark.py
```

保留用于实验历史。

正式推荐：

```powershell
python scripts/run_interleaved_retrieval_latency_benchmark.py
```

采用：

```text
Query-level Interleaving
Deterministic Shuffle
CUDA Synchronization
Repeated Runs
P50 / P95
```

README 与 Evaluation Report 中的正式 Retrieval Latency 均以该实验为准。

---

## 33.4 Inspect Evaluation Dataset

```powershell
python scripts/inspect_retrieval_eval_dataset.py
```

用于检查：

```text
Case 数量
Answerable / Unanswerable
Category
Gold Annotation
```

---

## 33.5 Inspect Corpus

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
Near-max Chunk
Multi-part Section
```

---

## 33.6 Full-RAG Answer / Refusal Evaluation

运行前需要：

```text
Qdrant
BGE-M3
bge-reranker-v2-m3
SILICONFLOW_API_KEY
```

执行：

```powershell
python scripts/run_answer_eval.py
```

当前该实验仍对应：

```text
V1 20-case Full-RAG Dataset
```

执行：

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
Snapshot Persistence
```

---

## 33.7 Targeted Citation Regression

```powershell
python scripts/run_targeted_citation_regression.py
```

当前主要测试：

```text
R001
R006
R008
```

验证：

```text
修复 Over-Citation
+
不破坏合理 Multi-Evidence Citation
```

---

## 33.8 Offline Snapshot Analysis

```powershell
python scripts/analyze_answer_eval_snapshot.py
```

该脚本不会：

```text
加载 BGE-M3
加载 Reranker
查询 Qdrant
调用 SiliconFlow
```

只读取：

```text
Evaluation Snapshot
```

因此可低成本重复执行：

```text
Metrics
Failure Analysis
Citation Audit
```

---

## 33.9 Prompt v1 / Prompt v2 Comparison

```powershell
python scripts/compare_answer_eval_snapshots.py
```

比较：

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

整个过程为 Offline Analysis，不调用 LLM。

---

## 33.10 Citation Gold Migration

```powershell
python scripts/migrate_citation_gold.py
```

用于将旧 Evaluation Dataset Annotation 升级为：

```text
Retrieval Gold
+
Citation Gold
+
strict_citation_eval
```

该脚本属于：

```text
Dataset Migration Tool
```

而不是日常 Evaluation Command。

---

## 33.11 Snapshot Annotation Enrichment

```powershell
python scripts/enrich_answer_eval_snapshot.py
```

用于将新的 Citation Annotation 合并进历史 Snapshot。

只更新：

```text
citation_gold_chunk_ids
strict_citation_eval
```

不会修改：

```text
answer
cited_chunk_ids
reason
rerank score
latency
```

因此不需要重新调用 LLM。

---

# 34. Current Retrieval Conclusions

当前 Corpus V2 可以得到以下 Retrieval 结论。

---

## 34.1 Dense

```text
Dense
```

是当前 104-Chunk 中英异构 Corpus 上表现最稳定的 Baseline。

26 条 Answerable Query：

```text
Dense Non-Top1 Gold Cases = 0
```

说明：

> 每条 Query 至少有一个核心 Gold 位于 Rank1。

BGE-M3 还能够稳定完成：

```text
中文 Query
→
英文 OWASP / FastAPI
```

跨语言语义检索。

---

## 34.2 BM25

BM25：

```text
Latency 极低
```

但 Corpus V2 中出现 6 个：

```text
Gold Miss @10
```

并且均与：

```text
中文 Query
→
英文 Evidence
```

高度相关。

因此：

> BM25 可以作为低成本 lexical complement，但在当前中英混合 Corpus 中不能作为主 Retriever。

---

## 34.3 Hybrid RRF

Hybrid RRF 在：

```text
R021
R022
R023
R027
```

把 Dense Rank1 Gold 推迟到：

```text
Rank2–4
```

因此：

> Hybrid 并不天然优于 Dense。

Rank-based Fusion 能解决 Score Scale 不一致问题，但不能自动判断：

```text
某个 Retriever
在当前 Query 上是否可靠
```

---

## 34.4 Rerank

Reranker 成功修复：

```text
R021
R022
R023
R027
```

四个 RRF Failure。

因此 Reranker 具有明确：

```text
Fusion Noise Repair
```

价值。

但整体：

```text
Hybrid + Rerank
```

仍未超过：

```text
Dense
```

并且正式 Latency Benchmark 中额外成本明显。

所以：

> Rerank 是否应该始终开启，需要由真实 Quality × Latency Evaluation 决定，而不是因为 Cross-Encoder 架构更复杂就默认认为更优。

---

# 35. Current ACL Conclusions

当前 ACL 实验支持：

```text
guest
→ public

developer
→ public + developer

admin
→ all
```

并确认：

```text
ACL Filter
```

发生在：

```text
Candidate Generation
```

之前。

这避免 Unauthorized Chunk 参与：

```text
Top-K Competition
Fusion
Rerank
```

当前 API 中：

```text
role
```

仍然只是 Demo Input。

生产系统需要由可信：

```text
SSO
JWT
IAM
```

产生 AccessContext。

---

# 36. Current Evidence / Refusal Conclusions

当前实验支持：

```text
Retrieval Relevance
≠
Answerability
```

因此没有使用：

```text
Single Rerank Threshold
```

直接决定：

```text
Answer / Refuse
```

而是：

```text
Coarse Relevance Gate
+
Evidence-Constrained Generation
```

无充分证据时：

```text
Structured Refusal
```

而不是由 LLM 使用外部常识补充缺失事实。

---

# 37. Current Citation Conclusions

当前 Citation Evaluation 得到：

```text
Retrieval Gold
≠
Citation Gold
```

以及：

```text
More Citations
≠
Safer Answer
```

同时：

```text
Fewer Citations
≠
Better Answer
```

真正目标是：

```text
Minimal Sufficient Evidence Set
```

即：

> 用最少但充分的 Evidence 完整支持最终答案中的真实陈述。

---

# 38. Current Evaluation Methodology

当前项目不使用：

```text
“回答看起来不错”
```

作为主要优化依据。

而是形成：

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

另外明确区分：

```text
Regression Benchmark
```

与：

```text
Current Capability Benchmark
```

避免 Corpus Expansion 后丢失历史对照。

当前也明确坚持：

```text
Gold
```

必须来自：

```text
Independent Corpus / Source Audit
```

而不是：

```text
Current Retriever / LLM Output
```

---

# 39. Limitations

当前 Evaluation 仍存在明显限制：

1. 当前 Corpus 只有 5 篇文档、104 个 Chunk；
2. OWASP 当前只加入 LLM01 Prompt Injection，尚未覆盖完整 LLM Top 10；
3. 尚未加入 Qdrant 官方技术文档；
4. Retrieval Eval V2 只有 34 条 Case，其中 Answerable Query 26 条；
5. 当前 V2 尚未拆分独立 Development Set 和 Held-out Test Set；
6. 因此当前没有基于 V2 调整 RRF Weight、TopK 或其他 Retrieval 参数；
7. 最新 Full-RAG Answer / Refusal / Citation Evaluation 仍基于 V1 的 20 条 Cases；
8. 尚未在完整 Retrieval Eval V2 上重新运行 Full-RAG Evaluation；
9. Prompt v1 / v2 当前均主要依赖单次 Full-RAG Snapshot；
10. 尚未系统评估自然语言 Answer Correctness；
11. 尚未完成独立完整 Faithfulness Evaluation；
12. 尚未建立专门的 ACL Unauthorized Retrieval Benchmark；
13. 尚未进行正式高并发 API Load Test；
14. Citation Gold 仍由人工小规模维护；
15. 当前 ACL Access Level 中存在人为模拟的 developer 文档；
16. 当前 Generic Technical Document Pipeline 尚未实现完整 Code-aware Chunking；
17. 当前结果不能解释为生产环境准确率、可靠性保证或 SLA。

因此所有百分比结果均应理解为：

> 当前 Seed Evaluation Dataset 上的开发阶段实验结果。

---

# 40. Next Evaluation Work

随着知识库继续扩展，后续计划：

```text
更多 OWASP LLM Top 10
↓
Qdrant 官方文档
↓
更多公开 / 模拟内部规范
↓
30–80 Documents
↓
50–100 Evaluation Queries
↓
Development Set
+
Held-out Test Set
↓
ACL-specific Evaluation
↓
More Hard Negatives
↓
Cross-document Questions
↓
Full-RAG Evaluation V2
↓
Answer Correctness
↓
Faithfulness
↓
Latency / Quality Trade-off
```

届时重点继续验证：

```text
Dense vs Hybrid
Reranker Real Gain
Cross-lingual Retrieval
ACL Leakage
Refusal Robustness
Citation Quality
Answer Correctness
Faithfulness
Latency / Quality Trade-off
```

最终目标不是追求某个单一：

```text
100%
```

指标，而是建立一套：

```text
可复现
可解释
可定位 Failure
可比较改动
可验证 Regression
```

的 RAG Evaluation Pipeline。