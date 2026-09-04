# RAG Evaluation Report

本文档记录「企业 AI 合规与应用规范助手」当前阶段的评测设计、实验结果、Failure Analysis 与 Evaluation Infrastructure 演进。

> 当前指标用于验证作品级 RAG 系统的设计与迭代闭环，不代表生产环境准确率，也不构成 SLA。

---

# 1. Evaluation Goals

当前系统不只评估“回答看起来是否正确”，而是拆成多个可独立验证的层次：

1. Manifest 中启用的文档是否真正进入 Corpus；
2. Chunk 是否满足结构与字段约束；
3. Retrieval 是否找回核心 Evidence；
4. Corpus 扩展后旧能力是否发生 Regression；
5. Dense / BM25 / Hybrid / Rerank 的收益分别是什么；
6. ACL 是否在 Candidate Generation 前限制搜索空间；
7. Retrieval Relevance 是否等于 Answerability；
8. Coarse Evidence Gate 是否只过滤明显无关问题；
9. Full-RAG 是否正确 Answer / Refuse；
10. Citation 是否正确且必要；
11. Evaluation Harness 是否真正遵守 Dataset 中的 role；
12. 昂贵 LLM Evaluation 中断后是否可以安全恢复。

当前评测链路：

```text
Document Manifest
        ↓
Manifest → Corpus Coverage Check
        ↓
Chunk Validation
        ↓
Evaluation Dataset
        ↓
Gold Integrity Check
        ↓
Role-aware Retrieval Evaluation
        ↓
Retrieval Quality / Regression
        ↓
Failure Inspector / Latency Benchmark
        ↓
Evidence Gate Audit
        ↓
Full-RAG Answer / Refusal
        ↓
Citation Evaluation
        ↓
Case-level Checkpoint
        ↓
Immutable Final Snapshot
        ↓
Offline Failure Analysis
```

---

# 2. Current Corpus

项目最初只有：

```text
2 篇中文法规
49 regulation chunks
```

当前最终 Corpus：

```text
Enabled Documents           : 28
Corpus Documents            : 28
KnowledgeChunks             : 835
Qdrant Points               : 835
Manifest → Corpus Coverage  : PASS
Chunk Validation            : PASS
Dense ACL Check             : PASS
Full Retrieval ACL Check    : PASS
```

当前主要文档类型：

```text
中文法规与政策规范
OWASP LLM Top 10
FastAPI 官方文档
Qdrant 官方文档
```

FastAPI / Qdrant 文档本身是公开资料，但项目将其标记为：

```text
access_level = developer
```

仅用于模拟企业内部技术规范的 ACL 场景，不表示这些官方文档真实存在访问限制。

---

# 3. Corpus Coverage ≠ Chunk Validity

Corpus 扩展过程中曾出现真实问题：

```text
Manifest enabled = true
↓
HTML 下载成功
↓
NormalizedDocument 成功
↓
Parser 生成 0 Article
↓
最终生成 0 Chunk
```

已有 Chunk Validation 仍然可能 PASS，因为它只检查：

```text
“已经生成出来的 Chunk 是否有效”
```

而不会检查：

```text
“每一篇 enabled 文档是否至少生成了一个 Chunk”
```

因此新增：

```text
Manifest → Corpus Coverage Validation
```

最终结论：

> Chunk-level validity 不等于 Manifest-to-Corpus completeness。知识库构建必须同时验证“Chunk 是否有效”和“启用文档是否真正进入 Corpus”。

---

# 4. Evaluation Dataset Versioning

## 4.1 V1 — Frozen Seed Regression Benchmark

路径：

```text
data/eval/retrieval_eval_v1.jsonl
```

规模：

```text
20 cases
14 answerable
6 unanswerable
```

V1 建立时 Corpus 主要是 2 篇中文法规、49 chunks。其 Gold 已冻结，不因后续 Corpus 扩展或 Retriever 新排序而修改。

## 4.2 V2 — Corpus V2 Capability Benchmark

路径：

```text
data/eval/retrieval_eval_v2.jsonl
```

规模：

```text
34 cases
26 answerable
8 unanswerable
```

V2 增加 OWASP、FastAPI、developer-role cases 与新的 hard negatives。

## 4.3 V3 — Final Capability Benchmark

路径：

```text
data/eval/retrieval_eval_v3.jsonl
```

规模：

```text
46 cases
36 answerable
10 unanswerable
```

V3 增加更多中文法规、完整 OWASP LLM Top 10、Qdrant 技术文档、ACL same-query negative、多证据问题与更强 hard negatives。

V3 当前保持冻结，不根据 Final Benchmark 输出反向修改 Gold 或 Retrieval 参数。

---

# 5. Gold Annotation Design

当前 Evaluation Case 主要字段：

```text
query_id
query
gold_chunk_ids
category
answerable
note
citation_gold_chunk_ids
strict_citation_eval
role
```

其中：

```text
gold_chunk_ids
```

用于 Recall@K / MRR，表示 Retriever 应优先找回的核心 Evidence。

```text
citation_gold_chunk_ids
```

用于 Citation Precision / Recall / Hit Rate，表示最终答案允许直接引用、且能够支持答案陈述的 Evidence。

因此：

```text
Retrieval Gold != Citation Gold
```

`strict_citation_eval` 用于区分 Citation Gold 边界明确的 Case 与存在多个合理证据组合的 Case。

正式评测前还会校验 Gold 是否真实存在于 `data/processed/chunks.jsonl`，防止 Chunk ID 漂移或人工拼写错误造成假的 Failure。

---

# 6. Retrieval Methods

## 6.1 Dense

```text
Query
↓
BAAI/bge-m3
↓
1024-d Embedding
↓
ACL Qdrant Filter
↓
Qdrant Dense Search
```

Distance：Cosine。

## 6.2 BM25

```text
Query
↓
Jieba Tokenization
↓
Authorized BM25 Corpus
↓
BM25
```

主要依赖 lexical overlap。

## 6.3 Hybrid RRF

```text
Dense Top20
+
BM25 Top20
↓
Reciprocal Rank Fusion
↓
Hybrid Top20
```

当前：

```text
score(d) = Σ 1 / (60 + rank(d))
k = 60
```

没有额外人工 Weight。

## 6.4 Hybrid + Rerank

```text
Dense Top20
+
BM25 Top20
↓
RRF
↓
Hybrid Top20
↓
BAAI/bge-reranker-v2-m3
↓
Final Ranking
```

Rerank raw score 用于排序、Coarse Gate 与 Failure Analysis，不解释为概率。

---

# 7. Retrieval Regression and Final V3 Results

## 7.1 Frozen V1 on Final 835-Chunk Corpus

14 条 answerable case：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 0.8214 | 0.8929 | 1.0000 | 1.0000 | 0.9071 |
| Hybrid RRF | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid + Rerank | 0.9643 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

结论：

> Corpus 从 49 Chunk 扩展到 835 Chunk 后，冻结 V1 上没有观察到 Retrieval Regression。

## 7.2 V2 on Final Corpus

26 条 answerable case：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9038 | 1.0000 | 1.0000 | 1.0000 | 0.9487 |
| BM25 | 0.5577 | 0.7115 | 0.7692 | 0.7692 | 0.6487 |
| Hybrid RRF | 0.8654 | 0.9231 | 0.9615 | 1.0000 | 0.9179 |
| Hybrid + Rerank | 0.9038 | 1.0000 | 1.0000 | 1.0000 | 0.9615 |

典型 Dense → RRF 退化：

```text
R022 : Rank3 → Rank5
R024 : Rank3 → Rank6
R027 : Rank1 → Rank2
```

典型 RRF → Rerank 修复：

```text
R022 : Rank5 → Rank2
R024 : Rank6 → Rank1
R027 : Rank2 → Rank1
```

## 7.3 Final Retrieval V3

V3 共有 36 条 answerable case：

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.8472 | 0.9722 | 1.0000 | 1.0000 | 0.9130 |
| BM25 | 0.5139 | 0.6528 | 0.7083 | 0.7778 | 0.6051 |
| Hybrid RRF | 0.7639 | 0.9028 | 0.9444 | 1.0000 | 0.8509 |
| Hybrid + Rerank | 0.8472 | 0.9861 | 0.9861 | 1.0000 | 0.9259 |

V3 Dense → RRF Rank Degradation：

```text
R022
R024
R027
R035
R040
R041
R042
```

V3 RRF → Rerank Improvement：

```text
R022
R024
R027
R035
R040
R041
R042
```

---

# 8. Retrieval Conclusions

## 8.1 Dense

BGE-M3 是当前中英混合 Corpus 上最稳定的基础 Retriever，并能够完成：

```text
中文 Query
→ 英文 OWASP / FastAPI / Qdrant Evidence
```

的跨语言语义检索。

## 8.2 BM25

V3 BM25：

```text
Recall@10 = 0.7778
MRR@10    = 0.6051
```

典型 Top10 Gold Miss：

```text
R021
R022
R023
R024
R025
R030
R042
R043
```

这些 Failure 大量集中在中文 Query → 英文 Evidence。

因此：

> BM25 延迟极低，适合作为 lexical complement，但在当前中英混合 Corpus 中不能作为主 Retriever。

## 8.3 Hybrid RRF

RRF 关注 Rank Position，而不知道某个 Retriever 在当前 Query 上是否可靠。

因此：

```text
Strong Dense
+
Weak BM25
↓
Unweighted RRF
↓
Weak branch may contaminate fusion
```

V3：

```text
Dense MRR      = 0.9130
Hybrid RRF MRR = 0.8509
```

结论：

> Hybrid Retrieval 并不天然优于 Dense。Equal-weight RRF 会融合互补信息，也会继承弱 Retriever 的错误。

## 8.4 Reranker

V3：

```text
Hybrid RRF MRR      = 0.8509
Hybrid+Rerank MRR   = 0.9259
```

因此 Reranker 对 Fusion Noise 有明确修复价值。

但它并非单调改进，而且计算成本显著更高。

---

# 9. R044 — Local Relevance ≠ Evidence-Set Coverage

R044：

```text
Qdrant Hybrid Queries 支持哪些融合方式，RRF 的作用是什么？
```

这是一个复合问题，需要至少覆盖：

```text
融合方式
+
RRF 的作用
```

Retrieval V3 中，Reranker Top5 包含多个高度相关 RRF 局部 Chunk，但没有完整保留回答整个问题所需的 evidence set。

Full-RAG 最终：

```text
Gate reason       : passed
Actual answerable : false
```

拒答原因是 Evidence 只足以说明 RRF 的作用与公式，但不足以完整列出 Hybrid Queries 支持的融合方式。

结论：

> Cross-Encoder Reranker 优化的是 Query-Passage 局部相关性，不天然优化复合问题所需的最小充分证据集合。

即：

```text
Local Relevance != Evidence-Set Coverage
```

---

# 10. Retrieval Latency Benchmark

正式 Retrieval Latency 使用 Query-level Interleaved Benchmark，以减少 Method Order Bias 与 GPU Warm-State Bias。

Benchmark：

```text
Queries        : 20
Rounds         : 5
Samples/method : 100
Warmup         : 2
GPU            : RTX 4060 Laptop
```

不包含模型初始化、FastAPI、HTTP、LLM Generation。

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
Dense ≈ Hybrid RRF
<<<<
Hybrid + Rerank
```

Reranker 是当前 Retrieval Pipeline 的主要额外延迟来源。

---

# 11. ACL Evaluation

角色：

```text
guest     → public
developer → public + developer
admin     → public + developer + admin
```

ACL 在 Candidate Generation 前执行：

```text
User Role
↓
AccessContext
↓
Allowed Access Levels
↓
Qdrant Payload Filter / Authorized BM25 Corpus
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

原因是 Unauthorized Chunk 即使最终被删除，只要先参与 Top-K、Fusion 或 Rerank，就可能已经污染合法结果排序。

## 11.1 Same-Query ACL Pair

V3 特意设计：

```text
R042
R045
```

Query 完全相同：

```text
Qdrant 中 Payload Filter 是如何用于过滤查询结果的？
```

R042：

```text
role = developer
answerable = true
Top1 score = 4.4219
Gold hit Top5 = true
Gate = PASS
Full-RAG = Answer
```

R045：

```text
role = guest
answerable = false
Top1 score = -7.3516
Gold hit Top5 = false
Gate = REJECT
Full-RAG = Refuse
```

结论：

> ACL 是 Pre-Retrieval Candidate Filtering，而不是 Post-Generation Cleanup。同一 Query 在不同 Role 下进入不同 Candidate Space，并产生不同合法结果。

---

# 12. Evaluation Harness Role Bug

第一次 V3 Full-RAG 尝试 `run_001` 得到：

```text
TP 27
TN 10
FP 0
FN 9
```

9 个 FN 恰好全部是 developer 技术文档 Case：

```text
R027-R032 FastAPI
R042-R044 Qdrant
```

进一步 Gate Audit 发现这些 Case 的真实 rerank score 都明显高于 -3.0。

最终定位到 `answer_runner.py`：

```text
Dataset 中有 case.role
↓
Runner 却把所有 Case 统一按 guest 执行
```

因此：

```text
run_001 = INVALID
```

修复后增加 Role-aware Regression Test：

```text
guest case     → guest
developer case → developer
```

并测试显式 `access_context` override。

工程结论：

> Dataset 中存在 role 字段，不代表 Evaluation Harness 真正执行了 Role-aware behavior；评测基础设施本身也必须测试。

---

# 13. Evidence Gate Audit

当前 Coarse Gate：

```text
Top1 rerank_score
↓
threshold = -3.0
↓
pass / reject
```

职责不是最终 Answerability，而是便宜过滤明显 OOD 或 ACL 后无相关 Evidence 的 Query。

V3 Gate Audit：

```text
TP: 36
TN: 3
FP: 7
FN: 0
```

这里只表示 Gate 是否放行，不是最终 Full-RAG Decision。

Top1 Score 分布：

```text
Answerable positives:
count   = 36
min     = -2.0859
median  = 5.1738
mean    = 4.4044
max     = 7.7852

Unanswerable negatives:
count   = 10
min     = -10.4453
median  = 0.9504
mean    = -1.0861
max     = 4.8008
```

关键：

```text
Lowest positive Top1  = -2.0859
Highest negative Top1 = 4.8008
Positive / Negative overlap = YES
```

因此：

> 单一 rerank raw score threshold 无法可靠区分 Answerable 与 Unanswerable。

---

# 14. Relevance != Answerability

典型 hard negative：

```text
深度合成服务必须使用哪一种内容审核算法？
```

Top1 rerank score：

```text
4.8008
```

虽然问题和 Evidence 高度相关，但 Evidence 并没有给出“必须使用某一种具体算法”的事实。

R046：

```text
根据现有法规，公司内部所有 RAG 系统都必须将用户对话日志至少保存三年吗？
```

Gate Audit：

```text
Top1 = -2.9219
Gate = PASS
```

但相关法规只能支持某些特定记录保存期限，不能推出“所有内部 RAG 对话日志都必须保存三年”。

最终 Full-RAG 正确 Refuse。

结论：

```text
Retrieval Relevance != Answerability
```

Reranker 判断相关性，Evidence Sufficiency 判断证据是否真的足够支撑用户要求的事实。

---

# 15. Structured Output Contract

Generation 要求 LLM 只能返回：

```json
{
  "answerable": true,
  "answer": "回答文本",
  "reason": "证据为何足够",
  "citations": ["E1"]
}
```

或者：

```json
{
  "answerable": false,
  "answer": null,
  "reason": "证据不足的原因",
  "citations": []
}
```

Parser 严格检查：

```text
JSON 合法性
字段类型
answerable / answer 一致性
reason 非空
citations 类型
Evidence ID 白名单
```

Parser 不会自动修坏 JSON、补 reason 或创造 Citation。

---

# 16. Full-RAG run_002 — Structured Output Failure

修复 Role-aware Evaluation 后执行 `run_002`，运行到 R032 时，LLM 输出没有满足：

```text
reason 必须为非空字符串
```

严格 Parser 正确抛出 `ValueError`。

因此：

```text
run_002 = ABORTED
```

这类 Failure 属于：

```text
Structured Output Contract Failure
```

不能伪装成：

```text
answerable = false
```

否则会污染业务指标。

---

# 17. Bounded Structured-Output Retry

当前保留 Strict Parser，不放宽；在 Answerer 中增加最多一次结构化输出重试：

```text
Generation Attempt 1
↓
Strict Parser
├─ PASS → return
└─ FAIL
    ↓
Structured Output Retry
    ↓
Strict Parser
    ├─ PASS → return
    └─ FAIL → raise
```

当前：

```text
max_parse_retries = 1
```

Retry 只针对输出协议失败，不会因为模型选择 Refuse 就强迫模型重新 Answer。

R032 单 Case 修复后验证：

```text
role        : developer
answerable  : true
gate_reason : passed
top_score   : 5.3711
citation    : fastapi_lifespan__...__sub_applications__0001
```

---

# 18. Case-Level Checkpoint / Resume

原始 Full-RAG Eval 只有：

```text
46 cases 全部完成
↓
一次性写 Final Snapshot
```

因此 run_002 在 R032 崩溃时，R001-R031 已经产生 LLM 成本，却没有留下可恢复结果。

当前改为：

```text
R001 完成 → partial snapshot
R002 完成 → partial snapshot
...
R046 完成 → completeness check
          → final snapshot
          → remove partial
```

Partial：

```text
answer_eval_v3_run_003.partial.jsonl
```

Final：

```text
answer_eval_v3_run_003.jsonl
```

中断后：

```powershell
python scripts/run_answer_eval.py --dataset v3 --run-id 003 --resume
```

Resume 只接受当前 Dataset 的连续前缀，防止乱序、缺洞、重复调用 LLM 或 Dataset 中途漂移。

工程结论：

> 有 Final Snapshot 不等于中途失败时已有昂贵推理结果不会丢；长时间 LLM Evaluation 需要 case-level durability。

---

# 19. Full-RAG Experiment Lineage

```text
run_001
INVALID
原因：Evaluation Runner 将所有 Case 错误地统一按 guest 执行，忽略 case.role。
```

```text
run_002
ABORTED
原因：R032 structured output 的 reason 字段违反 Parser Contract。
```

```text
run_003
VALID FINAL
包含：
Role-aware Evaluation
Strict Parser
Bounded Structured Output Retry
Case-level Checkpoint
Resume Safety Validation
Final Snapshot
```

保留这些失败实验，是为了记录 Evaluation Infrastructure 自身的 Failure 与修复过程。

---

# 20. Final Full-RAG V3 run_003

```text
Dataset : v3
Run ID  : 003
Cases   : 46
```

| Metric | Result |
|---|---:|
| TP | 35 |
| TN | 10 |
| FP | 0 |
| FN | 1 |
| Overall Decision Accuracy | 0.9783 |
| Answerable Accuracy | 0.9722 |
| Refusal Accuracy | 1.0000 |
| Hard Negative Refusal Accuracy | 1.0000 |
| OOD Refusal Accuracy | 1.0000 |

其中：

```text
10 / 10 Unanswerable
全部正确拒答
```

没有 False Positive。

对于“无依据必须拒答”的产品目标，这是当前最重要的 Full-RAG 结果之一。

唯一 FN：

```text
R044
```

它不是 Gate 误杀，而是进入 Evidence Sufficiency 后，因为最终 Top-K 没有覆盖完整 evidence set 而拒答。

---

# 21. Final Citation Metrics

## 21.1 All-case Citation

```text
Citation Cases : 35
Precision      : 0.8629
Recall         : 0.9690
Hit Rate       : 1.0000
```

## 21.2 Strict Citation

```text
Strict Cases   : 30
Precision      : 0.8944
Recall         : 0.9917
Hit Rate       : 1.0000
```

整体特点：

```text
Recall 很高
Hit Rate = 100%
Precision 相对较低
```

主要问题更偏向 Over-Citation，而不是完全引用错误。

---

# 22. Citation Failure Analysis

典型 Over-Citation：

## R011

Gold：

```text
生成式 AI 第十一条
```

Actual：

```text
生成式 AI 第十一条
+
网络安全法 第四十二条
```

后者相关，但超过 Strict Minimal Citation Gold。

## R022

Gold：Indirect Prompt Injection Definition。

Actual：Definition + Indirect Injection Scenario。

第二条是合理 supporting evidence，但不是 Strict Gold 中的最小集合。

## R024

Gold：LLM01 Human Approval。

Actual：LLM01 Human Approval + LLM06 User Approval。

说明模型倾向于把同主题补充证据一起引用。

## R030

Gold：FastAPI Lifespan Core Section。

Actual：Deprecated Startup/Shutdown Section + Lifespan Core + Lifespan Part 2。

这些 Evidence 并非无关，但存在明显冗余。

---

# 23. R006 — Finite Citation Gold Limitation

R006：

```text
深度合成服务提供者提供人脸、人声编辑功能时有什么要求？
```

Citation Gold：

```text
第十四条
第十五条
第十六条
第十七条
```

Actual：

```text
第十四条
第十五条
第十七条
```

对应：

```text
Precision = 1.0000
Recall    = 0.7500
```

但该次 Answer 只陈述：

```text
告知 + 单独同意
安全评估
显著标识
```

实际 Citation 已覆盖这些陈述。

因此：

> Citation Gold 是有限人工标注集合，不一定等价于某一次具体答案的唯一 Minimal Sufficient Evidence Set。

V3 Gold 保持冻结，不根据当前输出反向修改。

---

# 24. Citation Conclusion

当前实验支持：

```text
Retrieval Gold != Citation Gold
```

```text
More Citations != Safer Answer
```

```text
Fewer Citations != Better Answer
```

真正目标是：

```text
Minimal Sufficient Evidence Set
```

即用尽可能少、但足以完整支持最终答案实际陈述的 Evidence。

Prompt v2 已加入 Minimal Sufficient Evidence 与 Scope Matching，但 V3 仍存在 Over-Citation，因此当前将其作为 limitation，而不是继续围绕 Final Benchmark 调 Prompt。

---

# 25. Full-RAG End-to-End Timing

run_003：

```text
Total Eval Time      : 764.12s
Mean End-to-End Time : 16611.27ms/query
```

当前支持 Resume 后，总时间定义为：

```text
sum(per-case latency_ms)
```

该 Timing 包含：

```text
Dense
BM25
RRF
Rerank
Evidence Gate
SiliconFlow Network
LLM Generation
可能的 Structured Output Retry
```

因此：

> Full-RAG End-to-End Time 只作为诊断指标，不等价于正式 API Latency Benchmark。

正式 Retrieval Latency 仍以 Query-level Interleaved Benchmark 为准。

---

# 26. Evaluation Leakage Policy

当前项目明确避免：

```text
Retriever 找到 X
↓
把 X 加进 Gold
↓
指标提高
```

以及：

```text
看到 Final Benchmark Failure
↓
立即调 Weight / TopK / Prompt
↓
在同一个 Dataset 上宣称提升
```

原则：

```text
Measure
↓
Inspect
↓
Explain
↓
Tune on Dev Set
↓
Validate on Held-out
```

V3 已经作为 Final Capability Benchmark 使用，因此不会针对 R044 或 Citation mismatch 调参后继续用同一 V3 宣称“最终 100%”。

---

# 27. README-ready Conclusions

## Hybrid Retrieval

> BGE-M3 Dense 是当前中英混合 Corpus 上最稳定的基础 Retriever。Jieba BM25 在中文 Query → 英文技术文档场景明显偏弱；Equal-weight RRF 并不天然优于 Dense，弱 lexical branch 可能降低 Gold 排名。

## Rerank

> bge-reranker-v2-m3 能修复多起 RRF Rank Degradation，V3 MRR 从 0.8509 提升到 0.9259；但 Rerank 并非单调改进，而且代价明显更高。R044 进一步说明单 passage relevance 提升并不保证复合问题所需 evidence-set coverage。

## ACL

> ACL 在 Retrieval Candidate Generation 前生效，而不是生成后过滤。R042 / R045 使用相同 Query、不同 Role 验证：developer 能访问 Qdrant developer technical chunks 并回答；guest 无法进入该 Candidate Space，最终被拒答。

## Evaluation

> 项目将 Manifest Coverage、Chunk Validation、Retrieval、Answer / Refusal、Citation 分开评测，并保持 Retrieval Gold 与 Citation Gold 解耦。V3 Final Full-RAG 共 46 Case，Decision Accuracy 97.83%，Hard Negative / OOD Refusal 均为 100%，Citation Hit Rate 为 100%。

---

# 28. Current Engineering Lessons

```text
Chunk Validation != Manifest → Corpus Coverage
```

```text
Dataset 中有 role != Evaluation Harness 真正执行 role
```

```text
LLM 被要求输出 JSON != 每次都满足 Structured Output Contract
```

```text
有 Final Snapshot != 中途失败时已有昂贵推理结果不会丢
```

```text
Retrieval Relevance != Answerability
```

```text
Single-Passage Relevance != Evidence-Set Coverage
```

```text
Retrieval Gold != Citation Gold
```

---

# 29. Reproduction Commands

构建并验证 Corpus：

```powershell
python scripts/build_chunks.py
```

V1 Retrieval Regression：

```powershell
python scripts/run_retrieval_eval.py --dataset v1
```

V2 Retrieval：

```powershell
python scripts/run_retrieval_eval.py --dataset v2
```

V3 Retrieval：

```powershell
python scripts/run_retrieval_eval.py --dataset v3
```

正式 Retrieval Latency：

```powershell
python scripts/run_interleaved_retrieval_latency_benchmark.py
```

Evidence Gate Audit：

```powershell
python scripts/audit_evidence_gate_scores.py
```

Full-RAG V3：

```powershell
python scripts/run_answer_eval.py --dataset v3 --run-id 003
```

中断后恢复：

```powershell
python scripts/run_answer_eval.py --dataset v3 --run-id 003 --resume
```

已有 Final Snapshot 不允许覆盖；已有 Partial Snapshot 必须显式 `--resume`。

---

# 30. Current Limitations

1. Corpus 规模为 28 篇文档、835 Chunk，仍明显小于真实企业知识库；
2. V3 只有 46 Case，其中 Answerable 36 条；
3. 尚未构建更大规模独立 Development Set + Held-out Test Set；
4. 当前未对 RRF Weight、Retriever Weight、TopK 做系统调参；
5. Citation Gold 仍为人工小规模维护；
6. Citation Precision 仍受 Over-Citation 影响；
7. R044 表明当前 Reranker Top-K 对多证据问题的 Evidence-Set Coverage 仍有限；
8. 尚未系统评估自然语言 Answer Correctness；
9. 尚未建立独立 Faithfulness 自动评测；
10. 尚未进行正式高并发 API Load Test；
11. 当前技术文档 ACL 为作品中的模拟企业权限设计；
12. 当前 Full-RAG E2E latency 受网络和 LLM 服务影响，不作为 API SLA；
13. 当前结果不能解释为生产环境准确率或可靠性保证。

---

# 31. Next Evaluation Work

当前不继续针对 V3 做 Benchmark Tuning。

如果后续继续演进，优先方向：

```text
Independent Dev Set
+
New Held-out Test Set
↓
More Cross-document Questions
↓
More ACL-negative Pairs
↓
Evidence-Set Coverage Evaluation
↓
Answer Correctness
↓
Faithfulness
↓
Citation Entailment / Pruning
↓
API Load Test
```

如果未来优化 RRF Weight、TopK、Reranker Strategy、Evidence Selection 或 Prompt，应先在 Dev Set 上完成，再到新的 Held-out Test Set 验证。

---

# 32. Final Evaluation Summary

当前项目已经形成：

```text
28 Documents
↓
835 Chunks
↓
Manifest Coverage Validation
↓
Chunk Validation
↓
Qdrant
↓
ACL-aware Dense + BM25
↓
RRF
↓
Reranker
↓
Coarse Evidence Gate
↓
Evidence-Constrained Generation
↓
Deterministic Citation
↓
Frozen V1 / V2 / V3 Evaluation
↓
Case-level Checkpoint
↓
Immutable Final Snapshot
↓
Failure Analysis
```

最终 V3 结果：

```text
Retrieval:
Dense Recall@10            = 1.0000
Hybrid+Rerank Recall@10    = 1.0000
Hybrid+Rerank MRR@10       = 0.9259

Full-RAG:
Decision Accuracy          = 0.9783
Answerable Accuracy        = 0.9722
Refusal Accuracy           = 1.0000
Hard Negative Refusal      = 1.0000
OOD Refusal                = 1.0000

Citation:
All-case Precision         = 0.8629
All-case Recall            = 0.9690
All-case Hit Rate          = 1.0000

Strict Precision           = 0.8944
Strict Recall              = 0.9917
Strict Hit Rate            = 1.0000
```

最终目标不是追求某个单一 100% 指标，而是建立一套：

```text
可复现
可解释
可定位 Failure
可比较改动
可验证 Regression
能够安全拒答
```

的 RAG Evaluation Pipeline。
