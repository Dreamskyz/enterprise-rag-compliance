"""企业 AI 合规助手的核心 Query Application Service。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.evidence.models import (
    EvidenceDecisionReason,
)
from enterprise_rag.generation.answerer import (
    EvidenceGroundedAnswerer,
)
from enterprise_rag.retrieval.reranked import (
    RerankedRetriever,
)
from enterprise_rag.service.models import (
    QueryResult,
)


class QueryService:
    """
    企业 AI 合规助手核心查询服务。

    当前完整在线业务链：

        Query
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
         /                  \\
      Reject                Pass
                              ↓
                  Evidence-Constrained
                       Generation
                         /      \\
                      Answer   Refuse
                         ↓
                     Citation

    QueryService 是 Application Service。

    FastAPI / Streamlit 后续只调用这一层，
    不直接编排 Retriever、Gate、LLM。
    """

    def __init__(
        self,
        retriever: RerankedRetriever,
        evidence_gate: EvidenceGate,
        answerer: EvidenceGroundedAnswerer,
        retrieval_top_k: int = 5,
    ) -> None:
        """
        初始化 QueryService。

        参数：
            retriever:
                ACL-aware Hybrid + Reranker。

            evidence_gate:
                粗粒度 Relevance Gate。

                threshold 必须由外部显式配置，
                QueryService 不写死阈值。

            answerer:
                Evidence-Constrained
                Generation Service。

            retrieval_top_k:
                最终送入 Gate / Generation
                的 Reranked Candidate 数量。
        """

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k 必须大于 0"
            )

        self.retriever = retriever

        self.evidence_gate = (
            evidence_gate
        )

        self.answerer = answerer

        self.retrieval_top_k = (
            retrieval_top_k
        )

    def ask(
        self,
        query: str,
        access_context: AccessContext | None = None,
    ) -> QueryResult:
        """
        执行一次完整 RAG Query。

        如果没有显式 AccessContext，
        默认 guest。

        这是 Least Privilege。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        # --------------------------------------------------
        # 1. 默认最小权限。
        # --------------------------------------------------

        if access_context is None:
            access_context = AccessContext(
                role=UserRole.GUEST
            )

        # --------------------------------------------------
        # 2. ACL-aware Retrieval。
        #
        # 权限控制在 Dense / BM25
        # Candidate Generation 前已经生效。
        # --------------------------------------------------

        retrieval_results = (
            self.retriever.search(
                query=query,
                top_k=self.retrieval_top_k,
                access_context=access_context,
            )
        )

        retrieval_count = len(
            retrieval_results
        )

        top_rerank_score = (
            float(
                retrieval_results[
                    0
                ].rerank_score
            )
            if retrieval_results
            else None
        )

        # --------------------------------------------------
        # 3. Coarse Relevance Gate。
        #
        # 注意：
        # 它不负责最终 answerability。
        # --------------------------------------------------

        gate_decision = (
            self.evidence_gate.evaluate(
                retrieval_results
            )
        )

        # --------------------------------------------------
        # Case A：
        # 完全没有 Retrieval Evidence。
        #
        # 不调用 LLM。
        # --------------------------------------------------

        if (
            gate_decision.reason
            == EvidenceDecisionReason.NO_RESULTS
        ):
            return QueryResult(
                query=query.strip(),
                role=access_context.role,
                answerable=False,
                answer=None,
                reason=(
                    "当前知识库未检索到"
                    "可用于回答该问题的证据。"
                ),
                citations=(),
                retrieval_count=(
                    retrieval_count
                ),
                top_rerank_score=(
                    top_rerank_score
                ),
                gate_reason=(
                    gate_decision.reason.value
                ),
            )

        # --------------------------------------------------
        # Case B：
        # 有 Retrieval Result，
        # 但相关性低于粗粒度 Gate。
        #
        # 同样不调用 LLM。
        # --------------------------------------------------

        if not gate_decision.passed:
            return QueryResult(
                query=query.strip(),
                role=access_context.role,
                answerable=False,
                answer=None,
                reason=(
                    "当前检索结果与问题的"
                    "相关性不足，"
                    "无法依据知识库可靠回答。"
                ),
                citations=(),
                retrieval_count=(
                    retrieval_count
                ),
                top_rerank_score=(
                    top_rerank_score
                ),
                gate_reason=(
                    gate_decision.reason.value
                ),
            )

        # --------------------------------------------------
        # Case C：
        # 通过 Coarse Relevance Gate。
        #
        # 这里仍然不能认为：
        #
        #     answerable = True
        #
        # 必须交给 Evidence-Constrained
        # Generation 做第二阶段
        # Evidence Sufficiency 判断。
        # --------------------------------------------------

        grounded_answer = (
            self.answerer.answer(
                query=query,
                results=retrieval_results,
            )
        )

        return QueryResult(
            query=query.strip(),
            role=access_context.role,
            answerable=(
                grounded_answer.answerable
            ),
            answer=(
                grounded_answer.answer
            ),
            reason=(
                grounded_answer.reason
            ),
            citations=(
                grounded_answer.citations
            ),
            retrieval_count=(
                retrieval_count
            ),
            top_rerank_score=(
                top_rerank_score
            ),
            gate_reason=(
                gate_decision.reason.value
            ),
        )