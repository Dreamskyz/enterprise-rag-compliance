"""基于 Retrieval Evidence 的受约束回答生成器。"""

from collections.abc import Sequence

from enterprise_rag.generation.models import (
    GroundedAnswer,
)
from enterprise_rag.generation.parser import (
    parse_grounded_answer,
)
from enterprise_rag.generation.prompt_builder import (
    build_evidence_items,
    build_generation_messages,
)
from enterprise_rag.llm.siliconflow import (
    SiliconFlowLLMService,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
)


class EvidenceGroundedAnswerer:
    """
    Evidence-Constrained Generation Service。

    当前流程：

        Query
        +
        Reranked Evidence
               ↓
        Evidence ID 编号
               ↓
         Prompt Builder
               ↓
              LLM
               ↓
          JSON Response
               ↓
            Parser
               ↓
      Citation 白名单校验
               ↓
         GroundedAnswer
    """

    def __init__(
        self,
        llm_service: SiliconFlowLLMService,
        max_evidence: int = 5,
    ) -> None:
        if max_evidence <= 0:
            raise ValueError(
                "max_evidence 必须大于 0"
            )

        self.llm_service = (
            llm_service
        )

        self.max_evidence = (
            max_evidence
        )

    def answer(
        self,
        query: str,
        results: Sequence[
            RerankedSearchResult
        ],
    ) -> GroundedAnswer:
        """
        基于当前检索证据生成回答。

        注意：

        如果 results 完全为空，
        不应该浪费 LLM 调用。

        当前直接返回程序化拒答。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if not results:
            return GroundedAnswer(
                answerable=False,
                answer=None,
                reason=(
                    "当前检索未返回可用证据。"
                ),
                citations=(),
            )

        evidence_items = (
            build_evidence_items(
                results=results,
                max_evidence=(
                    self.max_evidence
                ),
            )
        )

        messages = (
            build_generation_messages(
                query=query,
                evidence_items=(
                    evidence_items
                ),
            )
        )

        response = (
            self.llm_service.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
        )

        return parse_grounded_answer(
            raw_content=(
                response.content
            ),
            evidence_items=(
                evidence_items
            ),
        )