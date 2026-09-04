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
    build_generation_retry_messages,
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
        Strict Parser
          /        \\
       PASS        FAIL
        ↓           ↓
    Grounded     一次受控
     Answer      Schema Retry
                    ↓
                 Strict Parser
                  /       \\
               PASS       FAIL
                ↓          ↓
             Grounded     抛出异常
              Answer

    ------------------------------------------------------
    重要原则
    ------------------------------------------------------

    Parser 始终保持严格。

    Retry 只针对：

        Structured Output Contract Failure

    不针对：

        answer / refusal 语义结果

    因此不会因为模型选择拒答
    就反复要求模型重新回答。
    """

    def __init__(
        self,
        llm_service: SiliconFlowLLMService,
        max_evidence: int = 5,
        max_parse_retries: int = 1,
    ) -> None:
        """
        初始化 EvidenceGroundedAnswerer。

        max_evidence:
            最多提供给 LLM 的 Evidence 数量。

        max_parse_retries:
            当 LLM 输出未通过严格 Parser 时，
            最多额外允许多少次结构化输出重试。

        当前生产默认：

            1

        也就是：

            首次 Generation
            +
            最多一次 Retry

        防止无限重试导致：

            API 成本失控
            延迟失控
            死循环
        """

        if max_evidence <= 0:
            raise ValueError(
                "max_evidence 必须大于 0"
            )

        if max_parse_retries < 0:
            raise ValueError(
                "max_parse_retries 不能小于 0"
            )

        self.llm_service = (
            llm_service
        )

        self.max_evidence = (
            max_evidence
        )

        self.max_parse_retries = (
            max_parse_retries
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

        如果 results 完全为空，
        不浪费 LLM 调用，
        直接返回程序化拒答。

        如果 LLM 输出违反 JSON Contract：

            严格 Parser 拒绝
            ↓
            最多进行一次
            Structured Output Retry

        Retry 不改变：

            Query
            Evidence
            Evidence Sufficiency Rules

        只要求模型重新输出
        满足结构协议的完整 JSON。
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

        # --------------------------------------------------
        # 1. 构造 Evidence。
        # --------------------------------------------------

        evidence_items = (
            build_evidence_items(
                results=results,
                max_evidence=(
                    self.max_evidence
                ),
            )
        )

        # --------------------------------------------------
        # 2. 构造第一次 Generation Messages。
        # --------------------------------------------------

        messages = (
            build_generation_messages(
                query=query,
                evidence_items=(
                    evidence_items
                ),
            )
        )

        # --------------------------------------------------
        # 3. 第一次 Generation。
        # --------------------------------------------------

        response = (
            self.llm_service.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
        )

        raw_content = (
            response.content
        )

        # --------------------------------------------------
        # 4. 首次严格 Parser。
        # --------------------------------------------------

        try:
            return parse_grounded_answer(
                raw_content=raw_content,
                evidence_items=(
                    evidence_items
                ),
            )

        except ValueError as first_error:

            # ------------------------------------------------
            # 如果不允许 Retry，
            # 保留原始 strict failure。
            # ------------------------------------------------

            if self.max_parse_retries == 0:
                raise

            last_error: ValueError = (
                first_error
            )

        # --------------------------------------------------
        # 5. Structured Output Retry。
        #
        # 当前默认只执行一次，
        # 但这里保留整数配置形式，
        # 便于单元测试。
        # --------------------------------------------------

        current_invalid_content = (
            raw_content
        )

        for _ in range(
            self.max_parse_retries
        ):
            retry_messages = (
                build_generation_retry_messages(
                    original_messages=(
                        messages
                    ),
                    invalid_response=(
                        current_invalid_content
                    ),
                    validation_error=(
                        str(last_error)
                    ),
                )
            )

            retry_response = (
                self.llm_service.generate(
                    messages=retry_messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
            )

            current_invalid_content = (
                retry_response.content
            )

            try:
                return parse_grounded_answer(
                    raw_content=(
                        current_invalid_content
                    ),
                    evidence_items=(
                        evidence_items
                    ),
                )

            except ValueError as retry_error:
                last_error = (
                    retry_error
                )

        # --------------------------------------------------
        # 6. 所有受控 Retry 均失败。
        #
        # 不伪造 GroundedAnswer，
        # 不替模型填 reason，
        # 不绕过 Citation Validation。
        #
        # 继续抛出最后一次严格校验错误。
        # --------------------------------------------------

        raise last_error