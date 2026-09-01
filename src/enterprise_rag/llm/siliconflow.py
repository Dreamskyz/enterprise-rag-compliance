"""SiliconFlow OpenAI-compatible LLM Client。"""

import os
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from enterprise_rag.llm.models import (
    LLMResponse,
)


SILICONFLOW_BASE_URL = (
    "https://api.siliconflow.cn/v1"
)

DEFAULT_MODEL = (
    "deepseek-ai/DeepSeek-V4-Flash"
)


class SiliconFlowLLMService:
    """
    SiliconFlow LLM 服务封装。

    当前职责：

    1. 从环境变量读取 API Key；
    2. 创建 OpenAI-compatible Client；
    3. 调用 Chat Completions；
    4. 将第三方 SDK Response
       转换成项目自己的 LLMResponse。

    上层业务不直接依赖 OpenAI SDK。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = SILICONFLOW_BASE_URL,
    ) -> None:
        """
        初始化 SiliconFlow LLM Service。

        参数：
            api_key:
                可显式传入 API Key。

                正式运行时建议不传，
                自动从：
                    SILICONFLOW_API_KEY
                环境变量读取。

            model:
                模型名称。

                如果未显式传入，
                优先读取：
                    LLM_MODEL

                如果环境变量也没有，
                使用：
                    deepseek-ai/DeepSeek-V4-Flash

            base_url:
                SiliconFlow OpenAI-compatible
                API Base URL。
        """

        # --------------------------------------------------
        # 加载项目根目录附近的 .env。
        #
        # 如果真实环境已经设置环境变量，
        # dotenv 默认不会覆盖已有值。
        # --------------------------------------------------

        load_dotenv()

        resolved_api_key = (
            api_key
            or os.getenv(
                "SILICONFLOW_API_KEY"
            )
        )

        if not resolved_api_key:
            raise ValueError(
                "缺少 SILICONFLOW_API_KEY。"
                "请在 .env 或系统环境变量中配置。"
            )

        resolved_model = (
            model
            or os.getenv(
                "LLM_MODEL"
            )
            or DEFAULT_MODEL
        )

        if not resolved_model.strip():
            raise ValueError(
                "LLM model 不能为空"
            )

        if not base_url.strip():
            raise ValueError(
                "base_url 不能为空"
            )

        self.model = resolved_model

        self.base_url = base_url

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
        )

    def generate(
        self,
        messages: Sequence[
            dict[str, str]
        ],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        执行一次非流式 Chat Completion。

        当前 V1 先使用非流式调用，
        方便：

        - 单元测试；
        - QueryService 封装；
        - Token Usage 统计；
        - Evaluation。

        Streamlit Streaming 后面再加。

        参数：
            messages:
                OpenAI Chat 格式：

                [
                    {
                        "role": "system",
                        "content": "...",
                    },
                    {
                        "role": "user",
                        "content": "...",
                    },
                ]

            temperature:
                生成随机性。

                RAG / 合规回答更强调稳定，
                所以当前调用默认较低。

            max_tokens:
                最多生成 Token 数。
        """

        if not messages:
            raise ValueError(
                "messages 不能为空"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens 必须大于 0"
            )

        if not (
            0.0
            <= temperature
            <= 2.0
        ):
            raise ValueError(
                "temperature 必须位于 "
                "[0.0, 2.0] 范围"
            )

        # --------------------------------------------------
        # 基础输入检查。
        #
        # 避免把明显坏数据发送到远程 API。
        # --------------------------------------------------

        normalized_messages: list[
            dict[str, str]
        ] = []

        for message in messages:
            role = message.get(
                "role",
                ""
            ).strip()

            content = message.get(
                "content",
                ""
            ).strip()

            if not role:
                raise ValueError(
                    "message.role 不能为空"
                )

            if not content:
                raise ValueError(
                    "message.content 不能为空"
                )

            normalized_messages.append({
                "role": role,
                "content": content,
            })

        # --------------------------------------------------
        # SiliconFlow 支持 OpenAI-compatible
        # Chat Completions。
        # --------------------------------------------------

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        )

        if not response.choices:
            raise RuntimeError(
                "LLM Response 未包含 choices"
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            raise RuntimeError(
                "LLM Response content 为空"
            )

        content = content.strip()

        if not content:
            raise RuntimeError(
                "LLM Response content "
                "为空字符串"
            )

        # --------------------------------------------------
        # Usage 在某些兼容服务/异常情况下
        # 理论上可能不存在，
        # 因此安全读取。
        # --------------------------------------------------

        usage = response.usage

        if usage is None:
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

        else:
            prompt_tokens = int(
                usage.prompt_tokens
            )

            completion_tokens = int(
                usage.completion_tokens
            )

            total_tokens = int(
                usage.total_tokens
            )

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=(
                completion_tokens
            ),
            total_tokens=total_tokens,
        )