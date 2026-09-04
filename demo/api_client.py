"""Streamlit Demo 使用的 FastAPI HTTP Client。

该模块只负责：

1. 调用 FastAPI；
2. 检查 HTTP 状态；
3. 检查基础 Response Shape；
4. 将 JSON 返回给 UI。

它不会：

- 初始化 RAG Runtime；
- 直接调用 QueryService；
- 加载 BGE-M3；
- 加载 Reranker；
- 访问 Qdrant。

因此 Demo 与真正 Backend 之间保持：

    Streamlit
        ↓ HTTP
    FastAPI
        ↓
    RAG Runtime

的清晰边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


# ==========================================================
# Demo 默认连接本机 FastAPI。
#
# 这里不是密钥，
# 因此可以安全保存在源码中。
#
# 后面 Streamlit UI 也可以允许通过环境变量覆盖。
# ==========================================================

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class ApiClientError(RuntimeError):
    """Demo Client 调用 FastAPI 失败。"""


@dataclass(frozen=True)
class EnterpriseRagApiClient:
    """Enterprise RAG FastAPI Client。

    Parameters
    ----------
    base_url:
        FastAPI 服务地址。

    timeout_seconds:
        HTTP 请求超时时间。

        /ask 会真实经过：

            Retrieval
            Rerank
            SiliconFlow LLM

        因此不能使用普通 5 秒 HTTP timeout。
    """

    base_url: str = DEFAULT_API_BASE_URL

    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        """规范化并验证 Client 配置。"""

        normalized_url = (
            self.base_url
            .strip()
            .rstrip("/")
        )

        if not normalized_url:
            raise ValueError(
                "base_url 不能为空"
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds 必须大于 0"
            )

        # dataclass(frozen=True) 中不能直接：
        #
        #     self.base_url = ...
        #
        # 因此使用 object.__setattr__。
        object.__setattr__(
            self,
            "base_url",
            normalized_url,
        )

    def health(self) -> dict[str, Any]:
        """调用 GET /health。"""

        return self._get_json(
            path="/health",
        )

    def ready(self) -> dict[str, Any]:
        """调用 GET /ready。

        该接口适合 Demo 启动时检查：

            RAG Runtime
            Qdrant
            Heavy Models

        是否已经准备完成。
        """

        return self._get_json(
            path="/ready",
        )

    def ask(
        self,
        *,
        query: str,
        role: str,
    ) -> dict[str, Any]:
        """调用 POST /api/v1/ask。

        Parameters
        ----------
        query:
            用户问题。

        role:
            Demo 角色：

                guest
                developer
                admin

        Returns
        -------
        dict
            FastAPI AskResponse 对应 JSON。
        """

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "query 不能为空"
            )

        self._validate_role(
            role
        )

        payload = {
            "query": clean_query,
            "role": role,
        }

        data = self._post_json(
            path="/api/v1/ask",
            payload=payload,
        )

        self._validate_ask_response(
            data
        )

        return data

    def retrieve(
        self,
        *,
        query: str,
        role: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """调用 POST /api/v1/retrieve。

        Demo 使用该接口显示：

            Final Retrieval Rank
            Dense Rank
            BM25 Rank
            RRF Score
            Rerank Score
            Evidence Content
        """

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "query 不能为空"
            )

        self._validate_role(
            role
        )

        if not 1 <= top_k <= 20:
            raise ValueError(
                "top_k 必须位于 1~20"
            )

        payload = {
            "query": clean_query,
            "role": role,
            "top_k": top_k,
        }

        data = self._post_json(
            path="/api/v1/retrieve",
            payload=payload,
        )

        self._validate_retrieve_response(
            data
        )

        return data

    def _get_json(
        self,
        *,
        path: str,
    ) -> dict[str, Any]:
        """发送 GET 请求并返回 JSON。"""

        url = (
            f"{self.base_url}{path}"
        )

        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
            )

        except requests.RequestException as exc:
            raise ApiClientError(
                "无法连接 FastAPI："
                f"{url}。"
                "请确认后端已经启动。"
            ) from exc

        return self._parse_response(
            response=response,
            url=url,
        )

    def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """发送 JSON POST 请求。"""

        url = (
            f"{self.base_url}{path}"
        )

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )

        except requests.Timeout as exc:
            raise ApiClientError(
                "FastAPI 请求超时："
                f"{url}。"
                "Full-RAG /ask 包含 LLM 调用，"
                "如果模型服务较慢可提高 timeout。"
            ) from exc

        except requests.RequestException as exc:
            raise ApiClientError(
                "无法连接 FastAPI："
                f"{url}。"
                "请确认后端已经启动。"
            ) from exc

        return self._parse_response(
            response=response,
            url=url,
        )

    @staticmethod
    def _parse_response(
        *,
        response: requests.Response,
        url: str,
    ) -> dict[str, Any]:
        """统一处理 HTTP Response。

        这里不会静默吞掉：

            4xx
            5xx
            非 JSON Response

        因为 Demo 应该明确告诉使用者：
        Backend 到底哪里失败。
        """

        if not response.ok:
            body = (
                response.text.strip()
            )

            if len(body) > 1000:
                body = (
                    body[:1000]
                    + "..."
                )

            raise ApiClientError(
                "FastAPI 返回错误："
                f"HTTP {response.status_code}\n"
                f"URL: {url}\n"
                f"Response: {body}"
            )

        try:
            data = response.json()

        except ValueError as exc:
            raise ApiClientError(
                "FastAPI 返回了非 JSON Response："
                f"{url}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ApiClientError(
                "FastAPI Response 顶层必须是 JSON Object："
                f"{url}"
            )

        return data

    @staticmethod
    def _validate_role(
        role: str,
    ) -> None:
        """在 Client 侧提前校验 Demo Role。

        FastAPI / Pydantic 本身也会检查，
        这里主要是为了给 Streamlit
        更直接的错误信息。
        """

        allowed_roles = {
            "guest",
            "developer",
            "admin",
        }

        if role not in allowed_roles:
            raise ValueError(
                "role 必须是："
                "guest / developer / admin"
            )

    @staticmethod
    def _validate_ask_response(
        data: dict[str, Any],
    ) -> None:
        """检查 AskResponse 的关键字段。

        注意：

        这里不是复制一套完整 Pydantic Schema。

        FastAPI Backend 已经负责真正的
        Response Model Validation。

        Demo Client 这里只做最基础 Contract Guard，
        防止 UI 因字段完全缺失而产生难理解错误。
        """

        required_fields = {
            "query",
            "role",
            "answerable",
            "answer",
            "reason",
            "citations",
            "retrieval_count",
            "top_rerank_score",
            "gate_reason",
        }

        missing_fields = (
            required_fields
            - data.keys()
        )

        if missing_fields:
            raise ApiClientError(
                "AskResponse 缺少字段："
                f"{sorted(missing_fields)}"
            )

        if not isinstance(
            data["answerable"],
            bool,
        ):
            raise ApiClientError(
                "AskResponse.answerable "
                "必须是 bool"
            )

        if not isinstance(
            data["citations"],
            list,
        ):
            raise ApiClientError(
                "AskResponse.citations "
                "必须是 list"
            )

    @staticmethod
    def _validate_retrieve_response(
        data: dict[str, Any],
    ) -> None:
        """检查 RetrieveResponse 的关键字段。"""

        required_fields = {
            "query",
            "role",
            "result_count",
            "results",
        }

        missing_fields = (
            required_fields
            - data.keys()
        )

        if missing_fields:
            raise ApiClientError(
                "RetrieveResponse 缺少字段："
                f"{sorted(missing_fields)}"
            )

        if not isinstance(
            data["result_count"],
            int,
        ):
            raise ApiClientError(
                "RetrieveResponse.result_count "
                "必须是 int"
            )

        if not isinstance(
            data["results"],
            list,
        ):
            raise ApiClientError(
                "RetrieveResponse.results "
                "必须是 list"
            )