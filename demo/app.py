"""Enterprise RAG Compliance Streamlit Demo。

该页面负责：

1. 接收用户 Query；
2. 选择 Demo Role；
3. 通过 HTTP 调用 FastAPI；
4. 展示 Answer / Refusal；
5. 展示 Citation；
6. 展示最终 Reranked Evidence；
7. 展示 ACL / Gate / Retrieval Trace。

系统边界：

    Streamlit
        ↓ HTTP
    FastAPI
        ↓
    QueryService
        ↓
    RAG Runtime

Streamlit 不会：

- import QueryService；
- 初始化 BGE-M3；
- 初始化 Reranker；
- 直接访问 Qdrant；
- 读取 SILICONFLOW_API_KEY。

Demo 的职责只是：

    Client + Observability UI
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from api_client import (
    ApiClientError,
    EnterpriseRagApiClient,
)


# ==========================================================
# Page
# ==========================================================

st.set_page_config(
    page_title="Enterprise RAG Compliance",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Constants
# ==========================================================

DEFAULT_QUERY = (
    "生成式人工智能服务管理暂行办法规定，"
    "发现违法内容后必须在几小时内处理？"
)

DEFAULT_TOP_K = 5

API_BASE_URL = os.getenv(
    "ENTERPRISE_RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
)


# ==========================================================
# Final Portfolio Metrics
#
# 这里只展示已经冻结的项目结果，
# 不参与任何在线推理逻辑。
# ==========================================================

CORPUS_DOCUMENT_COUNT = 28

FINAL_V3_DECISION_ACCURACY = "97.83%"

FINAL_V3_REFUSAL_ACCURACY = "100%"


# ==========================================================
# Demo Cases
# ==========================================================

EXAMPLE_CASES = {
    "法规 · 无具体时限": {
        "query": (
            "生成式人工智能服务管理暂行办法规定，"
            "发现违法内容后必须在几小时内处理？"
        ),
        "role": "guest",
        "description": (
            "展示：Retrieval 相关，但 Evidence "
            "没有具体小时数，因此最终拒答。"
        ),
    },
    "FastAPI · Grounded Answer": {
        "query": (
            "FastAPI 挂载的子应用"
            "会自动执行主应用的 lifespan 事件吗？"
        ),
        "role": "developer",
        "description": (
            "展示：developer 技术文档正常回答与 Citation。"
        ),
    },
    "Qdrant · ACL Developer": {
        "query": (
            "Qdrant 中 Payload Filter "
            "是如何用于过滤查询结果的？"
        ),
        "role": "developer",
        "description": (
            "展示：developer 可以访问 Qdrant "
            "技术规范并正常回答。"
        ),
    },
    "Qdrant · Evidence Coverage": {
        "query": (
            "Qdrant Hybrid Queries 支持哪些融合方式，"
            "RRF 的作用是什么？"
        ),
        "role": "developer",
        "description": (
            "展示：Gate PASS，但最终 Evidence Set "
            "不足以完整回答复合问题。"
        ),
    },
}


# ==========================================================
# Session State
# ==========================================================

if "query_input" not in st.session_state:
    st.session_state.query_input = DEFAULT_QUERY

if "selected_role" not in st.session_state:
    st.session_state.selected_role = "guest"

if "ask_result" not in st.session_state:
    st.session_state.ask_result = None

if "retrieve_result" not in st.session_state:
    st.session_state.retrieve_result = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ==========================================================
# HTTP Client
# ==========================================================

client = EnterpriseRagApiClient(
    base_url=API_BASE_URL,
    timeout_seconds=120.0,
)


# ==========================================================
# Helpers
# ==========================================================


def format_optional_rank(
    value: Any,
) -> str:
    """格式化 Optional Rank。"""

    if value is None:
        return "-"

    return str(value)


def format_score(
    value: Any,
) -> str:
    """格式化 Retrieval / Rerank Score。

    Score 是原始相关性信号，
    不是概率，因此不显示百分比。
    """

    if value is None:
        return "-"

    try:
        return f"{float(value):.4f}"

    except (TypeError, ValueError):
        return str(value)


def clear_previous_result() -> None:
    """清空上一轮结果。"""

    st.session_state.ask_result = None
    st.session_state.retrieve_result = None
    st.session_state.last_error = None


def load_example(
    *,
    query: str,
    role: str,
) -> None:
    """加载一个预设 Demo Case。"""

    st.session_state.query_input = query
    st.session_state.selected_role = role

    clear_previous_result()


def run_query(
    *,
    query: str,
    role: str,
    top_k: int,
) -> None:
    """执行一次 Demo Query。

    Demo 为展示完整 Retrieval Evidence，
    分别调用：

        /api/v1/retrieve
        /api/v1/ask

    因此当前 Demo V1 会执行两次 Retrieval。

    这是有意接受的展示层取舍：

        不为了 Demo Trace
        修改正式 /ask API Contract。
    """

    clean_query = query.strip()

    if not clean_query:
        st.session_state.last_error = (
            "请输入问题后再执行。"
        )
        return

    clear_previous_result()

    try:
        retrieve_result = client.retrieve(
            query=clean_query,
            role=role,
            top_k=top_k,
        )

        st.session_state.retrieve_result = (
            retrieve_result
        )

        ask_result = client.ask(
            query=clean_query,
            role=role,
        )

        st.session_state.ask_result = (
            ask_result
        )

    except (
        ApiClientError,
        ValueError,
    ) as exc:
        st.session_state.last_error = str(exc)


def render_backend_status() -> None:
    """渲染 Backend Ready 状态。"""

    st.sidebar.subheader(
        "Backend Status"
    )

    try:
        health = client.health()
        ready = client.ready()

    except ApiClientError as exc:
        st.sidebar.error(
            "FastAPI Not Ready"
        )

        st.sidebar.caption(
            str(exc)
        )

        return

    health_status = health.get(
        "status",
        "unknown",
    )

    ready_status = ready.get(
        "status",
        "unknown",
    )

    if (
        health_status == "ok"
        and ready_status == "ready"
    ):
        st.sidebar.success(
            "FastAPI Ready"
        )

    else:
        st.sidebar.warning(
            "Backend Status Unknown"
        )

    st.sidebar.caption(
        f"API: {API_BASE_URL}"
    )

    chunk_count = ready.get(
        "chunk_count"
    )

    if chunk_count is not None:
        st.sidebar.metric(
            "KnowledgeChunks",
            chunk_count,
        )


def render_citations(
    citations: list[dict[str, Any]],
) -> None:
    """渲染最终确定性 Citation。"""

    st.subheader(
        "Citations"
    )

    if not citations:
        st.info(
            "当前结果没有 Citation。"
        )
        return

    for index, citation in enumerate(
        citations,
        start=1,
    ):
        title = citation.get(
            "title",
            "Unknown Source",
        )

        article_number = citation.get(
            "article_number"
        )

        chunk_id = citation.get(
            "chunk_id",
            "",
        )

        evidence_id = citation.get(
            "evidence_id",
            "",
        )

        source_url = citation.get(
            "source_url",
            "",
        )

        if article_number:
            header = (
                f"{index}. {title} · "
                f"{article_number}"
            )

        else:
            header = (
                f"{index}. {title}"
            )

        with st.expander(
            header,
            expanded=True,
        ):
            st.markdown(
                f"**Evidence ID:** `{evidence_id}`"
            )

            st.markdown(
                f"**Chunk ID:** `{chunk_id}`"
            )

            if article_number:
                st.markdown(
                    "**Article:** "
                    f"{article_number}"
                )

            if source_url:
                st.markdown(
                    f"**Source:** [{source_url}]"
                    f"({source_url})"
                )


def render_retrieval_results(
    retrieve_result: dict[str, Any],
) -> None:
    """渲染最终 Reranked Evidence。"""

    st.subheader(
        "Retrieved Evidence"
    )

    st.caption(
        "这里展示 Retrieval / Fusion / Rerank "
        "之后的最终候选。Citation 不等于 Top-K 的简单复制。"
    )

    results = retrieve_result.get(
        "results",
        [],
    )

    if not results:
        st.info(
            "没有 Retrieval Result。"
        )
        return

    for item in results:
        rank = item.get(
            "rank",
            "-",
        )

        title = item.get(
            "title",
            "Unknown",
        )

        article_number = item.get(
            "article_number"
        )

        rerank_score = format_score(
            item.get(
                "rerank_score"
            )
        )

        if article_number:
            expander_title = (
                f"Rank {rank} · "
                f"{title} · "
                f"{article_number} · "
                f"Rerank {rerank_score}"
            )

        else:
            expander_title = (
                f"Rank {rank} · "
                f"{title} · "
                f"Rerank {rerank_score}"
            )

        with st.expander(
            expander_title,
            expanded=(rank == 1),
        ):
            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:
                st.metric(
                    "Rerank Score",
                    rerank_score,
                )

            with col2:
                st.metric(
                    "RRF Score",
                    format_score(
                        item.get(
                            "rrf_score"
                        )
                    ),
                )

            with col3:
                st.metric(
                    "Dense Rank",
                    format_optional_rank(
                        item.get(
                            "dense_rank"
                        )
                    ),
                )

            with col4:
                st.metric(
                    "BM25 Rank",
                    format_optional_rank(
                        item.get(
                            "bm25_rank"
                        )
                    ),
                )

            st.markdown(
                "**Chunk ID:** "
                f"`{item.get('chunk_id', '')}`"
            )

            st.markdown(
                "**Access Level:** "
                f"`{item.get('access_level', '')}`"
            )

            if article_number:
                st.markdown(
                    "**Article:** "
                    f"{article_number}"
                )

            source_url = item.get(
                "source_url",
                "",
            )

            if source_url:
                st.markdown(
                    f"**Source:** [{source_url}]"
                    f"({source_url})"
                )

            st.markdown(
                "**Content**"
            )

            st.write(
                item.get(
                    "content",
                    "",
                )
            )


# ==========================================================
# Sidebar
# ==========================================================

render_backend_status()

st.sidebar.divider()

st.sidebar.subheader(
    "Access Control"
)

st.sidebar.caption(
    "guest → public\n\n"
    "developer → public + developer\n\n"
    "admin → all"
)

role_options = [
    "guest",
    "developer",
    "admin",
]

current_role = (
    st.session_state.selected_role
)

if current_role not in role_options:
    current_role = "guest"

selected_role = st.sidebar.selectbox(
    "Demo Role",
    options=role_options,
    index=role_options.index(
        current_role
    ),
    help=(
        "Demo 中由用户手动选择。"
        "生产环境应由 JWT / SSO / IAM "
        "生成可信 AccessContext。"
    ),
)

st.session_state.selected_role = (
    selected_role
)

top_k = st.sidebar.slider(
    "Retrieved Evidence Top-K",
    min_value=1,
    max_value=10,
    value=DEFAULT_TOP_K,
    step=1,
)

st.sidebar.divider()

st.sidebar.subheader(
    "Example Cases"
)

st.sidebar.caption(
    "建议按顺序体验 Answer、Refusal、ACL "
    "和 Evidence Coverage。"
)

for label, case in EXAMPLE_CASES.items():
    if st.sidebar.button(
        label,
        use_container_width=True,
    ):
        load_example(
            query=case["query"],
            role=case["role"],
        )

        st.rerun()

    st.sidebar.caption(
        case["description"]
    )


# ==========================================================
# Header
# ==========================================================

st.title(
    "Enterprise RAG Compliance"
)

st.caption(
    "企业 AI 合规与应用规范助手 · "
    "ACL-aware Retrieval · "
    "Evidence-Constrained Generation"
)

st.info(
    "Demo 原则：有依据才回答，无依据必须拒答。"
)


# ==========================================================
# Portfolio Overview
# ==========================================================

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:
    st.metric(
        "Documents",
        CORPUS_DOCUMENT_COUNT,
    )

with overview_col2:
    st.metric(
        "KnowledgeChunks",
        835,
    )

with overview_col3:
    st.metric(
        "V3 Decision Accuracy",
        FINAL_V3_DECISION_ACCURACY,
    )

with overview_col4:
    st.metric(
        "Refusal Accuracy",
        FINAL_V3_REFUSAL_ACCURACY,
    )

st.caption(
    "V3 Benchmark: 46 cases · "
    "36 answerable · 10 unanswerable · "
    "Final run_003"
)


# ==========================================================
# Ask
# ==========================================================

st.subheader(
    "Ask"
)

query = st.text_area(
    "Question",
    key="query_input",
    height=110,
    placeholder=(
        "请输入合规、安全或内部技术规范问题..."
    ),
)

run_button = st.button(
    "Run RAG",
    type="primary",
    use_container_width=True,
)

if run_button:
    with st.spinner(
        "Retrieving evidence, reranking "
        "and evaluating evidence sufficiency..."
    ):
        run_query(
            query=query,
            role=selected_role,
            top_k=top_k,
        )


# ==========================================================
# Error
# ==========================================================

if st.session_state.last_error:
    st.error(
        st.session_state.last_error
    )


# ==========================================================
# Result
# ==========================================================

ask_result = (
    st.session_state.ask_result
)

retrieve_result = (
    st.session_state.retrieve_result
)

if ask_result is not None:
    st.divider()

    st.header(
        "RAG Result"
    )

    answerable = ask_result.get(
        "answerable",
        False,
    )

    gate_reason = ask_result.get(
        "gate_reason",
        "unknown",
    )

    top_rerank_score = format_score(
        ask_result.get(
            "top_rerank_score"
        )
    )

    retrieval_count = ask_result.get(
        "retrieval_count",
        0,
    )

    role = ask_result.get(
        "role",
        selected_role,
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.metric(
            "Decision",
            (
                "ANSWER"
                if answerable
                else "REFUSE"
            ),
        )

    with col2:
        st.metric(
            "Role",
            role,
        )

    with col3:
        st.metric(
            "Gate",
            gate_reason,
        )

    with col4:
        st.metric(
            "Top Rerank Score",
            top_rerank_score,
        )

    st.caption(
        f"Final retrieval count: "
        f"{retrieval_count}"
    )

    if answerable:
        st.success(
            "ANSWER · Evidence sufficient"
        )

        st.subheader(
            "Answer"
        )

        st.write(
            ask_result.get(
                "answer",
                "",
            )
        )

        with st.expander(
            "Why is the evidence sufficient?"
        ):
            st.write(
                ask_result.get(
                    "reason",
                    "",
                )
            )

    else:
        if gate_reason == "below_threshold":
            st.warning(
                "REFUSE · Coarse relevance gate rejected "
                "the available evidence"
            )

        else:
            st.warning(
                "REFUSE · Relevant evidence exists, "
                "but it is not sufficient to answer"
            )

        st.subheader(
            "Refusal Reason"
        )

        st.write(
            ask_result.get(
                "reason",
                "当前 Evidence 不足。",
            )
        )

    citations = ask_result.get(
        "citations",
        [],
    )

    render_citations(
        citations
    )


# ==========================================================
# Evidence
# ==========================================================

if retrieve_result is not None:
    st.divider()

    render_retrieval_results(
        retrieve_result
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Streamlit Demo Client → FastAPI → "
    "ACL-aware Retrieval → RRF → Rerank → "
    "Evidence Gate → Evidence-Constrained Generation"
)