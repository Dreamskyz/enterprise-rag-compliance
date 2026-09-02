"""Retrieval Latency Benchmark。"""

import random
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from statistics import (
    mean,
    median,
)
from time import perf_counter

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
)
from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
    RetrieverCallable,
)


# ==========================================================
# GPU Synchronization Callable。
#
# GPU Method：
#
#     Dense
#     Hybrid RRF
#     Hybrid + Rerank
#
# 可以传：
#
#     torch.cuda.synchronize
#
# BM25 为 CPU-only，因此传 None。
# ==========================================================

SynchronizeCallable = Callable[
    [],
    None,
]


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    """
    单种 Retrieval Method 的 Latency Benchmark 结果。
    """

    method: RetrievalMethod

    sample_count: int

    warmup_count: int

    rounds: int

    mean_ms: float

    p50_ms: float

    p95_ms: float

    min_ms: float

    max_ms: float

    samples_ms: tuple[
        float,
        ...,
    ]


@dataclass(frozen=True)
class InterleavedMethodConfig:
    """
    Interleaved Benchmark 中的一种 Method 配置。

    retrieve_fn:
        Evaluation Adapter。

    synchronize_fn:
        GPU Method 使用 torch.cuda.synchronize。

        CPU-only BM25 使用 None。
    """

    method: RetrievalMethod

    retrieve_fn: RetrieverCallable

    synchronize_fn: (
        SynchronizeCallable
        | None
    )


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float:
    """
    使用线性插值计算百分位。

    percentile_value：

        0 <= p <= 100
    """

    if not values:
        raise ValueError(
            "values 不能为空"
        )

    if not (
        0.0
        <= percentile_value
        <= 100.0
    ):
        raise ValueError(
            "percentile_value "
            "必须位于 0~100"
        )

    sorted_values = sorted(
        float(value)
        for value in values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        percentile_value
        / 100.0
        * (
            len(sorted_values)
            - 1
        )
    )

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = (
        position
        - lower_index
    )

    lower_value = (
        sorted_values[
            lower_index
        ]
    )

    upper_value = (
        sorted_values[
            upper_index
        ]
    )

    return (
        lower_value
        +
        (
            upper_value
            - lower_value
        )
        * fraction
    )


def _build_latency_result(
    *,
    method: RetrievalMethod,
    samples_ms: Sequence[float],
    warmup_count: int,
    rounds: int,
) -> LatencyBenchmarkResult:
    """
    将原始 Latency Samples 汇总为统计结果。
    """

    if not samples_ms:
        raise ValueError(
            "samples_ms 不能为空"
        )

    normalized_samples = [
        float(value)
        for value in samples_ms
    ]

    return LatencyBenchmarkResult(
        method=method,
        sample_count=len(
            normalized_samples
        ),
        warmup_count=warmup_count,
        rounds=rounds,
        mean_ms=mean(
            normalized_samples
        ),
        p50_ms=median(
            normalized_samples
        ),
        p95_ms=percentile(
            normalized_samples,
            95.0,
        ),
        min_ms=min(
            normalized_samples
        ),
        max_ms=max(
            normalized_samples
        ),
        samples_ms=tuple(
            normalized_samples
        ),
    )


def benchmark_retrieval_method(
    *,
    method: RetrievalMethod,
    cases: Sequence[
        RetrievalEvalCase
    ],
    retrieve_fn: RetrieverCallable,
    retrieval_top_k: int,
    rounds: int = 5,
    warmup_count: int = 2,
    access_context: AccessContext | None = None,
    synchronize_fn: SynchronizeCallable | None = None,
    random_seed: int = 42,
) -> LatencyBenchmarkResult:
    """
    原来的单 Method Benchmark。

    仍然保留，方便独立测量单个 Method。

    注意：

    正式方法间横向比较现在优先使用：

        benchmark_interleaved_methods()
    """

    if not cases:
        raise ValueError(
            "cases 不能为空"
        )

    if retrieval_top_k <= 0:
        raise ValueError(
            "retrieval_top_k 必须大于 0"
        )

    if rounds <= 0:
        raise ValueError(
            "rounds 必须大于 0"
        )

    if warmup_count < 0:
        raise ValueError(
            "warmup_count 不能小于 0"
        )

    if access_context is None:
        access_context = AccessContext(
            role=UserRole.GUEST
        )

    # ==================================================
    # Warmup。
    # ==================================================

    for index in range(
        warmup_count
    ):
        case = cases[
            index % len(cases)
        ]

        if synchronize_fn is not None:
            synchronize_fn()

        retrieve_fn(
            case.query,
            retrieval_top_k,
            access_context,
        )

        if synchronize_fn is not None:
            synchronize_fn()

    # ==================================================
    # Measurement。
    # ==================================================

    samples_ms: list[
        float
    ] = []

    for round_index in range(
        rounds
    ):
        round_cases = list(
            cases
        )

        random_generator = (
            random.Random(
                random_seed
                + round_index
            )
        )

        random_generator.shuffle(
            round_cases
        )

        for case in round_cases:
            if synchronize_fn is not None:
                synchronize_fn()

            started_at = (
                perf_counter()
            )

            retrieve_fn(
                case.query,
                retrieval_top_k,
                access_context,
            )

            if synchronize_fn is not None:
                synchronize_fn()

            latency_ms = (
                perf_counter()
                - started_at
            ) * 1000.0

            samples_ms.append(
                latency_ms
            )

    return _build_latency_result(
        method=method,
        samples_ms=samples_ms,
        warmup_count=warmup_count,
        rounds=rounds,
    )


def benchmark_interleaved_methods(
    *,
    cases: Sequence[
        RetrievalEvalCase
    ],
    method_configs: Sequence[
        InterleavedMethodConfig
    ],
    retrieval_top_k: int,
    rounds: int = 5,
    warmup_count: int = 2,
    access_context: AccessContext | None = None,
    random_seed: int = 42,
) -> list[
    LatencyBenchmarkResult
]:
    """
    对多种 Retrieval Method 执行交错式 Benchmark。

    与旧方案最大的区别：

    旧方案：

        Dense × N
        BM25 × N
        Hybrid × N
        Rerank × N

    当前方案：

        Query 1：
            随机 Method 顺序

        Query 2：
            再随机 Method 顺序

        ...

    这样可以降低：

        GPU clock state
        thermal state
        cache state
        long-running method order

    对方法横向比较造成的偏差。

    每一种 Method 最终仍然得到：

        len(cases) × rounds

    个正式 Sample。
    """

    if not cases:
        raise ValueError(
            "cases 不能为空"
        )

    if not method_configs:
        raise ValueError(
            "method_configs 不能为空"
        )

    if retrieval_top_k <= 0:
        raise ValueError(
            "retrieval_top_k 必须大于 0"
        )

    if rounds <= 0:
        raise ValueError(
            "rounds 必须大于 0"
        )

    if warmup_count < 0:
        raise ValueError(
            "warmup_count 不能小于 0"
        )

    methods = [
        config.method
        for config in method_configs
    ]

    if len(set(methods)) != len(
        methods
    ):
        raise ValueError(
            "method_configs 中 "
            "RetrievalMethod 不能重复"
        )

    if access_context is None:
        access_context = AccessContext(
            role=UserRole.GUEST
        )

    config_by_method = {
        config.method: config
        for config in method_configs
    }

    # ==================================================
    # 1. Interleaved Warmup。
    #
    # 每种 Method 都执行相同数量 warmup。
    #
    # warmup 顺序同样打乱，
    # 防止固定某一 Method 永远最先执行。
    # ==================================================

    for warmup_index in range(
        warmup_count
    ):
        case = cases[
            warmup_index
            % len(cases)
        ]

        warmup_methods = list(
            methods
        )

        warmup_random = (
            random.Random(
                random_seed
                + 10_000
                + warmup_index
            )
        )

        warmup_random.shuffle(
            warmup_methods
        )

        for method in warmup_methods:
            config = (
                config_by_method[
                    method
                ]
            )

            if (
                config.synchronize_fn
                is not None
            ):
                config.synchronize_fn()

            config.retrieve_fn(
                case.query,
                retrieval_top_k,
                access_context,
            )

            if (
                config.synchronize_fn
                is not None
            ):
                config.synchronize_fn()

    # ==================================================
    # 2. 为每种 Method 创建 Sample Container。
    # ==================================================

    samples_by_method: dict[
        RetrievalMethod,
        list[float],
    ] = {
        method: []
        for method in methods
    }

    # ==================================================
    # 3. 正式 Interleaved Measurement。
    # ==================================================

    for round_index in range(
        rounds
    ):
        # ----------------------------------------------
        # 每轮 Query 顺序也重新打乱。
        # ----------------------------------------------

        round_cases = list(
            cases
        )

        query_random = random.Random(
            random_seed
            + round_index
        )

        query_random.shuffle(
            round_cases
        )

        for case_index, case in enumerate(
            round_cases
        ):
            # ------------------------------------------
            # 对当前 Query，
            # 再随机 Method 执行顺序。
            #
            # 因此不会永远：
            #
            # Dense → BM25 → Hybrid → Rerank
            # ------------------------------------------

            method_order = list(
                methods
            )

            method_random = (
                random.Random(
                    random_seed
                    + (
                        round_index
                        * 1000
                    )
                    + case_index
                )
            )

            method_random.shuffle(
                method_order
            )

            for method in (
                method_order
            ):
                config = (
                    config_by_method[
                        method
                    ]
                )

                # --------------------------------------
                # GPU Method：
                # 计时前确保上一条 GPU 工作已完成。
                # --------------------------------------

                if (
                    config.synchronize_fn
                    is not None
                ):
                    config.synchronize_fn()

                started_at = (
                    perf_counter()
                )

                config.retrieve_fn(
                    case.query,
                    retrieval_top_k,
                    access_context,
                )

                # --------------------------------------
                # GPU Method：
                # 等待本次 GPU 工作真正结束，
                # 再结束计时。
                # --------------------------------------

                if (
                    config.synchronize_fn
                    is not None
                ):
                    config.synchronize_fn()

                latency_ms = (
                    perf_counter()
                    - started_at
                ) * 1000.0

                samples_by_method[
                    method
                ].append(
                    latency_ms
                )

    # ==================================================
    # 4. 汇总。
    # ==================================================

    results: list[
        LatencyBenchmarkResult
    ] = []

    for method in methods:
        results.append(
            _build_latency_result(
                method=method,
                samples_ms=(
                    samples_by_method[
                        method
                    ]
                ),
                warmup_count=(
                    warmup_count
                ),
                rounds=rounds,
            )
        )

    return results