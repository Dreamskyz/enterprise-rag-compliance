"""将 ACL 权限上下文转换为 Qdrant Payload Filter。"""

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
)

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.acl.policy import (
    get_allowed_access_levels,
)


def build_access_filter(
    role: UserRole,
) -> Filter:
    """
    根据用户角色构造 Qdrant Payload Filter。

    示例：

        guest
        → access_level IN ["public"]

        developer
        → access_level IN ["public", "developer"]

        admin
        → access_level IN [
            "public",
            "developer",
            "admin",
        ]

    该 Filter 会在 Qdrant 检索阶段生效，
    而不是检索结束后再做 Python 过滤。
    """

    allowed_levels = (
        get_allowed_access_levels(
            role
        )
    )

    allowed_values = [
        level.value
        for level in allowed_levels
    ]

    return Filter(
        must=[
            FieldCondition(
                key="access_level",
                match=MatchAny(
                    any=allowed_values
                ),
            )
        ]
    )