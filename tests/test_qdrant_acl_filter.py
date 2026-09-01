"""测试 ACL 到 Qdrant Filter 的转换。"""

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.acl.qdrant_filter import (
    build_access_filter,
)


def extract_allowed_values(
    role: UserRole,
) -> set[str]:
    """
    从构造出的 Qdrant Filter 中
    提取 access_level MatchAny 值。

    这里只用于测试。
    """

    query_filter = build_access_filter(
        role
    )

    assert query_filter.must

    condition = query_filter.must[0]

    assert condition.match is not None

    values = condition.match.any

    assert values is not None

    return set(values)


def test_guest_qdrant_filter() -> None:
    """guest 只能匹配 public。"""

    assert extract_allowed_values(
        UserRole.GUEST
    ) == {
        "public",
    }


def test_developer_qdrant_filter() -> None:
    """
    developer 可以匹配：
        public
        developer
    """

    assert extract_allowed_values(
        UserRole.DEVELOPER
    ) == {
        "public",
        "developer",
    }


def test_admin_qdrant_filter() -> None:
    """admin 可以匹配全部权限级别。"""

    assert extract_allowed_values(
        UserRole.ADMIN
    ) == {
        "public",
        "developer",
        "admin",
    }