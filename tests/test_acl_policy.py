"""测试 ACL 角色与资源权限策略。"""

import pytest

from enterprise_rag.acl.models import (
    AccessLevel,
    UserRole,
)
from enterprise_rag.acl.policy import (
    can_access,
    get_allowed_access_levels,
)


def test_guest_can_only_access_public() -> None:
    """
    guest 只能访问 public。
    """

    assert can_access(
        UserRole.GUEST,
        AccessLevel.PUBLIC,
    )

    assert not can_access(
        UserRole.GUEST,
        AccessLevel.DEVELOPER,
    )

    assert not can_access(
        UserRole.GUEST,
        AccessLevel.ADMIN,
    )


def test_developer_can_access_public_and_developer() -> None:
    """
    developer 可以访问：
        public
        developer

    但不能访问：
        admin
    """

    assert can_access(
        UserRole.DEVELOPER,
        AccessLevel.PUBLIC,
    )

    assert can_access(
        UserRole.DEVELOPER,
        AccessLevel.DEVELOPER,
    )

    assert not can_access(
        UserRole.DEVELOPER,
        AccessLevel.ADMIN,
    )


def test_admin_can_access_all_levels() -> None:
    """
    admin 可以访问当前全部 Access Level。
    """

    assert can_access(
        UserRole.ADMIN,
        AccessLevel.PUBLIC,
    )

    assert can_access(
        UserRole.ADMIN,
        AccessLevel.DEVELOPER,
    )

    assert can_access(
        UserRole.ADMIN,
        AccessLevel.ADMIN,
    )


def test_access_level_string_is_supported() -> None:
    """
    当前 KnowledgeChunk / Qdrant Payload
    保存的是字符串，因此 policy 必须能直接处理字符串。
    """

    assert can_access(
        UserRole.DEVELOPER,
        "developer",
    )

    assert not can_access(
        UserRole.GUEST,
        "developer",
    )


def test_invalid_access_level_is_rejected() -> None:
    """
    未知 Access Level 必须 Fail Closed，
    不能默认按 public 处理。
    """

    with pytest.raises(
        ValueError,
        match="access_level",
    ):
        can_access(
            UserRole.ADMIN,
            "unknown-level",
        )


def test_allowed_levels_are_correct() -> None:
    """
    验证角色到 Access Level 的显式映射。
    """

    assert get_allowed_access_levels(
        UserRole.GUEST
    ) == frozenset(
        {
            AccessLevel.PUBLIC,
        }
    )

    assert get_allowed_access_levels(
        UserRole.DEVELOPER
    ) == frozenset(
        {
            AccessLevel.PUBLIC,
            AccessLevel.DEVELOPER,
        }
    )

    assert get_allowed_access_levels(
        UserRole.ADMIN
    ) == frozenset(
        {
            AccessLevel.PUBLIC,
            AccessLevel.DEVELOPER,
            AccessLevel.ADMIN,
        }
    )