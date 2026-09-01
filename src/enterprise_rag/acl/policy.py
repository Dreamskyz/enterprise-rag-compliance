"""ACL 权限策略。"""

from enterprise_rag.acl.models import (
    AccessLevel,
    UserRole,
)


# 每种角色允许访问哪些 Knowledge Access Level。
#
# 使用 frozenset：
# 1. 表示这是静态规则；
# 2. 防止运行过程中被意外修改。
ROLE_ALLOWED_LEVELS: dict[
    UserRole,
    frozenset[AccessLevel],
] = {
    UserRole.GUEST: frozenset(
        {
            AccessLevel.PUBLIC,
        }
    ),

    UserRole.DEVELOPER: frozenset(
        {
            AccessLevel.PUBLIC,
            AccessLevel.DEVELOPER,
        }
    ),

    UserRole.ADMIN: frozenset(
        {
            AccessLevel.PUBLIC,
            AccessLevel.DEVELOPER,
            AccessLevel.ADMIN,
        }
    ),
}


def get_allowed_access_levels(
    role: UserRole,
) -> frozenset[AccessLevel]:
    """
    返回指定角色允许访问的 Access Level。

    示例：

        guest
        → {public}

        developer
        → {public, developer}

        admin
        → {public, developer, admin}
    """

    try:
        return ROLE_ALLOWED_LEVELS[
            role
        ]
    except KeyError as exc:
        # 正常情况下 role 已经是 UserRole，
        # 不应进入这里。
        #
        # 这里仍然显式失败，
        # 避免未知角色被默认赋予权限。
        raise ValueError(
            f"未知用户角色：{role}"
        ) from exc


def can_access(
    role: UserRole,
    access_level: AccessLevel | str,
) -> bool:
    """
    判断某个用户角色是否可以访问指定资源。

    参数：
        role:
            当前用户角色。

        access_level:
            KnowledgeChunk 所要求的权限级别。

            可以传：
                AccessLevel.PUBLIC

            也可以传当前数据层保存的字符串：
                "public"

    返回：
        True:
            允许访问。

        False:
            无权访问。
    """

    try:
        normalized_level = AccessLevel(
            access_level
        )
    except ValueError as exc:
        # 非法 access_level 不应该被当成 public。
        #
        # 权限系统原则：
        # 对未知权限配置 Fail Closed，
        # 而不是 Fail Open。
        raise ValueError(
            "非法 access_level："
            f"{access_level}"
        ) from exc

    allowed_levels = (
        get_allowed_access_levels(
            role
        )
    )

    return (
        normalized_level
        in allowed_levels
    )