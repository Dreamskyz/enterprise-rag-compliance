"""ACL 权限相关数据模型。"""

from dataclasses import dataclass
from enum import StrEnum


class UserRole(StrEnum):
    """
    系统当前支持的用户角色。

    当前 V1 只实现最小 RBAC：
        guest
        developer
        admin

    后续 FastAPI 收到字符串角色后，
    可以通过 UserRole(value) 转换为受控枚举。
    """

    GUEST = "guest"
    DEVELOPER = "developer"
    ADMIN = "admin"


class AccessLevel(StrEnum):
    """
    KnowledgeChunk 的访问级别。

    Chunk 当前在 JSON / Qdrant Payload 中
    仍然保存字符串，例如：

        access_level = "public"

    ACL 层会将这些字符串转换为 AccessLevel，
    从而避免业务代码到处直接比较裸字符串。
    """

    PUBLIC = "public"
    DEVELOPER = "developer"
    ADMIN = "admin"


@dataclass(frozen=True)
class AccessContext:
    """
    单次请求的访问控制上下文。

    当前 V1 只保存 role。

    后续如果项目继续扩展，可以增加：
        user_id
        department
        tenant_id
        groups

    但当前不提前引入复杂企业 IAM 模型。
    """

    role: UserRole