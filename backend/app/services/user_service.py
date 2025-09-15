from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from .. import models
from ..schemas import user as user_schemas
from ..core.security import get_password_hash, verify_password


class UserService:
    """Service layer for user management."""

    TYPE_ALIASES = {
        "player": "player",
        "players": "player",
        "groupadmin": "group_admin",
        "groupadministrator": "group_admin",
        "admin": "group_admin",
        "systemadmin": "system_admin",
        "systemadministrator": "system_admin",
        "superadmin": "system_admin",
    }

    TYPE_ROLE_MAP = {
        "player": ["player"],
        "group_admin": ["group_admin", "admin"],
        "system_admin": ["system_admin"],
    }

    TYPE_ROLE_TOKENS = {"player", "groupadmin", "admin", "systemadmin", "superadmin"}

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_token(value: Optional[str]) -> str:
        if not value:
            return ""
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    def _normalize_type(self, user_type: Optional[str]) -> Optional[str]:
        if not user_type:
            return None
        token = self._normalize_token(user_type)
        if token not in self.TYPE_ALIASES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user type specified",
            )
        return self.TYPE_ALIASES[token]

    def _normalized_roles(self, roles: Optional[List[str]]) -> List[str]:
        return [
            token
            for role in roles or []
            if (token := self._normalize_token(role))
        ]

    def _strip_type_roles(self, roles: Optional[List[str]]) -> List[str]:
        cleaned: List[str] = []
        for role in roles or []:
            token = self._normalize_token(role)
            if token in self.TYPE_ROLE_TOKENS:
                continue
            cleaned.append(role)
        return cleaned

    @staticmethod
    def _dedupe_roles(roles: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for role in roles:
            if role is None:
                continue
            if not isinstance(role, str):
                role = str(role)
            if role not in seen:
                deduped.append(role)
                seen.add(role)
        return deduped

    def _resolve_user_type(
        self,
        user_type: Optional[str],
        roles: Optional[List[str]],
        is_superuser: Optional[bool],
        existing_roles: Optional[List[str]],
        default_is_superuser: bool,
    ) -> str:
        normalized_type = self._normalize_type(user_type)
        if normalized_type:
            return normalized_type

        normalized_roles = set(self._normalized_roles(roles if roles is not None else existing_roles))
        if is_superuser is True or default_is_superuser:
            return "system_admin"
        if "systemadmin" in normalized_roles or "superadmin" in normalized_roles:
            return "system_admin"
        if "groupadmin" in normalized_roles or "admin" in normalized_roles:
            return "group_admin"
        return "player"

    def _normalize_group_id(self, group_id: Optional[Any]) -> Optional[int]:
        if group_id is None:
            return None
        if isinstance(group_id, str):
            stripped = group_id.strip()
            if not stripped:
                return None
            if not stripped.isdigit():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid group id",
                )
            return int(stripped)
        if isinstance(group_id, int):
            return group_id
        try:
            return int(group_id)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid group id",
            )

    def _validate_group_assignment(
        self,
        group_id: Optional[Any],
        user_type: str,
    ) -> (Optional[models.Group], Optional[int]):
        normalized_group_id = self._normalize_group_id(group_id)
        group: Optional[models.Group] = None
        if normalized_group_id is not None:
            group = self.db.query(models.Group).filter(models.Group.id == normalized_group_id).first()
            if not group:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Group not found",
                )

        if user_type in {"player", "group_admin"} and group is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A group assignment is required for this user type",
            )

        return group, normalized_group_id

    def _prepare_roles_for_type(
        self,
        base_roles: Optional[List[str]],
        user_type: str,
    ) -> List[str]:
        roles = self._strip_type_roles(base_roles)
        roles.extend(self.TYPE_ROLE_MAP[user_type])
        return self._dedupe_roles(roles)

    def _is_group_admin_user(self, user: models.User) -> bool:
        if not user or not user.group_id:
            return False
        normalized_roles = set(self._normalized_roles(user.roles))
        return "groupadmin" in normalized_roles or "admin" in normalized_roles

    def _find_group_admins(
        self,
        group_id: Optional[int],
        exclude_user_id: Optional[int] = None,
    ) -> List[models.User]:
        if not group_id:
            return []
        query = self.db.query(models.User).filter(models.User.group_id == group_id)
        if exclude_user_id is not None:
            query = query.filter(models.User.id != exclude_user_id)
        users = query.all()
        return [user for user in users if self._is_group_admin_user(user)]

    def _find_all_group_admins(self, exclude_user_id: Optional[int] = None) -> List[models.User]:
        query = self.db.query(models.User)
        if exclude_user_id is not None:
            query = query.filter(models.User.id != exclude_user_id)
        users = query.all()
        return [user for user in users if self._is_group_admin_user(user)]

    def _cleanup_group_admin_on_delete(self, user: models.User) -> Dict[str, Any]:
        if not self._is_group_admin_user(user):
            return {
                "group_deleted": False,
                "group_id": user.group_id,
                "group_name": None,
            }

        group = self.db.query(models.Group).filter(models.Group.id == user.group_id).first()
        if not group:
            return {
                "group_deleted": False,
                "group_id": user.group_id,
                "group_name": None,
            }

        other_admins = self._find_group_admins(group.id, exclude_user_id=user.id)
        if not other_admins:
            group_name = group.name
            self.db.delete(group)
            return {
                "group_deleted": True,
                "group_id": group.id,
                "group_name": group_name,
            }

        if group.admin_id == user.id:
            group.admin_id = other_admins[0].id
            self.db.add(group)

        return {
            "group_deleted": False,
            "group_id": group.id,
            "group_name": group.name,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_user(self, user_id: int) -> models.User:
        user = self.db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        return user

    def get_user_by_email(self, email: str) -> Optional[models.User]:
        return self.db.query(models.User).filter(models.User.email == email).first()

    def get_user_by_username(self, username: str) -> Optional[models.User]:
        return self.db.query(models.User).filter(models.User.username == username).first()

    def get_users(self, skip: int = 0, limit: int = 100) -> List[models.User]:
        return self.db.query(models.User).offset(skip).limit(limit).all()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def create_user(self, user: user_schemas.UserCreate) -> models.User:
        existing = self.get_user_by_email(user.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        existing = self.get_user_by_username(user.username)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

        desired_type = self._resolve_user_type(
            user_type=user.user_type,
            roles=user.roles,
            is_superuser=user.is_superuser,
            existing_roles=user.roles,
            default_is_superuser=bool(user.is_superuser),
        )

        group, normalized_group_id = self._validate_group_assignment(user.group_id, desired_type)
        roles = self._prepare_roles_for_type(user.roles, desired_type)
        hashed_password = get_password_hash(user.password)

        db_user = models.User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            is_active=True,
            is_superuser=(desired_type == "system_admin"),
            group_id=normalized_group_id,
            roles=roles,
        )

        try:
            self.db.add(db_user)
            self.db.flush()

            if desired_type == "group_admin" and group and (group.admin_id is None):
                group.admin_id = db_user.id
                self.db.add(group)

            self.db.commit()
            self.db.refresh(db_user)
            return db_user
        except SQLAlchemyError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user",
            )

    def update_user(
        self,
        user_id: int,
        user_update: user_schemas.UserUpdate,
        current_user: models.User,
    ) -> models.User:
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        db_user = self.get_user(user_id)

        if user_update.email is not None:
            existing = self.get_user_by_email(user_update.email)
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )
            db_user.email = user_update.email

        if user_update.username is not None:
            existing = self.get_user_by_username(user_update.username)
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken",
                )
            db_user.username = user_update.username

        if user_update.full_name is not None:
            db_user.full_name = user_update.full_name

        if user_update.is_active is not None and current_user.is_superuser:
            db_user.is_active = user_update.is_active

        previous_group_id = db_user.group_id
        previous_type = self._resolve_user_type(
            user_type=None,
            roles=db_user.roles,
            is_superuser=db_user.is_superuser,
            existing_roles=db_user.roles,
            default_is_superuser=db_user.is_superuser,
        )

        proposed_group_id = (
            user_update.group_id
            if user_update.group_id is not None
            else db_user.group_id
        )

        desired_type = self._resolve_user_type(
            user_type=user_update.user_type,
            roles=user_update.roles if user_update.roles is not None else db_user.roles,
            is_superuser=user_update.is_superuser,
            existing_roles=db_user.roles,
            default_is_superuser=user_update.is_superuser if user_update.is_superuser is not None else db_user.is_superuser,
        )

        group, normalized_group_id = self._validate_group_assignment(proposed_group_id, desired_type)

        # Prevent removing the last group admin from a group via update
        if previous_type == "group_admin":
            changing_group = normalized_group_id != previous_group_id
            losing_admin_role = desired_type != "group_admin"
            if changing_group or losing_admin_role:
                other_admins = self._find_group_admins(previous_group_id, exclude_user_id=db_user.id)
                if not other_admins:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cannot remove the last group admin from the group. Assign another group admin or delete the group first.",
                    )
                previous_group = self.db.query(models.Group).filter(models.Group.id == previous_group_id).first()
                if previous_group and previous_group.admin_id == db_user.id:
                    previous_group.admin_id = other_admins[0].id
                    self.db.add(previous_group)

        roles_source = user_update.roles if user_update.roles is not None else db_user.roles
        roles = self._prepare_roles_for_type(roles_source, desired_type)

        db_user.group_id = normalized_group_id
        db_user.roles = roles
        db_user.is_superuser = desired_type == "system_admin"

        if user_update.password:
            db_user.hashed_password = get_password_hash(user_update.password)

        if desired_type == "group_admin" and group and (group.admin_id is None or group.admin_id == db_user.id):
            group.admin_id = db_user.id
            self.db.add(group)

        try:
            self.db.commit()
            self.db.refresh(db_user)
            return db_user
        except SQLAlchemyError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating user",
            )

    def delete_user(
        self,
        user_id: int,
        current_user: models.User,
        replacement_admin_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        db_user = self.get_user(user_id)
        user_type = self._resolve_user_type(
            user_type=None,
            roles=db_user.roles,
            is_superuser=db_user.is_superuser,
            existing_roles=db_user.roles,
            default_is_superuser=db_user.is_superuser,
        )

        promoted_user: Optional[models.User] = None

        try:
            if user_type == "system_admin":
                other_admins = self.db.query(models.User).filter(
                    models.User.id != db_user.id,
                    models.User.is_superuser == True,
                    models.User.is_active == True,
                ).count()

                if other_admins == 0:
                    if replacement_admin_id is None:
                        candidates = self._find_all_group_admins(exclude_user_id=db_user.id)
                        if not candidates:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail={
                                    "code": "no_group_admin_available",
                                    "message": "Cannot delete the last system administrator. Create another system administrator first.",
                                },
                            )
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail={
                                "code": "replacement_required",
                                "message": "Select a group administrator to promote before deleting the last system administrator.",
                                "candidates": [
                                    {
                                        "id": candidate.id,
                                        "username": candidate.username,
                                        "email": candidate.email,
                                        "group_id": candidate.group_id,
                                        "group_name": candidate.group.name if candidate.group else None,
                                    }
                                    for candidate in candidates
                                ],
                            },
                        )

                    replacement_user = self.get_user(replacement_admin_id)
                    if replacement_user.id == db_user.id:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Replacement user must be different from the user being deleted",
                        )
                    if not self._is_group_admin_user(replacement_user):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Replacement user must be a group admin",
                        )

                    replacement_roles = list(replacement_user.roles or [])
                    replacement_roles.append("system_admin")
                    replacement_user.roles = self._dedupe_roles(replacement_roles)
                    replacement_user.is_superuser = True
                    self.db.add(replacement_user)
                    promoted_user = replacement_user

            group_cleanup = self._cleanup_group_admin_on_delete(db_user)

            if not group_cleanup.get("group_deleted"):
                self.db.delete(db_user)
            self.db.commit()

            response: Dict[str, Any] = {"message": "User deleted successfully"}
            response.update(group_cleanup)
            if promoted_user:
                response["replacement_promoted"] = {
                    "id": promoted_user.id,
                    "username": promoted_user.username,
                    "email": promoted_user.email,
                }
            return response
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting user",
            )

    def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str,
        current_user: models.User,
    ) -> Dict[str, str]:
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        db_user = self.get_user(user_id)

        if not (current_user.is_superuser and user_id != current_user.id):
            if not verify_password(current_password, db_user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect current password",
                )

        db_user.hashed_password = get_password_hash(new_password)

        try:
            self.db.commit()
            return {"message": "Password updated successfully"}
        except SQLAlchemyError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating password",
            )
