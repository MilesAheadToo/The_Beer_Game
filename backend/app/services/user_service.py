from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from .. import models
from ..schemas import user as user_schemas
from ..core.security import get_password_hash, verify_password
from ..models.user import UserTypeEnum


class UserService:
    """Service layer for user management."""

    TYPE_ALIASES = {
        "player": UserTypeEnum.PLAYER,
        "players": UserTypeEnum.PLAYER,
        "groupadmin": UserTypeEnum.GROUP_ADMIN,
        "groupadministrator": UserTypeEnum.GROUP_ADMIN,
        "admin": UserTypeEnum.GROUP_ADMIN,
        "systemadmin": UserTypeEnum.SYSTEM_ADMIN,
        "systemadministrator": UserTypeEnum.SYSTEM_ADMIN,
        "superadmin": UserTypeEnum.SYSTEM_ADMIN,
    }

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_token(self, value: Optional[Any]) -> str:
        if not value:
            return ""
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    def _normalize_type(self, user_type: Optional[Any]) -> Optional[UserTypeEnum]:
        if not user_type:
            return None
        if isinstance(user_type, UserTypeEnum):
            return user_type
        token = self._normalize_token(user_type)
        if token not in self.TYPE_ALIASES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user type specified",
            )
        return self.TYPE_ALIASES[token]

    def _resolve_user_type(
        self,
        user_type: Optional[Any],
        fallback: Optional[UserTypeEnum] = None,
        assume_superuser: bool = False,
    ) -> UserTypeEnum:
        normalized_type = self._normalize_type(user_type)
        if normalized_type:
            return normalized_type

        if assume_superuser:
            return UserTypeEnum.SYSTEM_ADMIN

        if fallback is not None:
            return fallback

        return UserTypeEnum.PLAYER

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
        user_type: UserTypeEnum,
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

        if user_type in {UserTypeEnum.PLAYER, UserTypeEnum.GROUP_ADMIN} and group is None:
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

    def _is_group_admin_user(self, user: Optional[models.User]) -> bool:
        if not user or not user.group_id:
            return False
        return self._get_user_type(user) == UserTypeEnum.GROUP_ADMIN

    def _get_user_type(self, user: models.User) -> UserTypeEnum:
        fallback = UserTypeEnum.SYSTEM_ADMIN if user.is_superuser else user.user_type
        return self._resolve_user_type(
            user_type=user.user_type,
            fallback=fallback,
            assume_superuser=user.is_superuser,
        )

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

    def list_group_players(
        self,
        group_id: Optional[int],
        skip: int = 0,
        limit: Optional[int] = 100,
    ) -> List[models.User]:
        if not group_id:
            return []

        query = (
            self.db.query(models.User)
            .filter(models.User.group_id == group_id)
            .order_by(models.User.username.asc())
        )
        users = query.all()
        players = [user for user in users if self._get_user_type(user) == UserTypeEnum.PLAYER]

        if skip or (limit is not None and limit >= 0):
            end = skip + limit if limit is not None else None
            return players[skip:end]
        return players

    def list_accessible_users(
        self,
        current_user: models.User,
        skip: int = 0,
        limit: int = 100,
        user_type: Optional[str] = None,
    ) -> List[models.User]:
        normalized_type = self._normalize_type(user_type) if user_type else None

        acting_type = self._get_user_type(current_user)

        if acting_type == UserTypeEnum.SYSTEM_ADMIN:
            target_type = normalized_type or UserTypeEnum.GROUP_ADMIN
            users = (
                self.db.query(models.User)
                .order_by(models.User.username.asc())
                .all()
            )
            filtered = [user for user in users if self._get_user_type(user) == target_type]

            start = max(skip, 0)
            end = start + limit if limit is not None else None
            return filtered[start:end]

        if acting_type == UserTypeEnum.GROUP_ADMIN:
            if normalized_type and normalized_type != UserTypeEnum.PLAYER:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins can only view players",
                )
            return self.list_group_players(current_user.group_id, skip=skip, limit=limit)

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    def is_group_admin(self, user: models.User) -> bool:
        return self._is_group_admin_user(user)

    def get_user_type(self, user: models.User) -> UserTypeEnum:
        return self._get_user_type(user)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def create_user(
        self,
        user: user_schemas.UserCreate,
        current_user: models.User,
    ) -> models.User:
        acting_type = self._get_user_type(current_user) if current_user else None
        acting_is_superuser = acting_type == UserTypeEnum.SYSTEM_ADMIN
        acting_is_group_admin = acting_type == UserTypeEnum.GROUP_ADMIN

        if not acting_is_superuser and not acting_is_group_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

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

        hashed_password = get_password_hash(user.password)

        if acting_is_superuser:
            desired_type = self._resolve_user_type(
                user_type=user.user_type,
                fallback=UserTypeEnum.GROUP_ADMIN,
                assume_superuser=bool(user.is_superuser),
            )

            if desired_type != UserTypeEnum.GROUP_ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System administrators can only create group admin users",
                )

            group, normalized_group_id = self._validate_group_assignment(user.group_id, desired_type)
            is_superuser_flag = False
        else:
            if not current_user or not current_user.group_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Group admins must belong to a group",
                )

            if user.group_id is not None and user.group_id != current_user.group_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins can only assign users to their own group",
                )

            if user.user_type and self._normalize_type(user.user_type) != UserTypeEnum.PLAYER:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins can only create players",
                )

            if user.is_superuser:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins cannot grant system permissions",
                )

            desired_type = UserTypeEnum.PLAYER
            group, normalized_group_id = self._validate_group_assignment(current_user.group_id, desired_type)
            is_superuser_flag = False

        db_user = models.User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            is_active=True,
            is_superuser=is_superuser_flag,
            group_id=normalized_group_id,
            user_type=desired_type,
        )

        try:
            self.db.add(db_user)
            self.db.flush()

            if desired_type == UserTypeEnum.GROUP_ADMIN and group and (group.admin_id is None):
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
        db_user = self.get_user(user_id)
        acting_type = self._get_user_type(current_user)
        acting_is_superuser = acting_type == UserTypeEnum.SYSTEM_ADMIN
        acting_is_group_admin = acting_type == UserTypeEnum.GROUP_ADMIN
        target_type = self._get_user_type(db_user)

        is_group_admin_managing_player = (
            acting_is_group_admin
            and not acting_is_superuser
            and user_id != current_user.id
            and target_type == UserTypeEnum.PLAYER
            and db_user.group_id == current_user.group_id
        )

        if (
            user_id != current_user.id
            and not acting_is_superuser
            and not is_group_admin_managing_player
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        if is_group_admin_managing_player:
            if user_update.group_id is not None and user_update.group_id != current_user.group_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins cannot change a player's group",
                )

            if user_update.user_type and self._normalize_type(user_update.user_type) != UserTypeEnum.PLAYER:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins can only manage players",
                )

            if user_update.is_superuser is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins cannot modify system permissions",
                )

            if user_update.is_active is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Group admins cannot change activation status",
                )

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

            if user_update.password:
                db_user.hashed_password = get_password_hash(user_update.password)

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

        if acting_is_superuser and user_id != current_user.id:
            if target_type != UserTypeEnum.GROUP_ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System administrators can only manage group admin users",
                )

            normalized_update_type = (
                self._normalize_type(user_update.user_type)
                if user_update.user_type is not None
                else None
            )
            if normalized_update_type and normalized_update_type != UserTypeEnum.GROUP_ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System administrators can only assign the group admin user type",
                )

            if user_update.is_superuser is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System administrators cannot modify system permissions for group admins",
                )

            if user_update.is_active is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System administrators cannot change activation status for group admins",
                )

            if user_update.group_id is not None:
                self._validate_group_assignment(user_update.group_id, UserTypeEnum.GROUP_ADMIN)

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

        if user_update.is_active is not None and acting_is_superuser:
            db_user.is_active = user_update.is_active

        previous_group_id = db_user.group_id
        previous_type = target_type

        proposed_group_id = (
            user_update.group_id
            if user_update.group_id is not None
            else db_user.group_id
        )

        desired_type = self._resolve_user_type(
            user_type=user_update.user_type,
            fallback=target_type,
            assume_superuser=bool(
                user_update.is_superuser
                if user_update.is_superuser is not None
                else db_user.is_superuser
            ),
        )

        group, normalized_group_id = self._validate_group_assignment(proposed_group_id, desired_type)

        if not acting_is_superuser:
            if normalized_group_id != previous_group_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change group",
                )

            if desired_type != previous_type:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change user type",
                )

            if user_update.is_superuser is not None and user_update.is_superuser != db_user.is_superuser:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change system privileges",
                )

        # Prevent removing the last group admin from a group via update
        if previous_type == UserTypeEnum.GROUP_ADMIN:
            changing_group = normalized_group_id != previous_group_id
            losing_admin_role = desired_type != UserTypeEnum.GROUP_ADMIN
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

        db_user.group_id = normalized_group_id
        db_user.user_type = desired_type
        db_user.is_superuser = desired_type == UserTypeEnum.SYSTEM_ADMIN

        if user_update.password:
            db_user.hashed_password = get_password_hash(user_update.password)

        if desired_type == UserTypeEnum.GROUP_ADMIN and group and (group.admin_id is None or group.admin_id == db_user.id):
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
        db_user = self.get_user(user_id)
        acting_type = self._get_user_type(current_user)
        acting_is_superuser = acting_type == UserTypeEnum.SYSTEM_ADMIN
        acting_is_group_admin = acting_type == UserTypeEnum.GROUP_ADMIN
        user_type = self._get_user_type(db_user)
        is_self_delete = user_id == current_user.id

        if acting_is_superuser and not is_self_delete and user_type != UserTypeEnum.GROUP_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System administrators can only delete group admin users",
            )

        can_group_admin_delete = (
            acting_is_group_admin
            and not acting_is_superuser
            and user_id != current_user.id
            and user_type == UserTypeEnum.PLAYER
            and db_user.group_id == current_user.group_id
        )

        if (
            user_id != current_user.id
            and not acting_is_superuser
            and not can_group_admin_delete
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        if can_group_admin_delete and replacement_admin_id is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Replacement admin is not required when deleting a player",
            )

        promoted_user: Optional[models.User] = None

        try:
            if user_type == UserTypeEnum.SYSTEM_ADMIN:
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

                    replacement_user.user_type = UserTypeEnum.SYSTEM_ADMIN
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
