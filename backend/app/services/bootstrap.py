"""Helpers to bootstrap a default group/game for developer convenience."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models.game import Game
from app.schemas.group import GroupCreate
from app.schemas.user import UserCreate

logger = logging.getLogger(__name__)

DEFAULT_GROUP_NAME = "Daybreak"
DEFAULT_GROUP_DESCRIPTION = "Default Daybreak group"
DEFAULT_GROUP_LOGO = "/daybreak_logo.png"
DEFAULT_ADMIN_USERNAME = "groupadmin"
DEFAULT_ADMIN_EMAIL = "groupadmin@daybreak.ai"
DEFAULT_ADMIN_FULL_NAME = "Group Administrator"
DEFAULT_ADMIN_PASSWORD = "Daybreak@2025"


def build_default_group_payload() -> GroupCreate:
    """Return the default Daybreak group payload used across the app."""
    return GroupCreate(
        name=DEFAULT_GROUP_NAME,
        description=DEFAULT_GROUP_DESCRIPTION,
        logo=DEFAULT_GROUP_LOGO,
        admin=UserCreate(
            username=DEFAULT_ADMIN_USERNAME,
            email=DEFAULT_ADMIN_EMAIL,
            password=DEFAULT_ADMIN_PASSWORD,
            full_name=DEFAULT_ADMIN_FULL_NAME,
            user_type="GROUP_ADMIN",
        ),
    )


def ensure_default_group_and_game(db: Session) -> Optional[Game]:
    """Ensure at least one game exists for developer-focused SQLite fallback."""
    existing_game = db.query(Game).first()
    if existing_game:
        return existing_game

    logger.info("Bootstrapping default group and game for local development")
    from .group_service import GroupService  # local import to avoid circular dependency

    service = GroupService(db)
    group = service.create_group(build_default_group_payload())
    db.refresh(group)
    return db.query(Game).first()
