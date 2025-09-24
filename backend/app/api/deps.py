"""FastAPI dependency helpers for legacy synchronous endpoints."""

from __future__ import annotations

from typing import Generator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.security import decode_token, get_token_from_request
from app.db.session import sync_engine
from app.models.user import User, UserTypeEnum

_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)

SessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a synchronous SQLAlchemy session for endpoints that expect it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _coerce_user_type(user: User, token_payload: dict) -> None:
    token_type = token_payload.get("user_type")
    if token_type:
        try:
            user.user_type = UserTypeEnum(token_type)
        except ValueError:
            user.user_type = UserTypeEnum.PLAYER
    elif getattr(user, "user_type", None) is None:
        user.user_type = UserTypeEnum.SYSTEM_ADMIN if getattr(user, "is_superuser", False) else UserTypeEnum.PLAYER


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user using the cookie/header token against the sync session."""
    raw_token = get_token_from_request(request, token)
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(raw_token)
    except HTTPException:
        # re-raise FastAPI HTTP errors unchanged
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    _coerce_user_type(user, payload)
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the resolved user account is active."""
    if not getattr(current_user, "is_active", True):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user
