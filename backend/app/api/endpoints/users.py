from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Any, Optional
from sqlalchemy.orm import Session
from ... import models
from ...schemas.user import User, UserCreate, UserUpdate, UserInDB, UserPasswordChange
from ...db.session import get_db
from ...core.security import get_current_active_user
from ...services.user_service import UserService

router = APIRouter()

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency to get an instance of UserService."""
    return UserService(db)

@router.get("/", response_model=List[User])
async def read_users(
    skip: int = 0, 
    limit: int = 100,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Retrieve all users (admin only).
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user_service.get_users(skip=skip, limit=limit)

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Create new user (admin only).
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user_service.create_user(user_in)

@router.delete("/{user_id}", response_model=dict)
async def delete_user(
    user_id: int,
    replacement_admin_id: Optional[int] = None,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Delete a user (admin only).
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user_service.delete_user(user_id, current_user, replacement_admin_id)

@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Update a user (admin or self).
    """
    return user_service.update_user(user_id, user_in, current_user)

@router.post("/{user_id}/change-password", response_model=dict)
async def change_password(
    user_id: int,
    password_change: UserPasswordChange,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Change a user's password.
    """
    return user_service.change_password(
        user_id=user_id,
        current_password=password_change.current_password,
        new_password=password_change.new_password,
        current_user=current_user
    )

@router.get("/me", response_model=User)
async def read_user_me(
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Get current user.
    """
    return current_user

@router.get("/{user_id}", response_model=User)
async def read_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Get a specific user by ID (admin only).
    """
    if not current_user.is_superuser and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user_service.get_user(user_id)
