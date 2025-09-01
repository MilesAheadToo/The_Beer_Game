from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any

from app.core.deps import get_db
from app.schemas.user import UserCreate, User, Token, UserUpdate, UserPasswordChange
from app.services.auth_service import AuthService, get_current_active_user

router = APIRouter()

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """Dependency to get an instance of AuthService."""
    return AuthService(db)

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(
    user_in: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Register a new user.
    
    - **username**: Must be unique, 3-50 characters, alphanumeric with underscores or hyphens
    - **email**: Must be a valid email address
    - **password**: Must be at least 8 characters, with at least one uppercase, one lowercase, and one number
    - **full_name**: Optional full name
    """
    try:
        return auth_service.create_user(user_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    - **username**: Your username or email
    - **password**: Your password
    
    Returns an access token and refresh token.
    """
    try:
        print(f"Attempting to authenticate user: {form_data.username}")
        user = auth_service.authenticate_user(form_data.username, form_data.password)
        if not user:
            print(f"Authentication failed for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        print(f"User {user.id} authenticated successfully")
        return auth_service.create_access_token(user)
    except Exception as e:
        print(f"Error during login: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during login: {str(e)}",
        )

@router.post("/refresh-token", response_model=Token)
def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Refresh an access token using a refresh token.
    
    - **refresh_token**: A valid refresh token
    
    Returns a new access token and the same refresh token.
    """
    try:
        return auth_service.refresh_access_token(refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me", response_model=User)
def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get current user information.
    
    Requires authentication.
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "is_superuser": current_user.is_superuser,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }

@router.put("/me", response_model=User)
def update_user_me(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Update current user information.
    
    - **email**: New email (optional)
    - **full_name**: New full name (optional)
    - **is_active**: Whether the user is active (admin only)
    
    Requires authentication.
    """
    return auth_service.update_user(current_user.id, user_in)

@router.post("/change-password", response_model=User)
def change_password(
    password_change: UserPasswordChange,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Change the current user's password.
    
    - **current_password**: Current password
    - **new_password**: New password (must be at least 8 characters, with at least one uppercase, one lowercase, and one number)
    
    Requires authentication.
    """
    try:
        return auth_service.change_password(
            current_user.id,
            password_change.current_password,
            password_change.new_password
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# Admin-only endpoints
@router.get("/users/", response_model=list[User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Retrieve all users (admin only).
    
    - **skip**: Number of users to skip (for pagination)
    - **limit**: Maximum number of users to return (for pagination)
    
    Requires admin privileges.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return auth_service.get_users(skip=skip, limit=limit)

@router.get("/users/{user_id}", response_model=User)
def read_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Get a specific user by ID (admin only).
    
    - **user_id**: ID of the user to retrieve
    
    Requires admin privileges.
    """
    if not current_user.is_superuser and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    user = auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user
