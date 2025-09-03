from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any, List, Optional

from app.core.config import settings
from app.core.deps import get_db, get_current_active_user
from app.schemas.user import (
    UserCreate, User, Token, UserUpdate, UserPasswordChange,
    MFASetupResponse, MFAVerifyRequest, MFARecoveryCodes
)
from app.services.auth_service import AuthService

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
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    - **username**: Your username or email
    - **password**: Your password
    
    Returns an access token and refresh token. If MFA is enabled for the user,
    returns a temporary token that can only be used for MFA verification.
    """
    try:
        # Get client IP for rate limiting and logging
        client_ip = request.client.host if request.client else "unknown"
        print(f"Attempting to authenticate user: {form_data.username}")
        user = auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password,
            request=request
        )
        if not user:
            print(f"Authentication failed for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        print(f"User {user.id} authenticated successfully")
        
        # If MFA is enabled, return a temporary token that can only be used for MFA verification
        if user.mfa_enabled and user.mfa_secret:
            # Create a temporary token with a short expiration
            temp_token = auth_service.create_access_token(user, mfa_verified=False)
            return Token(
                access_token=temp_token.access_token,
                refresh_token=temp_token.refresh_token,
                token_type="bearer",
                requires_mfa=True
            )
            
        return auth_service.create_access_token(user, mfa_verified=True)
    except Exception as e:
        print(f"Error during login: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during login: {str(e)}",
        )

@router.post("/mfa/setup", response_model=MFASetupResponse)
def setup_mfa(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Set up Multi-Factor Authentication for the current user.
    
    Generates a new MFA secret and returns it along with a provisioning URI
    that can be used with authenticator apps like Google Authenticator.
    
    Requires authentication.
    """
    if current_user.mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is already enabled for this account"
        )
        
    # Generate a new MFA secret
    secret = auth_service.generate_mfa_secret(current_user)
    
    # Generate a provisioning URI for the authenticator app
    uri = auth_service.generate_mfa_uri(current_user, secret)
    
    # Generate recovery codes
    recovery_codes = auth_service.generate_recovery_codes(current_user)
    
    return {
        "secret": secret,
        "uri": uri,
        "recovery_codes": recovery_codes
    }

@router.post("/mfa/verify", response_model=Token)
def verify_mfa(
    mfa_verify: MFAVerifyRequest,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Verify an MFA code and complete MFA setup.
    
    - **code**: The MFA code from the authenticator app
    
    Returns a new access token and refresh token if verification is successful.
    
    Requires authentication and an unverified MFA setup.
    """
    if current_user.mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is already enabled for this account"
        )
        
    if not current_user.mfa_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not set up for this account"
        )
    
    # Verify the MFA code
    if not auth_service.enable_mfa(current_user, mfa_verify.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid MFA code"
        )
    
    # Generate new tokens with MFA verified
    return auth_service.create_access_token(current_user, mfa_verified=True)

@router.post("/mfa/disable", status_code=status.HTTP_204_NO_CONTENT)
def disable_mfa(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """
    Disable MFA for the current user.
    
    Requires authentication and MFA to be enabled.
    """
    if not current_user.mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not enabled for this account"
        )
    
    auth_service.disable_mfa(current_user)

@router.post("/mfa/recovery-codes", response_model=MFARecoveryCodes)
def generate_recovery_codes(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Generate new MFA recovery codes.
    
    This will invalidate any previously generated recovery codes.
    
    Requires authentication and MFA to be enabled.
    """
    if not current_user.mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not enabled for this account"
        )
    
    recovery_codes = auth_service.generate_recovery_codes(current_user)
    return {"recovery_codes": recovery_codes}

@router.post("/mfa/verify-recovery", response_model=Token)
def verify_recovery_code(
    mfa_verify: MFAVerifyRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> Any:
    """
    Verify a recovery code and get a new access token.
    
    - **code**: The recovery code to verify
    
    Returns a new access token and refresh token if the recovery code is valid.
    """
    # Get user from the recovery code
    user = auth_service.verify_recovery_code(mfa_verify.code)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired recovery code"
        )
    
    # Generate new tokens with MFA verified
    return auth_service.create_access_token(user, mfa_verified=True)

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
