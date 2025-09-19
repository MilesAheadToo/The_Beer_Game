import uuid
from datetime import datetime, timedelta
from typing import Optional, Any, Union, Callable, Awaitable, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from sqlalchemy.future import select

from app.models.user import User, UserTypeEnum
from app.models.group import Group
from app.repositories.users import get_user_by_email

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False  # Don't auto-raise 401 for missing token
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    subject: str,
    user_type: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token with standard claims.
    
    Args:
        subject: The subject of the token (usually user ID or email)
        user_type: Serialized user type string
        expires_delta: Optional timedelta for token expiration
        
    Returns:
        str: Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),
        "type": "access",
        "user_type": user_type,
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        dict: The decoded token payload
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_token_from_request(request: Request, auth_header_token: Optional[str] = None) -> Optional[str]:
    """
    Get the token from either the Authorization header or cookie.
    
    Priority:
    1. Authorization header (Bearer token)
    2. Access token cookie
    
    Args:
        request: The incoming request
        auth_header_token: Token from Authorization header (if any)
        
    Returns:
        str or None: The JWT token if found, else None
    """
    # Check Authorization header first
    if auth_header_token:
        scheme, token = get_authorization_scheme_param(auth_header_token)
        if scheme.lower() == "bearer":
            return token
    
    # Fall back to cookie
    token = request.cookies.get(settings.COOKIE_ACCESS_TOKEN_NAME)
    if token:
        return token
        
    return None


async def get_current_user(
    request: Request, 
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current user from the JWT token in either cookie or header.
    
    Args:
        request: The incoming request
        token: JWT token from header or cookie
        db: Database session
        
    Returns:
        User: The authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    # Get token from header or cookie
    token = get_token_from_request(request, token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode the token
        payload = decode_token(token)
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Get user from database
        user = await get_user_by_email(db, email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
            
        token_user_type = payload.get("user_type")
        if token_user_type:
            try:
                user.user_type = UserTypeEnum(token_user_type)
            except ValueError:
                user.user_type = UserTypeEnum.PLAYER
        elif getattr(user, "user_type", None) is None:
            user.user_type = UserTypeEnum.SYSTEM_ADMIN if user.is_superuser else UserTypeEnum.PLAYER

        if (
            getattr(user, "group_id", None) in (None, 0)
            and user.user_type == UserTypeEnum.GROUP_ADMIN
        ):
            admin_group = getattr(user, "admin_of_group", None)
            if not admin_group:
                admin_group_result = await db.execute(
                    select(Group).filter(Group.admin_id == user.id)
                )
                admin_group = admin_group_result.scalars().first()
            if admin_group:
                user.group_id = admin_group.id

        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def set_auth_cookies(
    response: Response, 
    access_token: str, 
    token_type: str = "Bearer"
) -> None:
    """
    Set authentication cookies on the response.
    
    Args:
        response: The FastAPI response object
        access_token: The JWT access token
        token_type: The token type (default: "Bearer")
    """
    response.set_cookie(
        key=settings.COOKIE_ACCESS_TOKEN_NAME,
        value=access_token,
        httponly=settings.COOKIE_HTTP_ONLY,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAME_SITE,
        domain=settings.COOKIE_DOMAIN,
        path=settings.COOKIE_PATH,
        max_age=settings.COOKIE_MAX_AGE,
    )
    response.set_cookie(
        key=settings.COOKIE_TOKEN_TYPE_NAME,
        value=token_type,
        httponly=False,  # Allow JavaScript to read this
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAME_SITE,
        domain=settings.COOKIE_DOMAIN,
        path=settings.COOKIE_PATH,
        max_age=settings.COOKIE_MAX_AGE,
    )


def clear_auth_cookies(response: Response) -> None:
    """
    Clear authentication cookies from the response.
    
    Args:
        response: The FastAPI response object
    """
    response.delete_cookie(
        key=settings.COOKIE_ACCESS_TOKEN_NAME,
        path=settings.COOKIE_PATH,
        domain=settings.COOKIE_DOMAIN,
    )
    response.delete_cookie(
        key=settings.COOKIE_TOKEN_TYPE_NAME,
        path=settings.COOKIE_PATH,
        domain=settings.COOKIE_DOMAIN,
    )


def verify_password_strength(password: str) -> bool:
    """Verify that a password meets strength requirements.
    
    Args:
        password: The password to check
        
    Returns:
        bool: True if password meets requirements, False otherwise
    """
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT refresh token with standard claims.
    
    Args:
        user_id: The user's unique identifier
        expires_delta: Optional time delta for token expiration
        
    Returns:
        str: Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
        
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "type": "refresh"
    }
    
    return jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )

def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with standard claims.
    
    Args:
        user_id: The user's unique identifier
        expires_delta: Optional timedelta for token expiration
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = {
        "sub": str(user_id),  # Subject (whom the token refers to)
        "iat": datetime.utcnow(),  # Issued at
        "jti": str(uuid.uuid4()),  # Unique identifier for the token
    }
    
    # Set expiration (default 15 minutes if not specified)
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    
    # Create token with the secret key
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

async def get_token_from_cookie_or_header(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[str]:
    """Get token from either cookie or Authorization header.
    
    Priority:
    1. Authorization header (Bearer token)
    2. Access token cookie
    
    Args:
        request: The incoming request
        token: Token from Authorization header (if any)
        
    Returns:
        str or None: The JWT token if found, else None
    """
    # Try to get token from Authorization header first
    if token:
        return token
        
    # Then try to get from cookie
    token = request.cookies.get(settings.ACCESS_TOKEN_COOKIE_NAME)
    if token:
        return token
        
    return None

async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(get_token_from_cookie_or_header)
) -> Dict[str, Any]:
    """Get the current user from the JWT token in either cookie or header.
    
    Args:
        request: The incoming request
        db: Database session
        token: JWT token from cookie or header
        
    Returns:
        dict: The authenticated user data
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False}
        )
        
        # Get user_id from token
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception
            
        # Get token type (access or refresh)
        token_type: str = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except JWTError as e:
        raise credentials_exception from e
        
    # Fetch user from database
    from app.models.user import User
    user = db.query(User).filter(User.id == int(user_id)).first()
    
    if not user:
        raise credentials_exception
        
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
        
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get the current active user."""
    # TODO: Add any additional checks for active status, etc.
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def create_auth_response(
    user_id: str,
    response: Response,
    remember_me: bool = False
) -> Dict[str, str]:
    """Create an authentication response with tokens in cookies.
    
    Args:
        user_id: The user's ID
        response: FastAPI Response object to set cookies
        remember_me: Whether to use longer expiration for the refresh token
        
    Returns:
        dict: User data to include in the response body
    """
    # Create tokens
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    
    # Calculate token expiration times
    access_token_expires = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    refresh_token_expires = (settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)  # in seconds
    
    # Set access token cookie (httpOnly, secure)
    response.set_cookie(
        key=settings.ACCESS_TOKEN_COOKIE_NAME,
        value=f"{settings.TOKEN_PREFIX} {access_token}",
        max_age=access_token_expires,
        expires=access_token_expires,
        path="/",
        domain=settings.COOKIE_DOMAIN,
        secure=settings.COOKIE_SECURE,
        httponly=settings.COOKIE_HTTP_ONLY,
        samesite=settings.COOKIE_SAME_SITE
    )
    
    # Set refresh token cookie (httpOnly, secure, longer expiration)
    response.set_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token,
        max_age=refresh_token_expires if remember_me else None,  # Session cookie if not remember_me
        expires=refresh_token_expires if remember_me else None,
        path=f"{settings.API_V1_STR}/auth/refresh",  # Only sent for refresh endpoint
        domain=settings.COOKIE_DOMAIN,
        secure=settings.COOKIE_SECURE,
        httponly=True,  # Always httpOnly for refresh token
        samesite=settings.COOKIE_SAME_SITE
    )
    
    # Set CSRF token cookie (not httpOnly, used by frontend)
    csrf_token = set_csrf_cookie(
        response=response,
        expires_seconds=settings.CSRF_EXPIRE_SECONDS
    )
    
    # Return user data (without sensitive info)
    return {
        "user": {
            "id": user_id,
            # Add other non-sensitive user fields here
        },
        "access_token": access_token,  # For clients that need it in the response body
        "token_type": "bearer"
    }
