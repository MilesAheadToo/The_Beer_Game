import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings, create_access_token
from app.models.user import User, RefreshToken
from app.schemas.user import UserCreate, UserInDB, Token, TokenData
from app.core.security import get_password_hash, verify_password, oauth2_scheme
from app.core.deps import get_db

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Service for handling authentication and user management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user with hashed password."""
        # Check if user already exists
        db_user = self.get_user_by_email(user_create.email)
        if db_user:
            raise ValueError("Email already registered")
            
        # Create new user
        hashed_password = get_password_hash(user_create.password)
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password,
            full_name=user_create.full_name,
            is_active=True
        )
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username/email and password.
        
        Args:
            username: Can be either username or email
            password: Plain text password
            
        Returns:
            User object if authentication succeeds, None otherwise
        """
        # First try to find user by email
        user = self.get_user_by_email(username)
        
        # If not found by email, try by username
        if not user:
            user = self.get_user_by_username(username)
            
        if not user:
            return None
            
        if not verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(self, user: User) -> Token:
        """Create access and refresh tokens for a user.
        
        Args:
            user: The user object for which to create tokens
            
        Returns:
            Token: An object containing access_token, refresh_token, and token_type
        """
        from datetime import datetime, timedelta
        from jose import jwt
        
        # Create access token with email as subject
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + access_token_expires
        to_encode = {
            "exp": expire, 
            "sub": str(user.email),  # Using email as subject for better uniqueness
            "username": user.username  # Include username in the token if needed
        }
        access_token = jwt.encode(
            to_encode, 
            settings.SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        
        # Create refresh token
        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = self._create_refresh_token(user.id, refresh_token_expires)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token.token,
            token_type="bearer"
        )
    
    def refresh_access_token(self, refresh_token: str) -> Token:
        """Refresh an access token using a refresh token."""
        db_token = self.db.query(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.expires_at > datetime.utcnow()
        ).first()
        
        if not db_token:
            raise ValueError("Invalid refresh token")
        
        user = self.db.query(User).filter(User.id == db_token.user_id).first()
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=new_access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
    
    def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """Get the current user from a JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            # Decode the JWT token
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
                options={"require_exp": True}  # Ensure exp claim is present
            )
            
            # Get the email from the token
            email: str = payload.get("sub")
            if not email:
                raise credentials_exception
                
            # Get the user from the database
            user = self.get_user_by_email(email)
            if user is None:
                raise credentials_exception
                
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Inactive user"
                )
                
            return user
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def _create_refresh_token(self, user_id: int, expires_delta: timedelta) -> RefreshToken:
        """Create and store a refresh token in the database."""
        # Invalidate any existing refresh tokens for this user
        self.db.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()
        
        # Create new refresh token
        expires_at = datetime.utcnow() + expires_delta
        token = RefreshToken(
            user_id=user_id,
            token=secrets.token_urlsafe(32),
            expires_at=expires_at
        )
        
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

# Dependency to get the current active user
def get_current_active_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get the current active user from the JWT token."""
    auth_service = AuthService(db)
    return auth_service.get_current_user(token)
