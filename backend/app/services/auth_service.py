import secrets
import pyotp
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import Depends, HTTPException, status, Request
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.core.config import settings
from app.models.user import User, RefreshToken
from app.models.auth_models import PasswordHistory, PasswordResetToken
from app.schemas.user import (
    UserCreate, 
    UserInDB, 
    Token, 
    TokenData, 
    PasswordResetRequest,
    PasswordResetConfirm,
    MFASetupResponse,
    MFAVerifyRequest
)
from app.core.security import get_password_hash, verify_password, oauth2_scheme, verify_password_strength, create_access_token
from app.db.deps import get_db
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Service for handling authentication and user management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user with hashed password and security features."""
        # Check if user already exists
        db_user = self.get_user_by_email(user_create.email)
        if db_user:
            raise ValueError("Email already registered")
            
        # Check password strength
        if not verify_password_strength(user_create.password):
            raise ValueError("Password does not meet complexity requirements")
            
        # Create new user
        hashed_password = get_password_hash(user_create.password)
        now = datetime.utcnow()
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password,
            full_name=user_create.full_name,
            is_active=True,
            last_password_change=now,
            failed_login_attempts=0,
            is_locked=False,
            mfa_enabled=False
        )
        
        # Add initial password to history
        self._add_to_password_history(db_user.id, hashed_password)
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def authenticate_user(self, username: str, password: str, request: Request = None) -> Optional[User]:
        """
        Authenticate a user with username/email and password with security features.
        
        Args:
            username: Can be either username or email
            password: Plain text password
            request: FastAPI request object for IP tracking
            
        Returns:
            User object if authentication succeeds, None otherwise
            
        Raises:
            HTTPException: If account is locked or other security violations
        """
        # Try to find user by email or username
        user = self.db.query(User).filter(
            or_(User.email == username, User.username == username)
        ).first()
        
        if not user:
            # User not found, but don't reveal that to prevent user enumeration
            return None
            
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked due to too many failed login attempts"
            )
            
        # Verify password
        if not verify_password(password, user.hashed_password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            
            # Check if account should be locked
            if user.failed_login_attempts >= settings.PASSWORD_MAX_ATTEMPTS:
                user.is_locked = True
                user.lockout_until = datetime.utcnow() + timedelta(minutes=settings.PASSWORD_LOCKOUT_MINUTES)
                
            self.db.commit()
            return None
            
        # Reset failed login attempts on successful login
        if user.failed_login_attempts > 0:
            user.failed_login_attempts = 0
            user.is_locked = False
            user.lockout_until = None
            
        # Update last login time
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        # Return the user object to be used by the login endpoint
        return user
    
    def create_access_token(self, user: User, mfa_verified: bool = False) -> dict:
        """Create access and refresh tokens for a user with enhanced security.
        
        Args:
            user: The user object for which to create tokens
            mfa_verified: Whether MFA has been verified for this session
            
        Returns:
            dict: A dictionary containing access_token, refresh_token, and user details
        """
        # Create access token with standard claims
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            user_id=str(user.id),
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = secrets.token_urlsafe(32)
        self._create_refresh_token(
            user.id,
            refresh_token_expires,
            refresh_token
        )
        
        # Prepare user data to return
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_superuser": user.is_superuser,
            "is_active": user.is_active,
            "mfa_enabled": user.mfa_enabled,
            "mfa_verified": mfa_verified
        }
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(access_token_expires.total_seconds()),
            "refresh_token": refresh_token,
            "user": user_data
        }
    
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
        
    def _create_refresh_token(self, user_id: int, expires_delta: timedelta, jti: str = None) -> RefreshToken:
        """Create and store a refresh token in the database."""
        expires = datetime.utcnow() + expires_delta
        token = secrets.token_urlsafe(64)  # Longer token for refresh tokens
        db_token = RefreshToken(
            token=token,
            user_id=user_id,
            expires_at=expires,
            created_at=datetime.utcnow()
        )
        self.db.add(db_token)
        self.db.commit()
        return db_token
        
    def _add_to_password_history(self, user_id: int, hashed_password: str) -> None:
        """Add a password to the user's password history.
        
        Args:
            user_id: The ID of the user
            hashed_password: The hashed password to add to history
        """
        # Get the current password history count
        history_count = self.db.query(PasswordHistory).filter_by(user_id=user_id).count()
        
        # If we've reached the maximum history size, remove the oldest entry
        if history_count >= settings.PASSWORD_HISTORY_SIZE:
            oldest = self.db.query(PasswordHistory).filter_by(user_id=user_id)\
                .order_by(PasswordHistory.created_at.asc()).first()
            if oldest:
                self.db.delete(oldest)
        
        # Add the new password to history
        password_history = PasswordHistory(
            user_id=user_id,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        self.db.add(password_history)
        self.db.commit()
    
    def is_password_in_history(self, user_id: int, password: str) -> bool:
        """Check if the given password is in the user's password history.
        
        Args:
            user_id: The ID of the user
            password: The password to check
            
        Returns:
            bool: True if password is in history, False otherwise
        """
        # Get all password history entries for the user
        history = self.db.query(PasswordHistory).filter_by(user_id=user_id).all()
        
        # Check if the password matches any in history
        return any(pwd_context.verify(password, entry.hashed_password) for entry in history)
    
    def change_password(self, user: User, current_password: str, new_password: str) -> bool:
        """Change a user's password with security checks.
        
        Args:
            user: The user changing their password
            current_password: The user's current password
            new_password: The new password
            
        Returns:
            bool: True if password was changed successfully, False otherwise
            
        Raises:
            ValueError: If the current password is incorrect or new password is invalid
        """
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")
            
        # Check if new password is the same as current
        if verify_password(new_password, user.hashed_password):
            raise ValueError("New password must be different from current password")
            
        # Check password strength
        if not verify_password_strength(new_password):
            raise ValueError("New password does not meet complexity requirements")
            
        # Check if password was used before
        if self.is_password_in_history(user.id, new_password):
            raise ValueError("You cannot reuse a previous password")
            
        # Update password and add to history
        new_hashed_password = get_password_hash(new_password)
        user.hashed_password = new_hashed_password
        user.last_password_change = datetime.utcnow()
        
        # Add to password history
        self._add_to_password_history(user.id, new_hashed_password)
        
        # Save changes
        self.db.commit()
        return True
        
    def generate_mfa_secret(self, user: User) -> str:
        """Generate a new MFA secret for a user.
        
        Args:
            user: The user to generate the MFA secret for
            
        Returns:
            str: The generated MFA secret
        """
        # Generate a new secret
        secret = pyotp.random_base32()
        
        # Store the secret in the user's record
        user.mfa_secret = secret
        user.mfa_enabled = False  # Not enabled until verified
        self.db.commit()
        
        return secret
        
    def generate_mfa_uri(self, user: User, secret: str) -> str:
        """Generate a provisioning URI for the MFA app.
        
        Args:
            user: The user to generate the URI for
            secret: The MFA secret
            
        Returns:
            str: The provisioning URI
        """
        return pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name=settings.PROJECT_NAME
        )
        
    def verify_mfa_code(self, user: User, code: str) -> bool:
        """Verify an MFA code for a user.
        
        Args:
            user: The user to verify the code for
            code: The MFA code to verify
            
        Returns:
            bool: True if the code is valid, False otherwise
        """
        if not user.mfa_secret:
            return False
            
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(code, valid_window=1)  # Allow 30s before/after for clock skew
        
    def enable_mfa(self, user: User, code: str) -> bool:
        """Enable MFA for a user after verifying their first code.
        
        Args:
            user: The user to enable MFA for
            code: The MFA code to verify
            
        Returns:
            bool: True if MFA was enabled, False if the code was invalid
        """
        if not user.mfa_secret:
            return False
            
        if not self.verify_mfa_code(user, code):
            return False
            
        user.mfa_enabled = True
        self.db.commit()
        return True
        
    def disable_mfa(self, user: User) -> None:
        """Disable MFA for a user.
        
        Args:
            user: The user to disable MFA for
        """
        user.mfa_enabled = False
        user.mfa_secret = None
        self.db.commit()
        
    def generate_recovery_codes(self, user: User, count: int = 10) -> List[str]:
        """Generate recovery codes for MFA.
        
        Args:
            user: The user to generate recovery codes for
            count: Number of recovery codes to generate (default: 10)
            
        Returns:
            List[str]: List of recovery codes
        """
        codes = [secrets.token_urlsafe(16) for _ in range(count)]
        user.mfa_recovery_codes = codes
        self.db.commit()
        return codes
        
    def verify_recovery_code(self, user: User, code: str) -> bool:
        """Verify a recovery code and remove it if valid.
        
        Args:
            user: The user to verify the recovery code for
            code: The recovery code to verify
            
        Returns:
            bool: True if the recovery code was valid, False otherwise
        """
        if not user.mfa_recovery_codes:
            return False
            
        if code in user.mfa_recovery_codes:
            # Remove the used code
            user.mfa_recovery_codes = [c for c in user.mfa_recovery_codes if c != code]
            self.db.commit()
            return True
            
        return False

# Dependency to get the current active user
def get_current_active_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get the current active user from the JWT token."""
    auth_service = AuthService(db)
    return auth_service.get_current_user(token)
