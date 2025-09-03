from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime

class UserBase(BaseModel):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=50)
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    password: Optional[str] = Field(None, min_length=8, max_length=50)
    is_active: Optional[bool] = None

class UserInDBBase(UserBase):
    """Base schema for user data in the database."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class User(UserInDBBase):
    """Schema for returning user data (without sensitive information)."""
    pass

class UserInDB(UserInDBBase):
    """Schema for user data stored in the database (includes hashed password)."""
    hashed_password: str

class Token(BaseModel):
    """Schema for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for token payload data."""
    username: Optional[str] = None

class UserWithGames(User):
    """Schema for user data including their games."""
    games: List[dict] = []

class UserPasswordChange(BaseModel):
    """Schema for changing a user's password."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=50)
    
    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v


class PasswordResetRequest(BaseModel):
    """Schema for requesting a password reset."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for confirming a password reset."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=50)
    
    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v


class MFASetupResponse(BaseModel):
    """Response schema for MFA setup."""
    secret: str
    qr_code_uri: str
    recovery_codes: List[str]


class MFAVerifyRequest(BaseModel):
    """Request schema for MFA verification."""
    code: str


class MFARecoveryCodes(BaseModel):
    """Schema for MFA recovery codes."""
    recovery_codes: List[str]
