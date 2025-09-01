from pydantic import BaseSettings, Field
from typing import List, Optional, Any, Dict, Union
from datetime import timedelta
import secrets
import os

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Beer Game API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30  # 30 days
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./beer_game.db"
    
    # Remove MySQL configuration to prevent any accidental usage
    MYSQL_SERVER: str = ""
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = ""
    
    # WebSocket
    WEBSOCKET_PATH: str = "/ws"
    WEBSOCKET_PING_INTERVAL: int = 25  # seconds
    WEBSOCKET_PING_TIMEOUT: int = 5    # seconds
    
    # Game Settings
    INITIAL_INVENTORY: int = 12
    HOLDING_COST_PER_UNIT: float = 0.5
    BACKORDER_COST_PER_UNIT: float = 1.0
    DEFAULT_MAX_ROUNDS: int = 52
    
    # AI Settings
    AI_REACTION_TIME: float = 1.0  # seconds to wait before AI makes a move
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == "BACKEND_CORS_ORIGINS" and raw_val:
                if isinstance(raw_val, str) and not raw_val.startswith("["):
                    return [origin.strip() for origin in raw_val.split(",")]
                elif isinstance(raw_val, list):
                    return raw_val
            return cls.json_loads(raw_val)

def get_settings() -> Settings:
    """Get the application settings.
    
    This function ensures that the database URI is properly set up.
    """
    settings = Settings()
    
    # Ensure we're using SQLite
    if not settings.SQLALCHEMY_DATABASE_URI:
        settings.SQLALCHEMY_DATABASE_URI = "sqlite:///./beer_game.db"
    
    # Make sure the database URL is using SQLite
    if not settings.SQLALCHEMY_DATABASE_URI.startswith("sqlite"):
        print(f"Warning: Using non-SQLite database: {settings.SQLALCHEMY_DATABASE_URI}")
        print("Falling back to SQLite for this session.")
        settings.SQLALCHEMY_DATABASE_URI = "sqlite:///./beer_game.db"
    
    return settings

# Global settings instance
settings = get_settings()

# JWT token configuration
def create_access_token(
    subject: Union[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: The subject of the token (usually user ID or username)
        expires_delta: Optional timedelta for token expiration
        
    Returns:
        str: Encoded JWT token
    """
    from datetime import datetime, timezone
    import jwt
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
