from pydantic import BaseSettings, Field, validator
from typing import List, Optional, Any, Dict, Union, Set
from datetime import timedelta
import secrets
import os
import json
from urllib.parse import urlparse

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Beer Game API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    REFRESH_SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30  # 30 days
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Cookie settings
    COOKIE_SECURE: bool = False  # Set to False for development without HTTPS
    COOKIE_HTTP_ONLY: bool = True
    COOKIE_SAME_SITE: str = "lax"  # or 'strict', 'none'
    COOKIE_DOMAIN: Optional[str] = None
    COOKIE_ACCESS_TOKEN_NAME: str = "access_token"
    COOKIE_TOKEN_TYPE_NAME: str = "token_type"
    COOKIE_PATH: str = "/"
    COOKIE_MAX_AGE: int = 60 * 60 * 24 * 7  # 7 days
    
    # JWT Token settings
    ACCESS_TOKEN_COOKIE_NAME: str = "access_token"
    REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
    TOKEN_PREFIX: str = "Bearer"
    
    # CSRF settings
    CSRF_COOKIE_NAME: str = "csrf_token"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    CSRF_TOKEN_LENGTH: int = 32
    CSRF_COOKIE_SECURE: bool = True
    CSRF_COOKIE_HTTP_ONLY: bool = False  # Must be accessible from JS
    CSRF_COOKIE_SAME_SITE: str = "lax"
    CSRF_COOKIE_PATH: str = "/"
    CSRF_EXPIRE_SECONDS: int = 60 * 60 * 24 * 7  # 7 days
    
    # Password settings
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_MAX_ATTEMPTS: int = 5
    PASSWORD_LOCKOUT_MINUTES: int = 15
    PASSWORD_HISTORY_SIZE: int = 5
    MAX_LOGIN_ATTEMPTS: int = 5  # Maximum number of failed login attempts before lockout
    
    # Session settings
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_SESSIONS_PER_USER: int = 5
    
    # Rate limiting
    RATE_LIMIT: str = "100/minute"
    AUTH_RATE_LIMIT: str = "5/minute"
    
    # Security headers
    SECURE_HEADERS: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Frontend URL for CORS and redirects
    FRONTEND_URL: str = "http://localhost:3000"
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    
    # Refresh cookie config
    REFRESH_COOKIE_NAME: str = "refresh_token"
    REFRESH_COOKIE_PATH: str = "/api/v1/auth"
    REFRESH_COOKIE_DOMAIN: Optional[str] = None  # Set to your domain in production
    REFRESH_COOKIE_SAMESITE: str = "lax"  # 'lax' for production, 'none' for cross-origin
    REFRESH_COOKIE_SECURE: bool = False  # Set to True in production (HTTPS)
    REFRESH_COOKIE_HTTPONLY: bool = True
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI: str = ""
    
    class Config:
        env_file = ".env"
        
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if v is not None and v != "":
            return v
            
        server = os.getenv("MYSQL_SERVER", "db")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "beer_user")
        password = os.getenv("MYSQL_PASSWORD", "beer_password")
        db = os.getenv("MYSQL_DB", "beer_game")
        
        # URL encode the password to handle special characters
        from urllib.parse import quote_plus
        encoded_password = quote_plus(password)
        # Connection string with SSL disabled and URL-encoded password
        return f"mysql+pymysql://{user}:{encoded_password}@{server}:{port}/{db}?charset=utf8mb4&ssl=None"
    
    # WebSocket
    WEBSOCKET_PATH: str = "/ws"
    WEBSOCKET_PING_INTERVAL: int = 25  # seconds
    WEBSOCKET_PING_TIMEOUT: int = 5    # seconds
    WEBSOCKET_MAX_MESSAGE_SIZE: int = 1024 * 1024  # 1MB
    
    # Game Settings
    INITIAL_INVENTORY: int = 12
    HOLDING_COST_PER_UNIT: float = 0.5
    BACKORDER_COST_PER_UNIT: float = 1.0
    DEFAULT_MAX_ROUNDS: int = 52
    
    # AI Settings
    AI_REACTION_TIME: float = 1.0  # seconds to wait before AI makes a move
    
    # Redis (for WebSocket message broker in production)
    REDIS_URL: Optional[str] = None
    
    # Rate limiting
    RATE_LIMIT: str = "100/minute"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS methods and headers
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    CORS_EXPOSE_HEADERS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @property
    def allowed_origins(self) -> Set[str]:
        """Get the set of allowed origins for CORS."""
        if "*" in self.BACKEND_CORS_ORIGINS:
            return {"*"}
        return {origin.strip() for origin in self.BACKEND_CORS_ORIGINS if origin.strip()}
    
    @property
    def is_production(self) -> bool:
        """Check if the application is running in production mode."""
        return self.ENVIRONMENT == "production"
    
    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL for the current environment."""
        if self.ENVIRONMENT == "production":
            return f"wss://{self.HOST}{self.WEBSOCKET_PATH}"
        return f"ws://{self.HOST}:{self.PORT}{self.WEBSOCKET_PATH}"
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            # Prioritize environment variables over .env file
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
            
        # Allow extra fields in the config
        extra = "ignore"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == "BACKEND_CORS_ORIGINS" and raw_val:
                if isinstance(raw_val, str) and not raw_val.startswith("["):
                    return [origin.strip() for origin in raw_val.split(",")]
                elif isinstance(raw_val, list):
                    return raw_val
            return cls.json_loads(raw_val)

def get_settings() -> Settings:
    """Get the application settings with MySQL configuration."""
    settings = Settings()
    # Use MySQL configuration from environment variables
    server = os.getenv("MYSQL_SERVER", "db")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "beer_user")
    password = os.getenv("MYSQL_PASSWORD", "beer_password")
    db = os.getenv("MYSQL_DB", "beer_game")
    
    settings.SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{user}:{password}@{server}:{port}/{db}?charset=utf8mb4"
    print(f"Using MySQL database at: {user}@{server}:{port}/{db}")
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
