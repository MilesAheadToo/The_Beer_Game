from pydantic import BaseSettings, PostgresDsn
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Beer Game API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    MYSQL_SERVER: str = "AIServer"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "19890617"
    MYSQL_DB: str = "beer_game"
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    
    class Config:
        case_sensitive = True

settings = Settings()
settings.SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_SERVER}/{settings.MYSQL_DB}"
